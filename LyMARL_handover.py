import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

from env.basestation import BaseStation, SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    """
    Simple moving average for a 1D array.
    Returns an array with the same length as the input.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x

    out = np.zeros_like(x, dtype=np.float32)
    csum = 0.0
    for i in range(len(x)):
        csum += float(x[i])
        if i >= window:
            csum -= float(x[i - window])
            out[i] = csum / float(window)
        else:
            out[i] = csum / float(i + 1)
    return out


def block_avg_1d(x: np.ndarray, block: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: [T]
    returns:
      xs: [num_blocks]   -> end step index of each block
      ys: [num_blocks]   -> block mean
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = x.shape[0]

    if T == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

    xs, ys = [], []
    for start in range(0, T, block):
        chunk = x[start:start + block]
        if chunk.size == 0:
            continue
        xs.append(start + chunk.size)
        ys.append(float(chunk.mean()))

    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32)


# ============================================================
# Networks / Value Normalization
# ============================================================

class ValueNorm(nn.Module):
    """
    Running mean/std for scalar targets.
    Used for UE critic target normalization.
    """
    def __init__(self, eps: float = 1e-5, device: Optional[torch.device] = None):
        super().__init__()
        self.eps = eps
        self.device = device if device is not None else torch.device("cpu")
        self.register_buffer("count", torch.tensor(0.0, device=self.device))
        self.register_buffer("mean", torch.tensor(0.0, device=self.device))
        self.register_buffer("m2", torch.tensor(1.0, device=self.device))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().view(-1).to(self.device)
        if x.numel() == 0:
            return

        for v in x:
            self.count += 1.0
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2

    def variance(self):
        denom = torch.clamp(self.count - 1.0, min=1.0)
        return self.m2 / denom

    def std(self):
        return torch.sqrt(self.variance() + self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std() + self.mean


class ValueNormVec(nn.Module):
    """
    Running mean/std for vector targets: shape [..., D].
    Used for BS critic target normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-5, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.device = device if device is not None else torch.device("cpu")

        self.register_buffer("count", torch.zeros(self.dim, device=self.device))
        self.register_buffer("mean", torch.zeros(self.dim, device=self.device))
        self.register_buffer("m2", torch.ones(self.dim, device=self.device))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().to(self.device).view(-1, self.dim)
        if x.numel() == 0:
            return

        for i in range(x.shape[0]):
            v = x[i]
            self.count += 1.0
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2

    def variance(self):
        denom = torch.clamp(self.count - 1.0, min=1.0)
        return self.m2 / denom

    def std(self):
        return torch.sqrt(self.variance() + self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std() + self.mean


class UEActorNetwork(nn.Module):
    """
    Shared actor network for all UEs.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class BSActorNetwork(nn.Module):
    """
    Shared actor network for all BSs.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


class CentralizedCriticUE(nn.Module):
    """
    Centralized critic for UE objective.
    Outputs a scalar value.
    """
    def __init__(self, global_obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs):
        return self.net(global_obs).squeeze(-1)


class CentralizedCriticBS(nn.Module):
    """
    Centralized critic for BS objective.
    Outputs a vector of size [B].
    """
    def __init__(self, global_obs_dim: int, n_bs: int, hidden_dim: int = 256):
        super().__init__()
        self.n_bs = int(n_bs)
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.n_bs),
        )

    def forward(self, global_obs):
        return self.net(global_obs)


# ============================================================
# Environment
# ============================================================

class MAPPOEnvironment:
    """
    Environment for UE-BS heterogeneous MAPPO.

    - Training: soft constraint behavior only
    - Evaluation: optional hard constraint can be enabled
    """
    def __init__(
        self,
        base_stations: List[BaseStation],
        users: List[UserEquipment],
        V: float = 20.0,
        power_budget_ratio: float = 0.8,
        enable_mobility: bool = True,
        enable_channel_variation: bool = True,
        on_window: int = 1000,
        bs_top_k: int = 5,
        hard_window_len: int = 10000,
        bs_over_penalty: float = 50.0,
        eta_q: float = 1.0,
        alpha_rate: float = 3.0,
        beta_z: float = 1.0,
        alpha3: float = 50.0,
        use_hard_constraint: bool = False,
        use_imperfect_csi:bool=False,
        csi_error_var: float = 0.0,
        kappa: float = 0.03
    ):
        self.base_stations = [bs for bs in base_stations if bs.bs_id != 0]
        self.users = users
        self.n_agents = len(users)
        self.n_bs = len(self.base_stations)

        self.V = float(V)
        self.power_budget_ratio = float(power_budget_ratio)
        self.enable_mobility = bool(enable_mobility)
        self.enable_channel_variation = bool(enable_channel_variation)

        self.bs_over_penalty = float(bs_over_penalty)
        self.eta_q = float(eta_q)
        self.alpha_rate = float(alpha_rate)
        self.beta_z = float(beta_z)
        self.alpha3 = float(alpha3)

        self.bs_top_k = int(bs_top_k)
        assert self.bs_top_k >= 1

        self.hard_window_len = int(hard_window_len)
        assert self.hard_window_len >= 1

        self.use_hard_constraint = bool(use_hard_constraint)
        self.use_imperfect_csi = bool(use_imperfect_csi)
        self.csi_error_var = float(csi_error_var)
        self.kappa = float(kappa)
        assert 0.0 <= self.kappa <= 1.0

        # Power (Watt)
        self.P_max = {bs.bs_id: 10 ** (bs.tx_power_dbm / 10) / 1000 for bs in self.base_stations}
        self.P_bar = {bs.bs_id: self.power_budget_ratio * self.P_max[bs.bs_id] for bs in self.base_stations}

        # Hard constraint allowance:
        # each BS can be ON at most power_budget_ratio * hard_window_len times per window
        self.hard_on_limit = {
            bs.bs_id: int(np.floor(self.power_budget_ratio * self.hard_window_len))
            for bs in self.base_stations
        }

        # Queues
        self.Q_u = {u.ue_id: 0.1 for u in users}
        self.Z_b = {bs.bs_id: 0.0 for bs in self.base_stations}
        self.R_max = {u.ue_id: 5.0 for u in users}

        # Previous requested BS and previous actually serving BS are tracked separately.
        self.prev_request_bs = {u.ue_id: -1 for u in users}
        self.prev_serving_bs = {u.ue_id: -1 for u in users}
        self.handover_count_per_user = {u.ue_id: 0 for u in users}

        # Per-window hard handover usage (evaluation only).
        self.hard_ho_limit = int(np.floor(self.kappa * max(1, self.hard_window_len - 1)))
        self.ho_used_in_window = {u.ue_id: 0 for u in users}

        # Per-slot logging for candidate-level hard-HO filtering.
        # These values are set in build_bs_decision_inputs() immediately
        # before the BS actors sample their scheduling actions.
        self.last_ho_blocked_candidate_count = 0
        self.last_ho_no_valid_candidate_bs_count = 0

        # Channel / mobility
        self.noise_dbm = -174 + 10 * np.log10(500e6) + 5
        self.noise_watts = 10 ** (self.noise_dbm / 10) / 1000
        self.mobility_speed = 1.0
        self.area_size = 100
        self.channel_gains = defaultdict(dict)
        self.fading_std = 4.0

        self.timestep = 0

        # UE action: select exactly one BS (no NO-REQUEST action)
        self.action_dim = self.n_bs

        # BS action: Top-K candidates + NONE
        self.bs_action_dim = self.bs_top_k + 1

        # Recent ON ratio history
        self.on_window = int(on_window)
        self.bs_on_hist = {bs.bs_id: deque(maxlen=self.on_window) for bs in self.base_stations}

        # Congestion logging only
        self.prev_req_ratio = {bs.bs_id: 0.0 for bs in self.base_stations}

        # Previous-slot power for interference estimation in decision phase
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        # Hard window usage
        self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}
        self.window_step = 0

        # Fast lookup maps
        self.user_map = {u.ue_id: u for u in self.users}
        self.bs_map = {bs.bs_id: bs for bs in self.base_stations}
        self.ue_id_to_index = {u.ue_id: i for i, u in enumerate(self.users)}
        self.bs_id_to_index = {bs.bs_id: i for i, bs in enumerate(self.base_stations)}

        # Observation dimensions for pure LyMARL (no handover queue/state).
        # UE local: [Q_u] + rates(n_bs) + Z_b(n_bs)
        self.local_obs_dim = 1 + 2 * self.n_bs

        # BS local: [Z_b] + top-K Q_u * rate scores
        self.bs_obs_dim = 1 + self.bs_top_k

        # Global:
        # per UE: [Q_u, rates(n_bs)] => n_agents * (1 + n_bs)
        # per BS: [Z_b] => n_bs
        self.global_obs_dim = self.n_agents * (1 + self.n_bs) + self.n_bs

        self._rate_cache = np.zeros((self.n_agents, self.n_bs), dtype=np.float32)
        self.no_coverage_count = 0

        print(f"\n{'='*96}")
        print(" Pure LyMARL Environment (no HO queue)")
        print(f"{'='*96}")
        print(f"#UE={self.n_agents} | #BS={self.n_bs} | UE_action_dim={self.action_dim} | BS_action_dim={self.bs_action_dim}")
        print(f"V={self.V} | power_budget_ratio={self.power_budget_ratio} | bs_over_penalty={self.bs_over_penalty}")
        print(f"UE team reward = mean_u[Q_u(t)*served_rate_u]")
        print(f"Per-user reward = Q_u(t)*served_rate_u")
        print(f"BS reward = OFF: 0, ON: log(served_rate) - P_tx*Z_b(t)")
        print(f"Hard constraint enabled: {self.use_hard_constraint}")
        print("Hard HO handling = candidate-level filtering; BS OFF only if no valid candidate remains")
        print(f"local_obs_dim={self.local_obs_dim} | bs_obs_dim={self.bs_obs_dim} | global_obs_dim={self.global_obs_dim}")
        print(f"{'='*96}\n")

    def set_hard_constraint(self, enabled: bool):
        self.use_hard_constraint = bool(enabled)

    def set_imperfect_csi(self, enabled: bool, error_var: Optional[float] = None):
        self.use_imperfect_csi = bool(enabled)
        self.csi_error_var = float(error_var) if error_var is not None else self.csi_error_var

    def reset(self):
        self.timestep = 0
        self.no_coverage_count = 0

        for user in self.users:
            user.position = np.array([np.random.uniform(10, 90), np.random.uniform(10, 90)])

        self.update_channel_gains(0)

        self.Q_u = {u.ue_id: 0.1 for u in self.users}
        self.Z_b = {bs.bs_id: 0.0 for bs in self.base_stations}
        self.R_max = {u.ue_id: 5.0 for u in self.users}

        self.prev_request_bs = {u.ue_id: -1 for u in self.users}
        self.prev_serving_bs = {u.ue_id: -1 for u in self.users}
        self.handover_count_per_user = {u.ue_id: 0 for u in self.users}
        self.ho_used_in_window = {u.ue_id: 0 for u in self.users}
        self.last_ho_blocked_candidate_count = 0
        self.last_ho_no_valid_candidate_bs_count = 0

        self.bs_on_hist = {bs.bs_id: deque(maxlen=self.on_window) for bs in self.base_stations}
        self.prev_req_ratio = {bs.bs_id: 0.0 for bs in self.base_stations}
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}
        self.window_step = 0

        self.update_max_rates()
        return self._get_observations()

    # =========================================================
    # Dynamics
    # =========================================================
    def update_user_positions(self):
        if not self.enable_mobility:
            return

        for user in self.users:
            dx, dy = np.random.normal(0, self.mobility_speed, 2)
            new_x = np.clip(user.position[0] + dx, 5, self.area_size - 5)
            new_y = np.clip(user.position[1] + dy, 5, self.area_size - 5)
            user.position = np.array([new_x, new_y])

    def update_channel_gains(self, t: int):
        if not self.enable_channel_variation:
            for u in self.users:
                for bs in self.base_stations:
                    self.channel_gains[u.ue_id][bs.bs_id] = 1.0
            return

        for u in self.users:
            for bs in self.base_stations:
                if t == 0:
                    fading_db = np.random.normal(0, self.fading_std)
                else:
                    prev_db = 10 * np.log10(self.channel_gains[u.ue_id][bs.bs_id] + 1e-10)
                    fading_db = 0.9 * prev_db + np.random.normal(0, self.fading_std * np.sqrt(1 - 0.9**2))
                self.channel_gains[u.ue_id][bs.bs_id] = 10 ** (fading_db / 10)

    # =========================================================
    # PHY / Rate
    # =========================================================
    def _apply_csi_estimation_noise(self, gain: float) -> float:
        gain = max(float(gain), 1e-12)
        if (not self.use_imperfect_csi) or self.csi_error_var <= 0.0:
            return gain

        h_amp = np.sqrt(gain)
        sigma = np.sqrt(self.csi_error_var/2.0)
        e_real = np.random.normal(0.0, sigma)
        e_imag = np.random.normal(0.0, sigma)
        
        h_est = h_amp + e_real + 1j * e_imag
        noisy_gain = np.abs(h_est) ** 2

        return max(float(noisy_gain), 1e-12)
    
    def calculate_achievable_rate(self, user_id: int, bs_id: int) -> float:
        """
        Rate used for decision / cache.
        Interference is estimated using previous-slot BS power.
        Returns rate in Gbps.
        """
        user = self.user_map[user_id]
        bs = self.bs_map[bs_id]

        if not bs.can_serve(user.position):
            return 0.0

        dist = max(1, bs.distance_to(user.position))
        rx_dbm = bs.receive_power(dist)

        gain = self.channel_gains.get(user_id, {}).get(bs_id, 1.0)
        gain = self._apply_csi_estimation_noise(gain)
        rx_dbm += 10 * np.log10(gain + 1e-12)
        rx_watts = 10 ** (rx_dbm / 10) / 1000

        interference = 0.0
        for other_bs in self.base_stations:
            if other_bs.bs_id == bs_id:
                continue

            prev_p = float(self.prev_power.get(other_bs.bs_id, 0.0))
            if prev_p <= 0.0:
                continue

            other_dist = max(1, other_bs.distance_to(user.position))
            other_rx_dbm = other_bs.receive_power(other_dist)
            other_gain = self.channel_gains.get(user_id, {}).get(other_bs.bs_id, 1.0)
            other_gain = self._apply_csi_estimation_noise(other_gain)
            other_rx_dbm += 10 * np.log10(other_gain + 1e-12)
            other_rx_watts = 10 ** (other_rx_dbm / 10) / 1000

            denom = max(float(self.P_max.get(other_bs.bs_id, 1e-12)), 1e-12)
            power_scale = prev_p / denom
            interference += other_rx_watts * power_scale

        sinr = rx_watts / (self.noise_watts + interference)
        rate_bps = bs.bandwidth * np.log2(1 + sinr)
        return max(0.0, float(rate_bps / 1e9))

    def calculate_scheduled_rate(self, user_id: int, serving_bs_id: int, tx_power_map: Dict[int, float]) -> float:
        """
        Actual rate after scheduling decision.
        Interference is computed from current-slot tx_power_map.
        Returns rate in Gbps.
        """
        user = self.user_map[user_id]
        bs = self.bs_map[serving_bs_id]

        if not bs.can_serve(user.position):
            return 0.0

        dist = max(1, bs.distance_to(user.position))
        rx_dbm = bs.receive_power(dist)

        gain = self.channel_gains.get(user_id, {}).get(serving_bs_id, 1.0)
        rx_dbm += 10 * np.log10(gain + 1e-12)
        rx_watts = 10 ** (rx_dbm / 10) / 1000

        interference = 0.0
        for other_bs in self.base_stations:
            if other_bs.bs_id == serving_bs_id:
                continue

            p_now = float(tx_power_map.get(other_bs.bs_id, 0.0))
            if p_now <= 0.0:
                continue

            other_dist = max(1, other_bs.distance_to(user.position))
            other_rx_dbm = other_bs.receive_power(other_dist)
            other_gain = self.channel_gains.get(user_id, {}).get(other_bs.bs_id, 1.0)
            other_rx_dbm += 10 * np.log10(other_gain + 1e-12)
            other_rx_watts = 10 ** (other_rx_dbm / 10) / 1000

            denom = max(float(self.P_max.get(other_bs.bs_id, 1e-12)), 1e-12)
            power_scale = p_now / denom
            interference += other_rx_watts * power_scale

        sinr = rx_watts / (self.noise_watts + interference)
        rate_bps = bs.bandwidth * np.log2(1 + sinr)
        return max(0.0, float(rate_bps / 1e9))

    def compute_aux_rate(self, u_id: int) -> float:
        """
        Auxiliary rate term for queue update:
        r* = min{R_max, V / Q}
        """
        Q_u = self.Q_u[u_id]
        return min(self.R_max[u_id], self.V / max(Q_u, 1e-6))

    def update_max_rates(self):
        """
        Compute R_max and cache UE-BS achievable rates for the current state.
        """
        rates = np.zeros((self.n_agents, self.n_bs), dtype=np.float32)

        for ui, user in enumerate(self.users):
            max_rate = 0.0
            for bi, bs in enumerate(self.base_stations):
                r = self.calculate_achievable_rate(user.ue_id, bs.bs_id)
                rates[ui, bi] = float(r)
                if r > max_rate:
                    max_rate = r
            self.R_max[user.ue_id] = max_rate if max_rate > 0 else 1.0

        self._rate_cache = rates

    # =========================================================
    # Features / Observations
    # =========================================================
    def _get_bs_on_features(self) -> List[float]:
        feats = []
        for bs in self.base_stations:
            hist = self.bs_on_hist[bs.bs_id]
            feats.append(0.0 if len(hist) == 0 else float(sum(hist) / len(hist)))
        return feats

    def _get_local_observation_by_index(self, ui: int) -> np.ndarray:
        ue = self.users[ui]
        ue_id = ue.ue_id

        obs = [float(self.Q_u[ue_id])]
        obs.extend(self._rate_cache[ui, :].tolist())

        for bs in self.base_stations:
            obs.append(float(self.Z_b[bs.bs_id]))

        result = np.asarray(obs, dtype=np.float32)
        assert result.shape[0] == self.local_obs_dim, (
            f"UE obs dim mismatch: {result.shape[0]} vs {self.local_obs_dim}"
        )
        return result

    def _get_global_observation(self) -> np.ndarray:
        obs = []
        for ui, ue in enumerate(self.users):
            ue_id = ue.ue_id
            obs.append(float(self.Q_u[ue_id]))
            obs.extend(self._rate_cache[ui, :].tolist())

        for bs in self.base_stations:
            obs.append(float(self.Z_b[bs.bs_id]))

        result = np.asarray(obs, dtype=np.float32)
        assert result.shape[0] == self.global_obs_dim, (
            f"Global obs dim mismatch: {result.shape[0]} vs {self.global_obs_dim}"
        )
        return result

    def _get_observations(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        local_obs = {}
        for ui, ue in enumerate(self.users):
            local_obs[ue.ue_id] = self._get_local_observation_by_index(ui)

        global_obs = self._get_global_observation()
        return local_obs, global_obs

    # =========================================================
    # Masks / Decision Inputs
    # =========================================================
    def _get_action_mask(self, ue_id: int) -> np.ndarray:
        """
        UE action mask over B association actions. There is no NO-REQUEST action.

        Pure LyMARL does not observe or mask UE association actions using the
        handover budget. During hard-constrained evaluation, the handover
        budget is enforced at the BS candidate level: an exhausted UE is
        removed only from BSs that would cause a new handover, while other
        valid requesting UEs remain schedulable.
        """
        user = self.user_map[ue_id]
        mask = np.zeros(self.action_dim, dtype=bool)

        for bi, bs in enumerate(self.base_stations):
            mask[bi] = bool(bs.can_serve(user.position))

        if not mask.any():
            self.no_coverage_count += 1
            distances = [float(bs.distance_to(user.position)) for bs in self.base_stations]
            mask[int(np.argmin(distances))] = True

        return mask

    def build_bs_decision_inputs(self, ue_actions: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """
        Pure-LyMARL BS observation:
            [Z_b, s_1, ..., s_K]
        where
            s_{u,b}(t) = Q_u(t) * rate_{u,b}(t).

        Training / soft evaluation:
            Candidate construction is the original pure-LyMARL Top-K rule.

        Hard-constrained evaluation:
            The HO budget is enforced per UE-BS candidate before Top-K is
            formed. A candidate is infeasible only when

                ho_used_u >= hard_ho_limit
                and b != previous_serving_bs_u.

            Thus, an exhausted UE is blocked only from a BS that would create
            another handover. Other UEs requesting the same BS remain valid,
            and valid lower-ranked requesters can move into the Top-K list.
            The BS is forced to NONE/OFF only when no feasible candidate
            remains (or when the learned BS policy voluntarily chooses NONE).
        """
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}

        for ue_id, action in ue_actions.items():
            action = int(action)
            assert 0 <= action < self.n_bs, f"Invalid UE association action: {action}"
            bs_id = self.base_stations[action].bs_id
            bs_requests[bs_id].append(ue_id)

        bs_obs_batch = np.zeros((self.n_bs, self.bs_obs_dim), dtype=np.float32)
        bs_mask_batch = np.zeros((self.n_bs, self.bs_action_dim), dtype=bool)
        cand_lists: List[List[int]] = []

        blocked_candidate_count = 0
        no_valid_candidate_bs_count = 0

        for bi, bs in enumerate(self.base_stations):
            scored = []
            positive_rate_request_count = 0

            for ue_id in bs_requests[bs.bs_id]:
                ui = self.ue_id_to_index[ue_id]
                rate = float(self._rate_cache[ui, bi])
                if rate <= 0.0:
                    continue

                positive_rate_request_count += 1

                # Candidate-level hard HO feasibility check.
                # The first actual association is not a handover.
                if self.use_hard_constraint:
                    prev_bs_id = int(self.prev_serving_bs[ue_id])
                    ho_exhausted = (
                        self.ho_used_in_window[ue_id]
                        >= self.hard_ho_limit
                    )
                    would_handover = (
                        prev_bs_id >= 0
                        and bs.bs_id != prev_bs_id
                    )

                    if ho_exhausted and would_handover:
                        blocked_candidate_count += 1
                        continue

                score = float(self.Q_u[ue_id] * rate)
                scored.append((score, ue_id))

            # There were actual requesters with positive rate, but every one
            # was infeasible solely because of the hard HO budget. In this
            # case only the NONE/OFF action will remain valid for this BS.
            if positive_rate_request_count > 0 and len(scored) == 0:
                no_valid_candidate_bs_count += 1

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:self.bs_top_k]
            cand = [ue_id for score, ue_id in top]
            scores = [score for score, ue_id in top]

            while len(cand) < self.bs_top_k:
                cand.append(-1)
                scores.append(0.0)

            cand_lists.append(cand)
            bs_obs_batch[bi, :] = np.asarray(
                [float(self.Z_b[bs.bs_id]), *[float(s) for s in scores]],
                dtype=np.float32,
            )

            # Candidate actions 0,...,K-1 are enabled only for feasible UEs.
            for k in range(self.bs_top_k):
                bs_mask_batch[bi, k] = cand[k] >= 0

            # Dedicated NONE/OFF action is always valid.
            bs_mask_batch[bi, self.bs_top_k] = True

        self.last_ho_blocked_candidate_count = int(blocked_candidate_count)
        self.last_ho_no_valid_candidate_bs_count = int(no_valid_candidate_bs_count)

        return bs_obs_batch, bs_mask_batch, cand_lists

    def step_joint(self, ue_actions: Dict[int, int], bs_actions: Dict[int, int], cand_lists: List[List[int]]):
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}
        current_request_bs: Dict[int, int] = {}
        request_switch_flags = {u.ue_id: 0.0 for u in self.users}

        # UE association requests and request-switch metric.
        for ue_id, action in ue_actions.items():
            action = int(action)
            assert 0 <= action < self.action_dim, f"Invalid UE action {action}"

            requested_bs_id = self.base_stations[action].bs_id
            prev_request_bs = int(self.prev_request_bs[ue_id])
            current_request_bs[ue_id] = requested_bs_id
            bs_requests[requested_bs_id].append(ue_id)

            if prev_request_bs >= 0 and requested_bs_id != prev_request_bs:
                request_switch_flags[ue_id] = 1.0

        assert len(current_request_bs) == self.n_agents

        for bs in self.base_stations:
            self.prev_req_ratio[bs.bs_id] = len(bs_requests[bs.bs_id]) / max(1, self.n_agents)

        # BS scheduling / activation.
        bs_selections: Dict[int, Optional[int]] = {}
        for bi, bs in enumerate(self.base_stations):
            a_b = int(bs_actions[bs.bs_id])
            if a_b == self.bs_top_k:
                bs_selections[bs.bs_id] = None
                continue

            cand = cand_lists[bi]
            if not (0 <= a_b < self.bs_top_k):
                bs_selections[bs.bs_id] = None
                continue

            ue_id = cand[a_b]
            if ue_id < 0 or ue_id not in bs_requests[bs.bs_id]:
                bs_selections[bs.bs_id] = None
                continue

            ui = self.ue_id_to_index[ue_id]
            bs_selections[bs.bs_id] = ue_id if float(self._rate_cache[ui, bi]) > 0.0 else None

        # Hard caps during evaluation.
        #
        # 1) Energy cap: force the BS OFF after its ON budget is exhausted.
        # 2) Handover cap: normally already enforced in
        #    build_bs_decision_inputs() by removing only infeasible UE-BS
        #    candidates. The check below is retained as a safety guard.
        ho_safety_forced_off_count = 0
        if self.use_hard_constraint:
            for bs in self.base_stations:
                bs_id = bs.bs_id

                if self.bs_on_used_in_window[bs_id] >= self.hard_on_limit[bs_id]:
                    bs_selections[bs_id] = None
                    continue

                selected_ue = bs_selections[bs_id]
                if selected_ue is None:
                    continue

                prev_bs_id = int(self.prev_serving_bs[selected_ue])
                would_handover = (
                    prev_bs_id >= 0
                    and bs_id != prev_bs_id
                )
                ho_exhausted = (
                    self.ho_used_in_window[selected_ue]
                    >= self.hard_ho_limit
                )

                # This should normally be unreachable because the candidate
                # was already filtered before action sampling.
                if would_handover and ho_exhausted:
                    bs_selections[bs_id] = None
                    ho_safety_forced_off_count += 1

        ho_blocked_candidate_count = int(
            self.last_ho_blocked_candidate_count
        )
        ho_no_valid_candidate_bs_count = int(
            self.last_ho_no_valid_candidate_bs_count
        )

        # Backward-compatible aggregate: number of BSs forced to have no
        # service because all positive-rate candidates were HO-infeasible,
        # plus any unexpected safety-guard rejection.
        ho_forced_off_count = (
            ho_no_valid_candidate_bs_count
            + ho_safety_forced_off_count
        )

        tx_power_map_now = {
            bs.bs_id: (float(self.P_max[bs.bs_id]) if bs_selections[bs.bs_id] is not None else 0.0)
            for bs in self.base_stations
        }

        served_rates = {u.ue_id: 0.0 for u in self.users}
        bs_served_rate = {bs.bs_id: 0.0 for bs in self.base_stations}
        for bs in self.base_stations:
            served_ue_id = bs_selections[bs.bs_id]
            if served_ue_id is None:
                continue
            rate = self.calculate_scheduled_rate(served_ue_id, bs.bs_id, tx_power_map_now)
            served_rates[served_ue_id] = max(served_rates[served_ue_id], rate)
            bs_served_rate[bs.bs_id] = float(rate)

        total_rate = float(sum(served_rates.values()))
        power_consumed = {bs.bs_id: float(tx_power_map_now[bs.bs_id]) for bs in self.base_stations}

        # Actual serving-based HO, CondHO opportunity, and serving state.
        handover_flags = {u.ue_id: 0.0 for u in self.users}
        served_flags = {u.ue_id: 0.0 for u in self.users}
        conditional_ho_opportunity = {u.ue_id: 0.0 for u in self.users}
        current_serving_bs = self.prev_serving_bs.copy()

        for bs in self.base_stations:
            served_ue_id = bs_selections[bs.bs_id]
            if served_ue_id is None or served_rates[served_ue_id] <= 0.0:
                continue

            served_flags[served_ue_id] = 1.0
            prev_serving_bs = int(self.prev_serving_bs[served_ue_id])
            new_serving_bs = int(bs.bs_id)

            if prev_serving_bs >= 0:
                conditional_ho_opportunity[served_ue_id] = 1.0
                if new_serving_bs != prev_serving_bs:
                    handover_flags[served_ue_id] = 1.0

            current_serving_bs[served_ue_id] = new_serving_bs

        self.window_step += 1
        for bs in self.base_stations:
            if power_consumed[bs.bs_id] > 0.0:
                self.bs_on_used_in_window[bs.bs_id] += 1
            self.bs_on_hist[bs.bs_id].append(1.0 if power_consumed[bs.bs_id] > 0.0 else 0.0)

        for u in self.users:
            if handover_flags[u.ue_id] > 0.0:
                self.ho_used_in_window[u.ue_id] += 1

        # Reset hard-budget counters after the current slot has been accounted for.
        if self.window_step % self.hard_window_len == 0:
            self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}
            self.ho_used_in_window = {u.ue_id: 0 for u in self.users}

        self.prev_power = power_consumed.copy()

        old_Q_u = self.Q_u.copy()
        old_Z_b = self.Z_b.copy()

        # Virtual queue updates.
        for u in self.users:
            ue_id = u.ue_id
            aux_rate = self.compute_aux_rate(ue_id)
            self.Q_u[ue_id] = max(1e-12, self.Q_u[ue_id] + aux_rate - served_rates[ue_id])

        for bs in self.base_stations:
            bs_id = bs.bs_id
            self.Z_b[bs_id] = max(0.0, self.Z_b[bs_id] + power_consumed[bs_id] - self.P_bar[bs_id])

        for u in self.users:
            ue_id = u.ue_id
            h_u = float(handover_flags[ue_id])
            self.handover_count_per_user[ue_id] += int(h_u)

        # Pure LyMARL user reward: no handover queue and no handover penalty.
        ue_team_reward = float(np.mean([
            old_Q_u[u.ue_id] * served_rates[u.ue_id]
            for u in self.users
        ]))
        ue_per_user_rewards = {
            u.ue_id: float(
                old_Q_u[u.ue_id] * served_rates[u.ue_id]
            )
            for u in self.users
        }

        # Role-specific BS rewards: log(rate) - Z_b(t) P_tx when ON; zero when OFF.
        on_feats = self._get_bs_on_features()
        bs_rewards = []
        for bs in self.base_stations:
            bs_id = bs.bs_id
            if bs_selections[bs_id] is None:
                r_b = 0.0
            else:
                r_b = float(np.log(max(float(bs_served_rate[bs_id]), 1e-12))) \
                      - float(self.P_max[bs_id]) * float(old_Z_b[bs_id])
            bs_rewards.append(float(r_b))

        bs_rewards = np.asarray(bs_rewards, dtype=np.float32)
        bs_on_ratio_penalties = np.zeros(self.n_bs, dtype=np.float32)
        bs_team_reward = float(bs_rewards.mean())

        # State carried to the next slot.
        self.prev_request_bs = current_request_bs.copy()
        self.prev_serving_bs = current_serving_bs.copy()

        self.timestep += 1
        self.update_user_positions()
        self.update_channel_gains(self.timestep)
        self.update_max_rates()
        local_obs, global_obs = self._get_observations()

        info = {
            "total_throughput": total_rate,
            "power_consumed": power_consumed,
            "served_rates": served_rates,
            "Q_u": self.Q_u.copy(),
            "Z_b": self.Z_b.copy(),
            "ue_team_reward": ue_team_reward,
            "ue_per_user_rewards": ue_per_user_rewards,
            "bs_rewards": bs_rewards.copy(),
            "bs_team_reward": bs_team_reward,
            "bs_on_ratio_penalties": bs_on_ratio_penalties.copy(),
            "bs_selections": bs_selections,
            "bs_requests": {bs_id: len(reqs) for bs_id, reqs in bs_requests.items()},
            "handover_flags": handover_flags.copy(),
            "served_flags": served_flags.copy(),
            "conditional_ho_opportunity": conditional_ho_opportunity.copy(),
            "request_switch_flags": request_switch_flags.copy(),
            "handover_count": int(sum(handover_flags.values())),
            "handover_ratio": float(sum(handover_flags.values()) / max(1, self.n_agents)),
            "request_switch_ratio": float(sum(request_switch_flags.values()) / max(1, self.n_agents)),
            "cond_ho_numerator": int(sum(handover_flags.values())),
            "cond_ho_denominator": int(sum(conditional_ho_opportunity.values())),
            "handover_count_per_user": self.handover_count_per_user.copy(),
            "current_request_bs": current_request_bs.copy(),
            "current_serving_bs": current_serving_bs.copy(),
            "prev_request_bs": self.prev_request_bs.copy(),
            "prev_serving_bs": self.prev_serving_bs.copy(),
            "kappa": float(self.kappa),
            "prev_req_ratio": self.prev_req_ratio.copy(),
            "total_QR_dummy": float(sum(old_Q_u[u.ue_id] * served_rates[u.ue_id] for u in self.users)),
            "total_ZP_dummy": float(sum(old_Z_b[bs.bs_id] * power_consumed[bs.bs_id] for bs in self.base_stations)),
            "no_coverage_count": int(self.no_coverage_count),
            "bs_on_used_in_window": self.bs_on_used_in_window.copy(),
            "ho_used_in_window": self.ho_used_in_window.copy(),
            "window_step": int(self.window_step),
            "on_feats": on_feats,
            "rho": float(self.power_budget_ratio),
            "hard_constraint_enabled": bool(self.use_hard_constraint),
            "hard_on_limit": self.hard_on_limit.copy(),
            "hard_ho_limit": int(self.hard_ho_limit),
            "ho_blocked_candidate_count": int(ho_blocked_candidate_count),
            "ho_no_valid_candidate_bs_count": int(ho_no_valid_candidate_bs_count),
            "ho_safety_forced_off_count": int(ho_safety_forced_off_count),
            "ho_forced_off_count": int(ho_forced_off_count),
        }

        done = False
        return local_obs, global_obs, info, done

    def calculate_jain_fairness(self, rate_history: List) -> float:
        """
        Jain's fairness computed from the most recent up to 100 slots.
        """
        fair_window = 1000
        recent = rate_history if len(rate_history) < fair_window else rate_history[-fair_window:]
        if not recent:
            return 0.0

        rate_array = np.array(recent)
        per_user_avg = rate_array.mean(axis=0)

        sum_rates = per_user_avg.sum()
        sum_squared = (per_user_avg ** 2).sum()
        n_users = len(per_user_avg)

        if sum_squared < 1e-12:
            return 0.0

        return float((sum_rates ** 2) / (n_users * sum_squared))

def compute_block_jfi(slot_rates, block_size: int = 1000) -> float:
    """Mean Jain fairness over non-overlapping blocks, matching the table metric."""
    rates = np.asarray(slot_rates, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[0] == 0:
        return 0.0

    block_jfis = []
    for start in range(0, rates.shape[0], block_size):
        block = rates[start:start + block_size]
        if block.shape[0] == 0:
            continue
        per_user_avg = block.mean(axis=0)
        denom = float(len(per_user_avg) * np.square(per_user_avg).sum())
        if denom > 1e-12:
            block_jfis.append(float(per_user_avg.sum() ** 2) / denom)
    return float(np.mean(block_jfis)) if block_jfis else 0.0


# ============================================================
# Trainer
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
from typing import Dict, Optional

class MAPPOTrainer:
    def __init__(
        self,
        env,
        lr_actor_ue: float = 3e-4,
        lr_actor_bs: float = 3e-4,
        lr_critic_ue: float = 1e-3,
        lr_critic_bs: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef_ue: float = 0.05,
        entropy_coef_bs: float = 0.05,
        value_coef_ue: float = 0.5,
        value_coef_bs: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        minibatch_size: int = 256,
    ):
        self.env = env
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef_ue = float(entropy_coef_ue)
        self.entropy_coef_bs = float(entropy_coef_bs)
        self.value_coef_ue = float(value_coef_ue)
        self.value_coef_bs = float(value_coef_bs)
        self.max_grad_norm = float(max_grad_norm)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actors
        self.ue_actor = UEActorNetwork(env.local_obs_dim, env.action_dim).to(self.device)
        self.ue_actor_optim = optim.Adam(self.ue_actor.parameters(), lr=lr_actor_ue)

        self.bs_actor = BSActorNetwork(env.bs_obs_dim, env.bs_action_dim).to(self.device)
        self.bs_actor_optim = optim.Adam(self.bs_actor.parameters(), lr=lr_actor_bs)

        # Critics
        self.critic_ue = CentralizedCriticUE(env.global_obs_dim).to(self.device)
        self.critic_ue_opt = optim.Adam(self.critic_ue.parameters(), lr=lr_critic_ue)

        self.critic_bs = CentralizedCriticBS(env.global_obs_dim, n_bs=env.n_bs).to(self.device)
        self.critic_bs_opt = optim.Adam(self.critic_bs.parameters(), lr=lr_critic_bs)

        # Value normalization
        self.vn_ue = ValueNorm(device=self.device)
        self.vn_bs = ValueNormVec(dim=env.n_bs, device=self.device)

        self.reset_rollout()

        print(f"[TRAINER] UE agents(shared actor): {len(env.users)}")
        print(f"[TRAINER] BS agents(shared actor): {len(env.base_stations)} | TopK={env.bs_top_k}")
        print(f"[TRAINER] Device: {self.device}")
        print(f"[TRAINER] PPO epochs: {self.n_epochs} | minibatch_size: {self.minibatch_size}")
        print(f"[TRAINER] TWO critics: UE scalar / BS vector(B={env.n_bs})")
        # print(f"[TRAINER] UE action includes NO-REQUEST at index {env.no_request_action}\n")

    # =========================================================
    # Save / Load
    # =========================================================
    def save_model(self, path: str, save_optim: bool = False):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        payload = {
            "meta": {
                "local_obs_dim": self.env.local_obs_dim,
                "action_dim": self.env.action_dim,
                "bs_obs_dim": self.env.bs_obs_dim,
                "bs_action_dim": self.env.bs_action_dim,
                "global_obs_dim": self.env.global_obs_dim,
                "n_bs": self.env.n_bs,
                "n_users": self.env.n_agents,
            },
            "ue_actor": self.ue_actor.state_dict(),
            "bs_actor": self.bs_actor.state_dict(),
            "critic_ue": self.critic_ue.state_dict(),
            "critic_bs": self.critic_bs.state_dict(),
            "vn_ue": self.vn_ue.state_dict(),
            "vn_bs": self.vn_bs.state_dict(),
        }

        if save_optim:
            payload.update({
                "ue_actor_optim": self.ue_actor_optim.state_dict(),
                "bs_actor_optim": self.bs_actor_optim.state_dict(),
                "critic_ue_opt": self.critic_ue_opt.state_dict(),
                "critic_bs_opt": self.critic_bs_opt.state_dict(),
            })

        torch.save(payload, path)
        print(f"✅ Model saved: {path}")

    def load_model(self, path: str, load_optim: bool = False, map_location: Optional[str] = None):
        map_location = map_location if map_location is not None else str(self.device)
        payload = torch.load(path, map_location=map_location)

        self.ue_actor.load_state_dict(payload["ue_actor"])
        self.bs_actor.load_state_dict(payload["bs_actor"])
        self.critic_ue.load_state_dict(payload["critic_ue"])
        self.critic_bs.load_state_dict(payload["critic_bs"])
        self.vn_ue.load_state_dict(payload["vn_ue"])
        self.vn_bs.load_state_dict(payload["vn_bs"])

        if load_optim and ("ue_actor_optim" in payload):
            self.ue_actor_optim.load_state_dict(payload["ue_actor_optim"])
            self.bs_actor_optim.load_state_dict(payload["bs_actor_optim"])
            self.critic_ue_opt.load_state_dict(payload["critic_ue_opt"])
            self.critic_bs_opt.load_state_dict(payload["critic_bs_opt"])

        self.ue_actor.eval()
        self.bs_actor.eval()
        self.critic_ue.eval()
        self.critic_bs.eval()

        print(f"✅ Model loaded: {path} (optim={load_optim})")

    # =========================================================
    # Rollout buffer
    # =========================================================
    def reset_rollout(self):
        self.rb = {
            "local_obs": [],
            "ue_masks": [],
            "ue_actions": [],
            "ue_logp": [],

            "bs_obs": [],
            "bs_masks": [],
            "bs_actions": [],
            "bs_logp": [],
            "cand_lists": [],

            "global_obs": [],

            "rew_ue": [],
            "rew_bs": [],

            "v_ue_n": [],
            "nv_ue_n": [],
            "v_bs_n": [],
            "nv_bs_n": [],

            "dones": [],
        }

    @torch.no_grad()
    def select_actions(self, local_obs: Dict[int, np.ndarray], global_obs: np.ndarray):
        users = self.env.users
        global_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        v_ue_n = self.critic_ue(global_t).squeeze(0)
        v_bs_n = self.critic_bs(global_t).squeeze(0)

        # UE actions
        obs_batch = np.stack([local_obs[u.ue_id] for u in users], axis=0).astype(np.float32)
        ue_mask_batch = np.stack([self.env._get_action_mask(u.ue_id) for u in users], axis=0).astype(bool)

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        ue_mask_t = torch.as_tensor(ue_mask_batch, dtype=torch.bool, device=self.device)

        ue_logits = self.ue_actor(obs_t).masked_fill(~ue_mask_t, float("-inf"))
        ue_dist = Categorical(logits=ue_logits)
        ue_actions_t = ue_dist.sample()
        ue_logp_t = ue_dist.log_prob(ue_actions_t)
        ue_ent_t = ue_dist.entropy()

        ue_actions = {u.ue_id: int(ue_actions_t[i].item()) for i, u in enumerate(users)}

        # BS actions
        bs_obs_batch, bs_mask_batch, cand_lists = self.env.build_bs_decision_inputs(ue_actions)
        bs_obs_t = torch.as_tensor(bs_obs_batch, dtype=torch.float32, device=self.device)
        bs_mask_t = torch.as_tensor(bs_mask_batch, dtype=torch.bool, device=self.device)

        bs_logits = self.bs_actor(bs_obs_t).masked_fill(~bs_mask_t, float("-inf"))
        bs_dist = Categorical(logits=bs_logits)
        bs_actions_t = bs_dist.sample()
        bs_logp_t = bs_dist.log_prob(bs_actions_t)
        bs_ent_t = bs_dist.entropy()

        bs_actions = {bs.bs_id: int(bs_actions_t[i].item()) for i, bs in enumerate(self.env.base_stations)}

        return (
            ue_actions,
            ue_logp_t.detach().cpu().numpy().astype(np.float32),
            ue_ent_t.detach().cpu().numpy().astype(np.float32),
            ue_mask_batch,

            bs_actions,
            bs_logp_t.detach().cpu().numpy().astype(np.float32),
            bs_ent_t.detach().cpu().numpy().astype(np.float32),
            bs_obs_batch,
            bs_mask_batch,
            cand_lists,

            float(v_ue_n.item()),
            v_bs_n.detach().cpu().numpy().astype(np.float32),
        )

    def store_step(
        self,
        local_obs, global_obs,
        ue_actions_dict, ue_logp_np, ue_masks_np,
        bs_actions_dict, bs_logp_np, bs_obs_np, bs_masks_np, cand_lists,
        rew_ue: float, rew_bs: np.ndarray,
        v_ue_n: float, nv_ue_n: float,
        v_bs_n: np.ndarray, nv_bs_n: np.ndarray,
        done: bool
    ):
        users = self.env.users
        bss = self.env.base_stations
        B = len(bss)

        ue_obs_step = np.stack([local_obs[u.ue_id] for u in users], axis=0).astype(np.float32)
        ue_act_step = np.array([ue_actions_dict[u.ue_id] for u in users], dtype=np.int64)
        bs_act_step = np.array([bs_actions_dict[bs.bs_id] for bs in bss], dtype=np.int64)

        self.rb["local_obs"].append(ue_obs_step)
        self.rb["ue_masks"].append(ue_masks_np.astype(bool))
        self.rb["ue_actions"].append(ue_act_step)
        self.rb["ue_logp"].append(ue_logp_np)

        self.rb["bs_obs"].append(bs_obs_np.astype(np.float32))
        self.rb["bs_masks"].append(bs_masks_np.astype(bool))
        self.rb["bs_actions"].append(bs_act_step)
        self.rb["bs_logp"].append(bs_logp_np)
        self.rb["cand_lists"].append(cand_lists)

        self.rb["global_obs"].append(np.array(global_obs, dtype=np.float32))

        self.rb["rew_ue"].append(float(rew_ue))
        self.rb["rew_bs"].append(np.array(rew_bs, dtype=np.float32).reshape(B))

        self.rb["v_ue_n"].append(float(v_ue_n))
        self.rb["nv_ue_n"].append(float(nv_ue_n))

        self.rb["v_bs_n"].append(np.array(v_bs_n, dtype=np.float32).reshape(B))
        self.rb["nv_bs_n"].append(np.array(nv_bs_n, dtype=np.float32).reshape(B))

        self.rb["dones"].append(bool(done))

    def _iter_minibatches(self, N: int, batch_size: int):
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            yield idx[start:start + batch_size]

    # =========================================================
    # GAE
    # =========================================================
    def compute_gae_ue(self, rewards, values_n, next_values_n, dones):
        T = len(rewards)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        v_n = torch.tensor(values_n, dtype=torch.float32, device=self.device)
        nv_n = torch.tensor(next_values_n, dtype=torch.float32, device=self.device)

        v = self.vn_ue.denormalize(v_n)
        nv = self.vn_ue.denormalize(nv_n)

        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            done_mask = 1.0 - float(dones[t])
            delta = r_t[t] + self.gamma * nv[t] * done_mask - v[t]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            adv[t] = gae

        ret_raw = adv + v
        return adv, ret_raw

    def compute_gae_bs(self, rewards_bs, values_bs_n, next_values_bs_n, dones):
        T = rewards_bs.shape[0]
        B = rewards_bs.shape[1]

        r = torch.tensor(rewards_bs, dtype=torch.float32, device=self.device)
        v_n = torch.tensor(values_bs_n, dtype=torch.float32, device=self.device)
        nv_n = torch.tensor(next_values_bs_n, dtype=torch.float32, device=self.device)

        v = self.vn_bs.denormalize(v_n)
        nv = self.vn_bs.denormalize(nv_n)

        adv = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        gae = torch.zeros(B, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            done_mask = 1.0 - float(dones[t])
            delta = r[t] + self.gamma * nv[t] * done_mask - v[t]
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            adv[t] = gae

        ret_raw = adv + v
        return adv, ret_raw

    # =========================================================
    # PPO Update
    # =========================================================
    def update(self):
        T = len(self.rb["dones"])
        if T == 0:
            return {}

        N = len(self.env.users)
        B = len(self.env.base_stations)
        global_obs = torch.tensor(np.stack(self.rb["global_obs"], axis=0), dtype=torch.float32, device=self.device)
        dones = self.rb["dones"]

        # UE GAE
        adv_ue, ret_ue_raw = self.compute_gae_ue(
            rewards=self.rb["rew_ue"],
            values_n=self.rb["v_ue_n"],
            next_values_n=self.rb["nv_ue_n"],
            dones=dones
        )
        with torch.no_grad():
            self.vn_ue.update(ret_ue_raw)
        ret_ue_n = self.vn_ue.normalize(ret_ue_raw).detach()
        adv_ue = (adv_ue - adv_ue.mean()) / (adv_ue.std() + 1e-8)
        adv_ue = adv_ue.detach()

        # BS GAE
        rew_bs = np.stack(self.rb["rew_bs"], axis=0)
        v_bs_n = np.stack(self.rb["v_bs_n"], axis=0)
        nv_bs_n = np.stack(self.rb["nv_bs_n"], axis=0)

        adv_bs, ret_bs_raw = self.compute_gae_bs(rew_bs, v_bs_n, nv_bs_n, dones)
        with torch.no_grad():
            self.vn_bs.update(ret_bs_raw)
        ret_bs_n = self.vn_bs.normalize(ret_bs_raw).detach()
        adv_bs = (adv_bs - adv_bs.mean()) / (adv_bs.std() + 1e-8)
        adv_bs = adv_bs.detach()

        # UE tensors
        ue_local_obs = torch.tensor(np.stack(self.rb["local_obs"], axis=0), dtype=torch.float32, device=self.device)
        ue_masks = torch.tensor(np.stack(self.rb["ue_masks"], axis=0), dtype=torch.bool, device=self.device)
        ue_actions = torch.tensor(np.stack(self.rb["ue_actions"], axis=0), dtype=torch.long, device=self.device)
        ue_old_logp = torch.tensor(np.stack(self.rb["ue_logp"], axis=0), dtype=torch.float32, device=self.device)

        ue_local_f = ue_local_obs.reshape(T * N, -1)
        ue_masks_f = ue_masks.reshape(T * N, -1)
        ue_actions_f = ue_actions.reshape(T * N)
        ue_old_logp_f = ue_old_logp.reshape(T * N)
        ue_adv_f = adv_ue.repeat_interleave(N)

        # BS tensors
        bs_obs = torch.tensor(np.stack(self.rb["bs_obs"], axis=0), dtype=torch.float32, device=self.device)
        bs_masks = torch.tensor(np.stack(self.rb["bs_masks"], axis=0), dtype=torch.bool, device=self.device)
        bs_actions = torch.tensor(np.stack(self.rb["bs_actions"], axis=0), dtype=torch.long, device=self.device)
        bs_old_logp = torch.tensor(np.stack(self.rb["bs_logp"], axis=0), dtype=torch.float32, device=self.device)

        bs_obs_f = bs_obs.reshape(T * B, -1)
        bs_masks_f = bs_masks.reshape(T * B, -1)
        bs_actions_f = bs_actions.reshape(T * B)
        bs_old_logp_f = bs_old_logp.reshape(T * B)
        bs_adv_f = adv_bs.reshape(T * B)

        losses = {
            "critic_ue": 0.0, "critic_bs": 0.0,
            "actor_ue": 0.0, "actor_bs": 0.0,
            "entropy_ue": 0.0, "entropy_bs": 0.0
        }

        for _ in range(self.n_epochs):
            # Critic UE
            c_ue_epoch, c_ue_cnt = 0.0, 0
            critic_mb = max(32, min(self.minibatch_size, T))
            for mb in self._iter_minibatches(T, critic_mb):
                v_pred_n = self.critic_ue(global_obs[mb])
                loss_v = F.mse_loss(v_pred_n, ret_ue_n[mb])

                self.critic_ue_opt.zero_grad()
                (self.value_coef_ue * loss_v).backward()
                nn.utils.clip_grad_norm_(self.critic_ue.parameters(), self.max_grad_norm)
                self.critic_ue_opt.step()

                c_ue_epoch += float(loss_v.item())
                c_ue_cnt += 1

            # Critic BS
            c_bs_epoch, c_bs_cnt = 0.0, 0
            critic_mb2 = max(32, min(self.minibatch_size, T))
            for mb in self._iter_minibatches(T, critic_mb2):
                v_pred_n = self.critic_bs(global_obs[mb])
                loss_v = F.mse_loss(v_pred_n, ret_bs_n[mb])

                self.critic_bs_opt.zero_grad()
                (self.value_coef_bs * loss_v).backward()
                nn.utils.clip_grad_norm_(self.critic_bs.parameters(), self.max_grad_norm)
                self.critic_bs_opt.step()

                c_bs_epoch += float(loss_v.item())
                c_bs_cnt += 1

            # UE actor
            ue_epoch, ue_ent_epoch, ue_cnt = 0.0, 0.0, 0
            M_ue = T * N
            ue_mb = max(64, min(self.minibatch_size, M_ue))
            for mb in self._iter_minibatches(M_ue, ue_mb):
                logits = self.ue_actor(ue_local_f[mb]).masked_fill(~ue_masks_f[mb], float("-inf"))
                dist = Categorical(logits=logits)

                new_logp = dist.log_prob(ue_actions_f[mb])
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - ue_old_logp_f[mb])
                surr1 = ratio * ue_adv_f[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * ue_adv_f[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_ent = -entropy.mean()

                self.ue_actor_optim.zero_grad()
                (loss_pi + self.entropy_coef_ue * loss_ent).backward()
                nn.utils.clip_grad_norm_(self.ue_actor.parameters(), self.max_grad_norm)
                self.ue_actor_optim.step()

                ue_epoch += float(loss_pi.item())
                ue_ent_epoch += float(loss_ent.item())
                ue_cnt += 1

            # BS actor
            bs_epoch, bs_ent_epoch, bs_cnt = 0.0, 0.0, 0
            M_bs = T * B
            bs_mb = max(64, min(self.minibatch_size, M_bs))
            for mb in self._iter_minibatches(M_bs, bs_mb):
                logits = self.bs_actor(bs_obs_f[mb]).masked_fill(~bs_masks_f[mb], float("-inf"))
                dist = Categorical(logits=logits)

                new_logp = dist.log_prob(bs_actions_f[mb])
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - bs_old_logp_f[mb])
                surr1 = ratio * bs_adv_f[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * bs_adv_f[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_ent = -entropy.mean()

                self.bs_actor_optim.zero_grad()
                (loss_pi + self.entropy_coef_bs * loss_ent).backward()
                nn.utils.clip_grad_norm_(self.bs_actor.parameters(), self.max_grad_norm)
                self.bs_actor_optim.step()

                bs_epoch += float(loss_pi.item())
                bs_ent_epoch += float(loss_ent.item())
                bs_cnt += 1

            losses["critic_ue"] += c_ue_epoch / max(1, c_ue_cnt)
            losses["critic_bs"] += c_bs_epoch / max(1, c_bs_cnt)
            losses["actor_ue"] += ue_epoch / max(1, ue_cnt)
            losses["entropy_ue"] += ue_ent_epoch / max(1, ue_cnt)
            losses["actor_bs"] += bs_epoch / max(1, bs_cnt)
            losses["entropy_bs"] += bs_ent_epoch / max(1, bs_cnt)

        for k in losses:
            losses[k] /= self.n_epochs

        self.reset_rollout()
        return losses

    # =========================================================
    # Train / Eval
    # =========================================================
    def train(
        self,
        n_steps: int,
        update_interval: int = 128,
        save_npz_path: Optional[str] = None,
        steps_per_episode: int = 10000,
    ):
        """
        Episodic pure-LyMARL training (no handover queue).

        The actor/critic networks, optimizers, and ValueNorm statistics are
        carried across episodes. At every episode boundary, env.reset()
        reinitializes the topology/channel state and all virtual queues using
        the same reset convention as the HAPPO environment:

            Q_u = 0.1, Z_b = 0.0.

        The final transition of each episode is marked done=True and uses zero
        bootstrap values so GAE does not connect two independent episodes.
        """
        n_steps = int(n_steps)
        steps_per_episode = int(steps_per_episode)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be positive")
        if n_steps % steps_per_episode != 0:
            raise ValueError(
                f"n_steps={n_steps} must be divisible by "
                f"steps_per_episode={steps_per_episode}."
            )

        n_episodes = n_steps // steps_per_episode

        print("\n" + "=" * 100)
        print(" Episodic Hetero-MAPPO Training")
        print("=" * 100)
        print(f"Episodes: {n_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Total train steps: {n_steps}")
        print(f"Update interval: {update_interval}")
        print(f"Hard constraint during training: {self.env.use_hard_constraint}")
        print("Queue reset per episode: Q=0.1, Z=0.0")
        print("=" * 100 + "\n")

        throughput_history, fairness_history, slot_rates = [], [], []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        queue_history = {"Q_u": defaultdict(list), "Z_b": defaultdict(list)}
        ue_team_reward_hist, ue_per_user_reward_hist = [], []
        bs_reward_vec_hist, bs_reward_mean_hist = [], []

        handover_ratio_history, handover_flag_history = [], []
        request_switch_ratio_history, request_switch_flag_history = [], []
        cond_ho_numerator_history, cond_ho_denominator_history = [], []
        ho_blocked_candidate_history = []
        ho_no_valid_candidate_bs_history = []
        ho_safety_forced_off_history = []
        ho_forced_off_history = []
        request_bs_history, serving_bs_history = [], []
        episode_index_history, episode_step_history = [], []

        global_step = 0

        for episode in range(n_episodes):
            # Full environment reset: positions, channel, queues, previous
            # association/service memory, ON history, and budget counters.
            local_obs, global_obs = self.env.reset()
            print(
                f"\n[EPISODE {episode + 1}/{n_episodes}] "
                f"reset Q=0.1, Z=0.0"
            )

            for episode_step in range(steps_per_episode):
                global_step += 1
                episode_done = (episode_step == steps_per_episode - 1)

                (ue_actions, ue_logp_np, ue_ent_np, ue_masks_np,
                 bs_actions, bs_logp_np, bs_ent_np, bs_obs_np, bs_masks_np, cand_lists,
                 v_ue_n, v_bs_n_np) = self.select_actions(local_obs, global_obs)

                next_local_obs, next_global_obs, info, _ = self.env.step_joint(
                    ue_actions=ue_actions,
                    bs_actions=bs_actions,
                    cand_lists=cand_lists,
                )

                with torch.no_grad():
                    if episode_done:
                        # Terminal transition: do not bootstrap into next episode.
                        nv_ue_n = 0.0
                        nv_bs_n = np.zeros(self.env.n_bs, dtype=np.float32)
                    else:
                        next_global_t = torch.as_tensor(
                            next_global_obs, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        nv_ue_n = float(self.critic_ue(next_global_t).squeeze(0).item())
                        nv_bs_n = (
                            self.critic_bs(next_global_t)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                rew_ue = float(info["ue_team_reward"])
                rew_bs = np.asarray(info["bs_rewards"], dtype=np.float32).reshape(-1)
                self.store_step(
                    local_obs, global_obs, ue_actions, ue_logp_np, ue_masks_np,
                    bs_actions, bs_logp_np, bs_obs_np, bs_masks_np, cand_lists,
                    rew_ue, rew_bs, float(v_ue_n), float(nv_ue_n),
                    v_bs_n_np, nv_bs_n, episode_done,
                )

                throughput_history.append(float(info["total_throughput"]))
                rates_this_slot = [
                    float(info["served_rates"][u.ue_id]) for u in self.env.users
                ]
                slot_rates.append(rates_this_slot)
                fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

                handover_ratio_history.append(float(info["handover_ratio"]))
                handover_flag_history.append([
                    float(info["handover_flags"][u.ue_id]) for u in self.env.users
                ])
                request_switch_ratio_history.append(float(info["request_switch_ratio"]))
                request_switch_flag_history.append([
                    float(info["request_switch_flags"][u.ue_id]) for u in self.env.users
                ])
                cond_ho_numerator_history.append(int(info["cond_ho_numerator"]))
                cond_ho_denominator_history.append(int(info["cond_ho_denominator"]))
                ho_blocked_candidate_history.append(
                    int(info["ho_blocked_candidate_count"])
                )
                ho_no_valid_candidate_bs_history.append(
                    int(info["ho_no_valid_candidate_bs_count"])
                )
                ho_safety_forced_off_history.append(
                    int(info["ho_safety_forced_off_count"])
                )
                ho_forced_off_history.append(int(info["ho_forced_off_count"]))
                request_bs_history.append([
                    int(info["current_request_bs"][u.ue_id]) for u in self.env.users
                ])
                serving_bs_history.append([
                    int(info["current_serving_bs"][u.ue_id]) for u in self.env.users
                ])
                episode_index_history.append(episode)
                episode_step_history.append(episode_step)

                for bs_id, power in info["power_consumed"].items():
                    power_history[bs_id].append(float(power))
                for ue_id, q_val in info["Q_u"].items():
                    queue_history["Q_u"][ue_id].append(float(q_val))
                for bs_id, z_val in info["Z_b"].items():
                    queue_history["Z_b"][bs_id].append(float(z_val))

                ue_team_reward_hist.append(rew_ue)
                ue_per_user_reward_hist.append([
                    float(info["ue_per_user_rewards"][u.ue_id])
                    for u in self.env.users
                ])
                bs_reward_vec_hist.append(rew_bs.tolist())
                bs_reward_mean_hist.append(float(rew_bs.mean()))

                local_obs, global_obs = next_local_obs, next_global_obs

                if global_step % update_interval == 0:
                    losses = self.update()
                    if losses:
                        print(
                            f"[UPDATE] Global step {global_step} | "
                            f"UE_Actor:{losses['actor_ue']:.4f} | "
                            f"BS_Actor:{losses['actor_bs']:.4f} | "
                            f"C_UE:{losses['critic_ue']:.4f} | "
                            f"C_BS:{losses['critic_bs']:.4f} | "
                            f"Ent(UE):{losses['entropy_ue']:.4f} | "
                            f"Ent(BS):{losses['entropy_bs']:.4f}"
                        )

                log_window = 1000
                if global_step % log_window == 0:
                    on_parts = []
                    for bs in self.env.base_stations:
                        hist = list(self.env.bs_on_hist[bs.bs_id])
                        on_val = float(np.mean(hist[-log_window:])) if hist else 0.0
                        on_parts.append(f"BS{bs.bs_id}:{on_val:.3f}")
                    print(
                        f"Step {global_step:6d} | "
                        f"Ep:{episode + 1}/{n_episodes} "
                        f"EpStep:{episode_step + 1:5d} | "
                        f"Thr:{np.mean(throughput_history[-log_window:]):.3f} | "
                        f"Fair:{fairness_history[-1]:.3f} | "
                        f"HO:{np.mean(handover_ratio_history[-log_window:]):.4f} | "
                        f"ReqSwitch:{np.mean(request_switch_ratio_history[-log_window:]):.4f} | "
                        f"ON(1000): {' '.join(on_parts)} | "
                        f"UETeamRew:{np.mean(ue_team_reward_hist[-log_window:]):.3f} | "
                        f"BSTeamRew:{np.mean(bs_reward_mean_hist[-log_window:]):.3f}"
                    )

            # Flush the partial rollout at every episode boundary so no buffer
            # crosses from one independent episode into the next.
            if len(self.rb["dones"]) > 0:
                self.update()

        results = {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "queue_history": queue_history,
            "ue_team_reward": ue_team_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,
            "bs_reward_vec": bs_reward_vec_hist,
            "bs_reward_mean": bs_reward_mean_hist,
            "handover_ratio": handover_ratio_history,
            "handover_flags": handover_flag_history,
            "request_switch_ratio": request_switch_ratio_history,
            "request_switch_flags": request_switch_flag_history,
            "cond_ho_numerator": cond_ho_numerator_history,
            "cond_ho_denominator": cond_ho_denominator_history,
            "ho_blocked_candidate_count": ho_blocked_candidate_history,
            "ho_no_valid_candidate_bs_count": ho_no_valid_candidate_bs_history,
            "ho_safety_forced_off_count": ho_safety_forced_off_history,
            "ho_forced_off_count": ho_forced_off_history,
            "request_bs_history": request_bs_history,
            "serving_bs_history": serving_bs_history,
            "episode_index": episode_index_history,
            "episode_step": episode_step_history,
            "n_episodes": int(n_episodes),
            "steps_per_episode": int(steps_per_episode),
            "kappa": float(self.env.kappa),
        }
        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="train")
        return results

    @torch.no_grad()
    def evaluate(self, n_steps: int, save_npz_path: Optional[str] = None,
                 imperfect_csi: bool = False, csi_error_var: float = 0.0):
        print("\n" + "=" * 84)
        print(" EVALUATION (No Learning)")
        print(f"{'='*84}")
        print(f"Total eval steps: {n_steps}")
        print(f"Hard constraints during evaluation: {self.env.use_hard_constraint}")

        self.env.set_imperfect_csi(imperfect_csi, csi_error_var)
        self.ue_actor.eval(); self.bs_actor.eval(); self.critic_ue.eval(); self.critic_bs.eval()

        throughput_history, fairness_history, slot_rates = [], [], []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        ue_team_reward_hist, ue_per_user_reward_hist = [], []
        bs_reward_vec_hist, bs_reward_mean_hist = [], []
        handover_ratio_history, handover_flag_history = [], []
        request_switch_ratio_history, request_switch_flag_history = [], []
        cond_ho_numerator_history, cond_ho_denominator_history = [], []
        ho_blocked_candidate_history = []
        ho_no_valid_candidate_bs_history = []
        ho_safety_forced_off_history = []
        ho_forced_off_history = []
        request_bs_history, serving_bs_history = [], []

        local_obs, global_obs = self.env.reset()
        log_window = 1000

        for step in range(n_steps):
            (ue_actions, _, _, _, bs_actions, _, _, _, _, cand_lists, _, _) = self.select_actions(local_obs, global_obs)
            next_local_obs, next_global_obs, info, _ = self.env.step_joint(ue_actions, bs_actions, cand_lists)

            throughput_history.append(float(info["total_throughput"]))
            rates_this_slot = [float(info["served_rates"][u.ue_id]) for u in self.env.users]
            slot_rates.append(rates_this_slot)
            fairness_history.append(self.env.calculate_jain_fairness(slot_rates))
            handover_ratio_history.append(float(info["handover_ratio"]))
            handover_flag_history.append([float(info["handover_flags"][u.ue_id]) for u in self.env.users])
            request_switch_ratio_history.append(float(info["request_switch_ratio"]))
            request_switch_flag_history.append([float(info["request_switch_flags"][u.ue_id]) for u in self.env.users])
            cond_ho_numerator_history.append(int(info["cond_ho_numerator"]))
            cond_ho_denominator_history.append(int(info["cond_ho_denominator"]))
            ho_blocked_candidate_history.append(
                int(info["ho_blocked_candidate_count"])
            )
            ho_no_valid_candidate_bs_history.append(
                int(info["ho_no_valid_candidate_bs_count"])
            )
            ho_safety_forced_off_history.append(
                int(info["ho_safety_forced_off_count"])
            )
            ho_forced_off_history.append(int(info["ho_forced_off_count"]))
            request_bs_history.append([int(info["current_request_bs"][u.ue_id]) for u in self.env.users])
            serving_bs_history.append([int(info["current_serving_bs"][u.ue_id]) for u in self.env.users])

            for bs_id, power in info["power_consumed"].items():
                power_history[bs_id].append(float(power))

            ue_team_reward_hist.append(float(info["ue_team_reward"]))
            ue_per_user_reward_hist.append([float(info["ue_per_user_rewards"][u.ue_id]) for u in self.env.users])
            bs_vec = np.asarray(info["bs_rewards"], dtype=np.float32).reshape(-1)
            bs_reward_vec_hist.append(bs_vec.tolist())
            bs_reward_mean_hist.append(float(bs_vec.mean()))
            local_obs, global_obs = next_local_obs, next_global_obs

            if (step + 1) % log_window == 0:
                on_text = " ".join(
                    f"BS{bs.bs_id}:{np.mean(np.asarray(power_history[bs.bs_id][-log_window:]) > 0):.3f}"
                    for bs in self.env.base_stations
                )
                print(
                    f"[EVAL] Step {step+1:5d} | Thr:{np.mean(throughput_history[-log_window:]):.3f} | "
                    f"JFI(block):{compute_block_jfi(slot_rates[-log_window:], log_window):.3f} | "
                    f"HO:{np.mean(handover_ratio_history[-log_window:]):.4f} | "
                    f"ReqSwitch:{np.mean(request_switch_ratio_history[-log_window:]):.4f} | ON:{on_text}"
                )

        handover_arr = np.asarray(handover_flag_history, dtype=np.float64)
        request_switch_arr = np.asarray(request_switch_flag_history, dtype=np.float64)
        cond_num = float(np.sum(cond_ho_numerator_history))
        cond_den = float(np.sum(cond_ho_denominator_history))
        on_mat = np.stack([
            (np.asarray(power_history[bs.bs_id], dtype=np.float64) > 0.0).astype(np.float64)
            for bs in self.env.base_stations
        ], axis=1)

        final_metrics = {
            "throughput": float(np.mean(throughput_history)),
            "jfi": compute_block_jfi(slot_rates, block_size=1000),
            "ho_ratio": float(handover_arr.mean()) if handover_arr.size else 0.0,
            "cond_ho": float(cond_num / cond_den) if cond_den > 0 else 0.0,
            "req_switch": float(request_switch_arr.mean()) if request_switch_arr.size else 0.0,
            "on_ratio": float(on_mat.mean()) if on_mat.size else 0.0,
        }

        print("\n" + "=" * 80)
        print("FINAL EVALUATION METRICS")
        for key, value in final_metrics.items():
            print(f"{key:12s} = {value:.6f}")
        print("=" * 80)

        results = {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "ue_team_reward": ue_team_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,
            "bs_reward_vec": bs_reward_vec_hist,
            "bs_reward_mean": bs_reward_mean_hist,
            "handover_ratio": handover_ratio_history,
            "handover_flags": handover_flag_history,
            "request_switch_ratio": request_switch_ratio_history,
            "request_switch_flags": request_switch_flag_history,
            "cond_ho_numerator": cond_ho_numerator_history,
            "cond_ho_denominator": cond_ho_denominator_history,
            "ho_blocked_candidate_count": ho_blocked_candidate_history,
            "ho_no_valid_candidate_bs_count": ho_no_valid_candidate_bs_history,
            "ho_safety_forced_off_count": ho_safety_forced_off_history,
            "ho_forced_off_count": ho_forced_off_history,
            "request_bs_history": request_bs_history,
            "serving_bs_history": serving_bs_history,
            "kappa": float(self.env.kappa),
            "final_throughput": final_metrics["throughput"],
            "final_jfi": final_metrics["jfi"],
            "final_ho_ratio": final_metrics["ho_ratio"],
            "final_cond_ho": final_metrics["cond_ho"],
            "final_req_switch": final_metrics["req_switch"],
            "final_on_ratio": final_metrics["on_ratio"],
        }
        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="eval")
        return results

    def save_results_npz(self, results: Dict, npz_path: str, tag: str = "run"):
        os.makedirs(os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".", exist_ok=True)

        thr = np.asarray(results.get("throughput_history", []), dtype=np.float32)
        fair = np.asarray(results.get("fairness_history", []), dtype=np.float32)
        slot_rates = np.asarray(results.get("slot_rates", []), dtype=np.float32)
        ue_team = np.asarray(results.get("ue_team_reward", []), dtype=np.float32)
        ue_per_user = np.asarray(results.get("ue_per_user_reward", []), dtype=np.float32)
        bs_mean = np.asarray(results.get("bs_reward_mean", []), dtype=np.float32)
        bs_vec = np.asarray(results.get("bs_reward_vec", []), dtype=np.float32)

        if ue_per_user.ndim == 2 and ue_per_user.shape[0] > 0:
            mean_user_reward_step = ue_per_user.mean(axis=1).astype(np.float32)
            mean_user_reward_ma100 = moving_avg(mean_user_reward_step, 100)
        else:
            mean_user_reward_step = np.asarray([], dtype=np.float32)
            mean_user_reward_ma100 = np.asarray([], dtype=np.float32)

        reward_x_500, user_mean_reward_500 = block_avg_1d(mean_user_reward_step, 500)
        bs_reward_x_500, bs_mean_reward_500 = block_avg_1d(bs_mean, 500)

        power_hist = results.get("power_history", {})
        bs_ids_sorted = sorted(power_hist.keys()) if isinstance(power_hist, dict) else []
        power_mat = np.stack(
            [np.asarray(power_hist[bs_id], dtype=np.float32) for bs_id in bs_ids_sorted], axis=0
        ) if bs_ids_sorted else np.zeros((0, len(thr)), dtype=np.float32)

        np.savez_compressed(
            npz_path,
            tag=str(tag), n_users=int(self.env.n_agents), n_bs=int(self.env.n_bs),
            throughput=thr, fairness=fair, slot_rates=slot_rates,
            ue_team_reward=ue_team, ue_per_user_reward=ue_per_user,
            mean_user_reward_step=mean_user_reward_step,
            mean_user_reward_ma100=mean_user_reward_ma100,
            bs_reward_mean=bs_mean, bs_reward_vec=bs_vec,
            reward_x_500=reward_x_500, user_mean_reward_500=user_mean_reward_500,
            bs_reward_x_500=bs_reward_x_500, bs_mean_reward_500=bs_mean_reward_500,
            bs_ids=np.asarray(bs_ids_sorted, dtype=np.int32), power_mat=power_mat,
            handover_ratio=np.asarray(results.get("handover_ratio", []), dtype=np.float32),
            handover_flags=np.asarray(results.get("handover_flags", []), dtype=np.float32),
            request_switch_ratio=np.asarray(results.get("request_switch_ratio", []), dtype=np.float32),
            request_switch_flags=np.asarray(results.get("request_switch_flags", []), dtype=np.float32),
            cond_ho_numerator=np.asarray(results.get("cond_ho_numerator", []), dtype=np.float32),
            cond_ho_denominator=np.asarray(results.get("cond_ho_denominator", []), dtype=np.float32),
            ho_blocked_candidate_count=np.asarray(
                results.get("ho_blocked_candidate_count", []), dtype=np.int32
            ),
            ho_no_valid_candidate_bs_count=np.asarray(
                results.get("ho_no_valid_candidate_bs_count", []), dtype=np.int32
            ),
            ho_safety_forced_off_count=np.asarray(
                results.get("ho_safety_forced_off_count", []), dtype=np.int32
            ),
            ho_forced_off_count=np.asarray(results.get("ho_forced_off_count", []), dtype=np.int32),
            request_bs_history=np.asarray(results.get("request_bs_history", []), dtype=np.int32),
            serving_bs_history=np.asarray(results.get("serving_bs_history", []), dtype=np.int32),
            kappa=np.float32(results.get("kappa", self.env.kappa)),
            final_throughput=np.float32(results.get("final_throughput", np.nan)),
            final_jfi=np.float32(results.get("final_jfi", np.nan)),
            final_ho_ratio=np.float32(results.get("final_ho_ratio", np.nan)),
            final_cond_ho=np.float32(results.get("final_cond_ho", np.nan)),
            final_req_switch=np.float32(results.get("final_req_switch", np.nan)),
            final_on_ratio=np.float32(results.get("final_on_ratio", np.nan)),
        )
        print(f"✅ Saved results npz: {npz_path}")

# ============================================================
# Plotting utilities
# ============================================================

def plot_training_results(results: Dict, env: MAPPOEnvironment, block_size: int = 100):
    """
    Plot training metrics after training finishes.

    Figures:
      1) Q_u trend for each user
      2) Z_b trend for each BS
      3) Average throughput every block_size steps
      4) BS ON-ratio every block_size steps
    """
    import matplotlib.pyplot as plt

    throughput = np.asarray(results.get("throughput_history", []), dtype=np.float32)
    queue_history = results.get("queue_history", {})
    power_history = results.get("power_history", {})

    user_ids = [u.ue_id for u in env.users]
    bs_ids = [bs.bs_id for bs in env.base_stations]

    if throughput.size == 0:
        print("[PLOT] No throughput history found. Skip plotting.")
        return

    # --------------------------------------------------
    # Helper: block average
    # --------------------------------------------------
    def block_average_1d(x: np.ndarray, block: int):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n_blocks = len(x) // block
        if n_blocks == 0:
            return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)
        x = x[:n_blocks * block]
        y = x.reshape(n_blocks, block).mean(axis=1)
        xs = np.arange(1, n_blocks + 1, dtype=np.int32) * block
        return xs, y

    # ==================================================
    # 1. Q_u trend for each user
    # ==================================================
    plt.figure(figsize=(12, 6))
    for uid in user_ids:
        q_vals = np.asarray(queue_history["Q_u"][uid], dtype=np.float32)
        plt.plot(q_vals, linewidth=1.0, alpha=0.8, label=f"UE {uid}")

    plt.xlabel("Training step")
    plt.ylabel(r"$Q_u$")
    plt.title("Queue Backlog Trend for Each User")
    plt.grid(True)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()

    # ==================================================
    # 2. Z_b trend for each BS
    # ==================================================
    plt.figure(figsize=(12, 5))
    for bid in bs_ids:
        z_vals = np.asarray(queue_history["Z_b"][bid], dtype=np.float32)
        plt.plot(z_vals, linewidth=1.5, label=f"BS {bid}")

    plt.xlabel("Training step")
    plt.ylabel(r"$Z_b$")
    plt.title("Virtual Power Queue Trend for Each BS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==================================================
    # 3. Average throughput every block_size steps
    # ==================================================
    x_thr, throughput_avg = block_average_1d(throughput, block_size)

    plt.figure(figsize=(12, 5))
    plt.plot(x_thr, throughput_avg, linewidth=2.0)
    plt.xlabel("Training step")
    plt.ylabel("Average throughput")
    plt.title(f"Average Throughput Every {block_size} Steps")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==================================================
    # 4. BS ON-ratio every block_size steps
    # ==================================================
    plt.figure(figsize=(12, 5))
    for bid in bs_ids:
        power_vals = np.asarray(power_history[bid], dtype=np.float32)
        on_vals = (power_vals > 0.0).astype(np.float32)
        x_on, on_ratio = block_average_1d(on_vals, block_size)
        plt.plot(x_on, on_ratio, linewidth=1.8, label=f"BS {bid}")

    plt.axhline(
        y=env.power_budget_ratio,
        linestyle="--",
        linewidth=1.5,
        label=r"Target $\rho$"
    )

    plt.xlabel("Training step")
    plt.ylabel("ON-ratio")
    plt.title(f"BS ON-ratio Every {block_size} Steps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# Multi-seed experiment and summary
# ============================================================

def make_topology(seed: int, tx_power_dbm: float = 20.0):
    set_seed(seed)
    area_size = 100
    sbs_positions = generate_triangle_coverage(area_size, 35)
    sbs_list = [
        SmallCellBaseStation(i + 1, pos, tx_power_dbm, 35)
        for i, pos in enumerate(sbs_positions)
    ]
    users = [
        UserEquipment(i + 1, (np.random.uniform(10, 90), np.random.uniform(10, 90)))
        for i in range(20)
    ]
    return sbs_list, users


def build_env(seed: int, use_hard_constraint: bool, kappa: float = 0.03):
    sbs_list, users = make_topology(seed=seed, tx_power_dbm=20.0)
    return MAPPOEnvironment(
        base_stations=sbs_list,
        users=users,
        V=5.0,
        power_budget_ratio=0.6,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=1000,
        bs_top_k=5,
        hard_window_len=10000,
        alpha3=0.0,
        bs_over_penalty=0.0,
        alpha_rate=0.0,
        beta_z=0.0,
        eta_q=1.0,
        use_hard_constraint=use_hard_constraint,
        use_imperfect_csi=False,
        csi_error_var=0.0,
        kappa=kappa,
    )


def build_trainer(env):
    return MAPPOTrainer(
        env=env,
        lr_actor_ue=3e-4, lr_actor_bs=3e-4,
        lr_critic_ue=1e-3, lr_critic_bs=1e-3,
        gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
        entropy_coef_ue=0.05, entropy_coef_bs=0.05,
        value_coef_ue=0.5, value_coef_bs=0.5,
        n_epochs=4, minibatch_size=256,
    )


def summarize_results(root: str, train_seeds, eval_seeds):
    metric_keys = {
        "Throughput": "final_throughput",
        "JFI": "final_jfi",
        "HO ratio": "final_ho_ratio",
        "CondHO": "final_cond_ho",
        "ReqSwitch": "final_req_switch",
        "ON ratio": "final_on_ratio",
    }
    per_train = {name: [] for name in metric_keys}

    for train_seed in train_seeds:
        per_eval = {name: [] for name in metric_keys}
        for eval_seed in eval_seeds:
            path = os.path.join(root, f"seed_{train_seed}", f"eval_seed_{eval_seed}.npz")
            data = np.load(path, allow_pickle=False)
            for name, key in metric_keys.items():
                per_eval[name].append(float(data[key]))
        for name in metric_keys:
            per_train[name].append(float(np.mean(per_eval[name])))

    print("\n" + "=" * 96)
    print("PURE LyMARL FINAL SUMMARY (NO HO QUEUE)")
    print("Each train seed is first averaged over evaluation seeds; mean/std are across train seeds.")
    print("=" * 96)
    for name, values in per_train.items():
        arr = np.asarray(values, dtype=np.float64)
        print(f"{name:12s} = {arr.mean():.6f} +/- {arr.std(ddof=0):.6f}")
    print("=" * 96)

def summarize_kappa_results(
    base_root: str,
    kappa_list,
    train_seeds,
    eval_seeds,
):
    metric_keys = {
        "throughput": "final_throughput",
        "jfi": "final_jfi",
        "ho": "final_ho_ratio",
        "cond_ho": "final_cond_ho",
        "req_switch": "final_req_switch",
        "on_ratio": "final_on_ratio",
    }

    print("\n" + "=" * 150)
    print("PURE LyMARL FINAL SUMMARY (NO HO QUEUE)")
    print(
        "Each train seed is first averaged over evaluation seeds; "
        "mean/std below are across train seeds."
    )
    print("=" * 150)

    for kappa in kappa_list:
        kappa_tag = f"{kappa:.3f}"
        root_prefix = f"LyMARL_kappa_{kappa_tag}_seed_"

        # train seed별 결과
        per_train = {
            metric_name: []
            for metric_name in metric_keys
        }

        for train_seed in train_seeds:
            # 하나의 train seed에서 eval seed 평균
            per_eval = {
                metric_name: []
                for metric_name in metric_keys
            }

            for eval_seed in eval_seeds:
                path = os.path.join(
                    base_root,
                    f"{root_prefix}{train_seed}",
                    f"eval_seed_{eval_seed}.npz",
                )

                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Evaluation file not found: {path}"
                    )

                with np.load(path, allow_pickle=False) as data:
                    for metric_name, npz_key in metric_keys.items():
                        per_eval[metric_name].append(
                            float(data[npz_key])
                        )

            for metric_name in metric_keys:
                per_train[metric_name].append(
                    float(np.mean(per_eval[metric_name]))
                )

        # train seed 간 mean/std
        summary = {}

        for metric_name, values in per_train.items():
            arr = np.asarray(values, dtype=np.float64)

            summary[f"{metric_name}_mean"] = float(
                np.mean(arr)
            )
            summary[f"{metric_name}_std"] = float(
                np.std(arr, ddof=0)
            )

        # SameBS = 1 - CondHO
        summary["same_bs_mean"] = (
            1.0 - summary["cond_ho_mean"]
        )

        # std는 1-X 변환이므로 동일
        summary["same_bs_std"] = summary["cond_ho_std"]

        print(
            f"kappa={kappa:.3f} | "
            f"LyMARL | "
            f"Throughput={summary['throughput_mean']:.4f}"
            f" +/- {summary['throughput_std']:.4f} | "
            f"JFI={summary['jfi_mean']:.4f}"
            f" +/- {summary['jfi_std']:.4f} | "
            f"ON={summary['on_ratio_mean']:.4f}"
            f" +/- {summary['on_ratio_std']:.4f} | "
            f"HO={summary['ho_mean']:.4f}"
            f" +/- {summary['ho_std']:.4f} | "
            f"ReqSwitch={summary['req_switch_mean']:.4f}"
            f" +/- {summary['req_switch_std']:.4f} | "
            f"CondHO={summary['cond_ho_mean']:.4f}"
            f" +/- {summary['cond_ho_std']:.4f} | "
            f"SameBS={summary['same_bs_mean']:.4f}"
            f" +/- {summary['same_bs_std']:.4f}"
        )

    print("=" * 150 + "\n")

if __name__ == "__main__":
    TRAIN_SEEDS = [0, 1, 2]
    EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

    N_EPISODES = 10
    STEPS_PER_EPISODE = 10000
    TRAIN_STEPS = N_EPISODES * STEPS_PER_EPISODE
    EVAL_STEPS = 10000

    # Pure LyMARL is trained without a handover queue or handover cost.
    # kappa is used only during hard evaluation. Exhausted HO candidates are
    # filtered per UE-BS pair; a BS is forced to NONE/OFF only when no valid
    # requesting candidate remains (or its energy budget is exhausted).
    KAPPA_LIST = [0.010]

    UPDATE_INTERVAL = 128
    BASE_SAVE_ROOT = "results/results_kappa"
    OVERWRITE_EXISTING = False

    for train_seed in TRAIN_SEEDS:
        train_kappa = KAPPA_LIST[0]
        train_dir = os.path.join(
            BASE_SAVE_ROOT,
            f"LyMARL_kappa_{train_kappa:.3f}_seed_{train_seed}",
        )
        os.makedirs(train_dir, exist_ok=True)

        model_path = os.path.join(train_dir, "model.pt")
        train_npz_path = os.path.join(train_dir, "train.npz")

        if OVERWRITE_EXISTING or not os.path.exists(model_path):
            print(
                "\n"
                + "#" * 100
                + f"\nPURE LyMARL | TRAIN SEED={train_seed}\n"
                + "#" * 100
            )

            train_env = build_env(
                seed=train_seed,
                use_hard_constraint=False,
                kappa=train_kappa,
            )
            trainer = build_trainer(train_env)

            trainer.train(
                n_steps=TRAIN_STEPS,
                update_interval=UPDATE_INTERVAL,
                save_npz_path=train_npz_path,
                steps_per_episode=STEPS_PER_EPISODE,
            )
            trainer.save_model(model_path)
        else:
            print(f"[SKIP TRAIN] Existing model: {model_path}")

        for kappa in KAPPA_LIST:
            eval_dir = os.path.join(
                BASE_SAVE_ROOT,
                f"LyMARL_kappa_{kappa:.3f}_seed_{train_seed}",
            )
            os.makedirs(eval_dir, exist_ok=True)

            for eval_seed in EVAL_SEEDS:
                eval_path = os.path.join(
                    eval_dir,
                    f"eval_seed_{eval_seed}.npz",
                )

                if (
                    not OVERWRITE_EXISTING
                    and os.path.exists(eval_path)
                ):
                    print(f"[SKIP EVAL] Existing result: {eval_path}")
                    continue

                print(
                    f"\n[EVAL] Pure LyMARL | "
                    f"kappa={kappa:.3f}, "
                    f"train_seed={train_seed}, "
                    f"eval_seed={eval_seed}"
                )

                eval_env = build_env(
                    seed=eval_seed,
                    use_hard_constraint=True,
                    kappa=kappa,
                )
                eval_trainer = build_trainer(eval_env)
                eval_trainer.load_model(
                    model_path,
                    load_optim=False,
                )

                set_seed(eval_seed)
                eval_trainer.evaluate(
                    EVAL_STEPS,
                    eval_path,
                    imperfect_csi=False,
                    csi_error_var=0.0,
                )

    summarize_kappa_results(
        base_root=BASE_SAVE_ROOT,
        kappa_list=KAPPA_LIST,
        train_seeds=TRAIN_SEEDS,
        eval_seeds=EVAL_SEEDS,
    )
