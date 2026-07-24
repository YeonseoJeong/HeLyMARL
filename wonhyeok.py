# ============================================================
# HeLyMARL / LyMARL HAPPO single-file runner for 9-BS deployment
#
# External project files required in the same directory / PYTHONPATH:
#   - basestation.py
#   - core.py
#   - user_equipment.py
#
# This file combines:
#   - utils_happo
#   - networks_happo
#   - env_happo
#   - trainer_happo
#   - main_happo
#
# Main experiment:
#   - fixed 9-BS layout: 3x3 small-cell grid
#   - episodic training/evaluation: 10000 steps = 1 episode
#   - train one independent model per kappa on UE=20
#   - evaluate each model for 10 x 10,000 steps using the same 10 seeds
#   - save 10-run means/stds in plot-oriented NPZ files
# ============================================================

import os
import csv
import random
import secrets
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque

from env.basestation import BaseStation, SmallCellBaseStation
from env.core import plot_associations  # optional helper from your existing core.py, not required for training
from env.user_equipment import UserEquipment


# =========================================================
# Utils
# =========================================================
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


# =========================================================
# Networks
# =========================================================
class ValueNorm(nn.Module):
    """
    Running mean/std for scalar targets.
    Used for centralized critic target normalization.
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
    Kept for compatibility, although this script uses scalar centralized critic.
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
    """Shared actor network for all UEs."""
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
    """Shared actor network for all BSs."""
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


class CentralizedCritic(nn.Module):
    """Shared centralized critic for HAPPO. Outputs scalar V(s)."""
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


class HAPPOEnvironment:
    """
    Environment for UE-BS heterogeneous HAPPO.

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
        on_window: int = 100,
        bs_top_k: int = 5,
        hard_window_len: int = 10000,
        bs_over_penalty: float = 50.0,
        eta_q: float = 1.0,
        alpha_rate: float = 3.0,
        beta_z: float = 1.0,
        use_hard_constraint: bool = False,
        lambda_E: float = 1.0,
        kappa: float = 0.02
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

        self.bs_top_k = int(bs_top_k)
        assert self.bs_top_k >= 1

        self.hard_window_len = int(hard_window_len)
        assert self.hard_window_len >= 1

        self.use_hard_constraint = bool(use_hard_constraint)
        self.lambda_E = float(lambda_E)
        self.kappa = float(kappa)

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
        self.Z_b = {bs.bs_id: 0.01 for bs in self.base_stations}
        self.G_u = {u.ue_id: 0.0 for u in users}
        self.R_max = {u.ue_id: 5.0 for u in users}

        # most recent serving BS memory m_u(t-1), 0 means never served before
        self.m_u = {u.ue_id: 0 for u in users}

        # average handover budget H_bar = H_max / T ≈ kappa
        self.H_bar = {u.ue_id: self.kappa for u in users} 

        # Channel / mobility
        self.noise_dbm = -174 + 10 * np.log10(500e6) + 5
        self.noise_watts = 10 ** (self.noise_dbm / 10) / 1000
        self.mobility_speed = 1
        self.area_size = 100
        self.channel_gains = defaultdict(dict)
        self.fading_std = 4.0

        self.timestep = 0

        # UE action: [BS0..BS(n_bs-1)]
        #self.no_request_action = self.n_bs
        self.action_dim = self.n_bs

        # BS action: Top-K candidates + 0: inactive
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

        # Observation dimensions
        # UE local: [Q_u, G_u] +m_u_onehot(n_bs+1) + rates(n_bs) + Z_b(n_bs)
        self.local_obs_dim = 2 + (self.n_bs + 1) + 2 * self.n_bs

        # BS local: [Z_b] + top-K scores [(Q_u, G_u, rate)]
        self.bs_obs_dim = 1 + 3 * self.bs_top_k

        # Global:
        # per UE: [Q_u, G_u, m_u_onehot(n_bs+1), rates(n_bs)] => self.n_agents * (2 + (self.n_bs + 1) + self.n_bs)
        # per BS: [Z_b] => n_bs
        self.global_obs_dim = self.n_agents * (2 + (self.n_bs + 1) + self.n_bs) + self.n_bs

        self._rate_cache = np.zeros((self.n_agents, self.n_bs), dtype=np.float32)
        self.no_coverage_count = 0

        print(f"\n{'='*96}")
        print(" HAPPO Environment")
        print(f"{'='*96}")
        print(f"#UE={self.n_agents} | #BS={self.n_bs} | UE_action_dim={self.action_dim} | BS_action_dim={self.bs_action_dim}")
        print(f"V={self.V} | power_budget_ratio={self.power_budget_ratio} | lambda_E={self.lambda_E} | kappa={self.kappa}")
        print("UE local obs = [Q_u, G_u, m_u_onehot, rates_to_all_BS, Z_all_BS]")
        print("BS local obs = [Z_b, (Q_u, G_u, rate) for top-K requesters]")
        print("Global reward = sum_u[Q_u(t)R_u(t)] - sum_b[(Z_b(t)+lambda_E)e_b(t)] - sum_u[G_u(t)h_u(t)]")
        print(f"Hard constraint enabled: {self.use_hard_constraint}")
        print(f"local_obs_dim={self.local_obs_dim} | bs_obs_dim={self.bs_obs_dim} | global_obs_dim={self.global_obs_dim}")
        print(f"{'='*96}\n")

    def set_hard_constraint(self, enabled: bool):
        self.use_hard_constraint = bool(enabled)

    def reset(self):
        self.timestep = 0
        self.no_coverage_count = 0

        for user in self.users:
            user.position = np.array([np.random.uniform(10, 90), np.random.uniform(10, 90)])

        self.update_channel_gains(0)

        self.Q_u = {u.ue_id: 0.1 for u in self.users}
        self.G_u = {u.ue_id: 0.0 for u in self.users}
        self.Z_b = {bs.bs_id: 0.01 for bs in self.base_stations}
        self.R_max = {u.ue_id: 5.0 for u in self.users}

        self.m_u = {u.ue_id: 0 for u in self.users}
        self.H_bar = {u.ue_id: self.kappa for u in self.users}

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
    def _get_memory_onehot(self, ue_id: int) -> List[float]:
        """
        m_u in {0, actual_bs_id}
        one-hot length = n_bs + 1
        index 0: never served yet
        index i+1: base station with index i in self.base_stations
        """
        onehot = [0.0] * (self.n_bs + 1)
        mem_bs_id = int(self.m_u.get(ue_id, 0))

        if mem_bs_id == 0:
            onehot[0] = 1.0
        elif mem_bs_id in self.bs_id_to_index:
            bi = self.bs_id_to_index[mem_bs_id]
            onehot[bi + 1] = 1.0
        else:
            onehot[0] = 1.0

        return onehot

    def _compute_handover_indicator(self, ue_id: int, served_bs_id: Optional[int]) -> float:
        """
        h_u(t) = 1 if user is served now and current serving BS differs from m_u(t-1)
        If user is not served now, h_u(t) = 0
        """
        if served_bs_id is None:
            return 0.0
        prev_mem = self.m_u.get(ue_id, 0)
        if prev_mem == 0:
            return 0.0
        return 1.0 if (served_bs_id != prev_mem) else 0.0

    def _get_bs_on_features(self) -> List[float]:
        feats = []
        for bs in self.base_stations:
            hist = self.bs_on_hist[bs.bs_id]
            feats.append(0.0 if len(hist) == 0 else float(sum(hist) / len(hist)))
        return feats

    def _get_local_observation_by_index(self, ui: int) -> np.ndarray:
        ue = self.users[ui]
        ue_id = ue.ue_id

        obs = [
            float(self.Q_u[ue_id]),
            float(self.G_u[ue_id]),
        ]
        # m_u(t-1) one-hot
        obs.extend(self._get_memory_onehot(ue_id))
        
        # Achievable rates to all BSs
        obs.extend(self._rate_cache[ui, :].tolist())

        # Z_b for all BSs
        for bs in self.base_stations:
            obs.append(float(self.Z_b[bs.bs_id]))

        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.local_obs_dim, f"UE obs dim mismatch: {len(result)} vs {self.local_obs_dim}"
        return result

    def _get_global_observation(self) -> np.ndarray:
        obs = []
        
        for ui, ue in enumerate(self.users):
            ue_id = ue.ue_id
            obs.append(float(self.Q_u[ue_id]))
            obs.append(float(self.G_u[ue_id]))
            obs.extend(self._get_memory_onehot(ue_id))

        for bs in self.base_stations:
            obs.append(float(self.Z_b[bs.bs_id]))

        for ui, ue in enumerate(self.users):
            obs.extend(self._rate_cache[ui, :].tolist())

        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.global_obs_dim, f"Global obs dim mismatch: {len(result)} vs {self.global_obs_dim}"
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
        mask length = n_bs
        [0..n_bs-1]: selectable BSs based on coverage
        """
        user = self.user_map[ue_id]
        mask = np.zeros(self.action_dim, dtype=bool)

        for i, bs in enumerate(self.base_stations):
            mask[i] = bool(bs.can_serve(user.position))

        if not mask[:self.n_bs].any():
            self.no_coverage_count += 1

        return mask

    def build_bs_decision_inputs(self, ue_actions: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """
        Build per-BS observations and masks from UE requests.

        BS local obs:
        [Z_b, (Q_u, G_u, rate) for top-K requesting users]

        BS action:
        0 = inactive
        1..K = choose one among top-K candidates
        """
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}

        for ue_id, a in ue_actions.items():
            a = int(a)
            if not (0 <= a < self.n_bs):
                continue

            bs_id = self.base_stations[a].bs_id
            bs_requests[bs_id].append(ue_id)

        bs_obs_batch = np.zeros((self.n_bs, self.bs_obs_dim), dtype=np.float32)
        bs_mask_batch = np.zeros((self.n_bs, self.bs_action_dim), dtype=bool)
        cand_lists: List[List[int]] = []

        for bi, bs in enumerate(self.base_stations):
            reqs = bs_requests[bs.bs_id]

            scored = []
            for ue_id in reqs:
                ui = self.ue_id_to_index[ue_id]
                rate = float(self._rate_cache[ui, bi])
                if rate <= 0.0:
                    continue

                score = float(self.Q_u[ue_id] * rate)
                scored.append((score, ue_id, rate))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:self.bs_top_k]

            cand = []
            feat_triplets = []

            for (_, ue_id, rate) in top:
                cand.append(ue_id)
                feat_triplets.extend([
                    float(self.Q_u[ue_id]),
                    float(self.G_u[ue_id]),
                    float(rate)
                ])

            while len(cand) < self.bs_top_k:
                cand.append(-1)
                feat_triplets.extend([0.0, 0.0, 0.0])

            cand_lists.append(cand)

            obs = [float(self.Z_b[bs.bs_id])]
            obs.extend(feat_triplets)
            bs_obs_batch[bi, :] = np.array(obs, dtype=np.float32)

            # 1..K for candidates
            for k in range(self.bs_top_k):
                bs_mask_batch[bi, k + 1] = (cand[k] >= 0)
            
            # 0 = inactive is always valid
            bs_mask_batch[bi, 0] = True

        return bs_obs_batch, bs_mask_batch, cand_lists

    # =========================================================
    # Step
    # =========================================================
    def step_joint(self, ue_actions: Dict[int, int], bs_actions: Dict[int, int], cand_lists: List[List[int]]):
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}

        for ue_id, action in ue_actions.items():
            action = int(action)
            assert 0 <= action < self.action_dim, f"Invalid UE action {action}"

            bs_id = self.base_stations[action].bs_id
            bs_requests[bs_id].append(ue_id)

        # Congestion logging
        for bs in self.base_stations:
            self.prev_req_ratio[bs.bs_id] = len(bs_requests[bs.bs_id]) / max(1, self.n_agents)

        # BS selects one UE or NONE
        bs_selections: Dict[int, Optional[int]] = {}
        for bi, bs in enumerate(self.base_stations):
            a_b = int(bs_actions[bs.bs_id])

            # 0 = inactive
            if a_b == 0:
                bs_selections[bs.bs_id] = None
                continue

            cand = cand_lists[bi]
            cand_idx = a_b - 1

            if not (0 <= cand_idx < self.bs_top_k):
                bs_selections[bs.bs_id] = None
                continue

            ue_id = cand[cand_idx]
            if ue_id < 0:
                bs_selections[bs.bs_id] = None
                continue

            if ue_id not in bs_requests[bs.bs_id]:
                bs_selections[bs.bs_id] = None
                continue

            ui = self.ue_id_to_index[ue_id]
            if float(self._rate_cache[ui, bi]) <= 0.0:
                bs_selections[bs.bs_id] = None
            else:
                bs_selections[bs.bs_id] = ue_id

        # Optional hard constraint enforcement for evaluation
        if self.use_hard_constraint:
            for bs in self.base_stations:
                used = self.bs_on_used_in_window[bs.bs_id]
                limit = self.hard_on_limit[bs.bs_id]
                if used >= limit:
                    bs_selections[bs.bs_id] = None

        # Current-slot ON/OFF and tx power
        tx_power_map_now: Dict[int, float] = {}
        for bs in self.base_stations:
            sel = bs_selections[bs.bs_id]
            tx_power_map_now[bs.bs_id] = float(self.P_max[bs.bs_id]) if (sel is not None) else 0.0
                

        # Actual scheduled rates
        served_rates = {u.ue_id: 0.0 for u in self.users}
        bs_served_rate = {bs.bs_id: 0.0 for bs in self.base_stations}

        for bs in self.base_stations:
            sel = bs_selections[bs.bs_id]
            if sel is None:
                continue

            rate = self.calculate_scheduled_rate(sel, bs.bs_id, tx_power_map_now)
            served_rates[sel] = max(served_rates[sel], rate)
            bs_served_rate[bs.bs_id] = float(rate)

        total_rate = float(sum(served_rates.values()))
        power_consumed = {bs.bs_id: float(tx_power_map_now[bs.bs_id]) for bs in self.base_stations}

        served_bs_of_user = {u.ue_id: None for u in self.users}
        for bs in self.base_stations:
            sel = bs_selections[bs.bs_id]
            if sel is not None:
                served_bs_of_user[sel] = bs.bs_id
        
        handover_u = {}
        for u in self.users:
            ue_id = u.ue_id
            served_bs = served_bs_of_user[ue_id]
            handover_u[ue_id] = self._compute_handover_indicator(ue_id, served_bs)

        # Update hard window usage
        self.window_step += 1
        for bs in self.base_stations:
            if power_consumed[bs.bs_id] > 0.0:
                self.bs_on_used_in_window[bs.bs_id] += 1

        if self.window_step % self.hard_window_len == 0:
            self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}

        # ON history
        for bs in self.base_stations:
            self.bs_on_hist[bs.bs_id].append(1.0 if power_consumed[bs.bs_id] > 0.0 else 0.0)

        # Store current-slot power for next-slot decision interference
        self.prev_power = power_consumed.copy()

        old_Q_u = self.Q_u.copy()
        old_G_u = self.G_u.copy()
        old_Z_b = self.Z_b.copy()
        old_m_u = self.m_u.copy()

        # Queue updates
        for u in self.users:
            aux_rate = self.compute_aux_rate(u.ue_id)
            actual_rate = served_rates[u.ue_id]
            self.Q_u[u.ue_id] = max(1e-12, self.Q_u[u.ue_id] + (aux_rate - actual_rate))

        for bs in self.base_stations:
            power = power_consumed[bs.bs_id]
            budget = self.P_bar[bs.bs_id]
            self.Z_b[bs.bs_id] = max(0.001, self.Z_b[bs.bs_id] + (power - budget))

        for u in self.users:
            ue_id = u.ue_id
            self.G_u[ue_id] = max(0.0, self.G_u[ue_id] + handover_u[ue_id] - self.H_bar[ue_id])

        for u in self.users:
            ue_id = u.ue_id
            served_bs_id = served_bs_of_user[ue_id]
            if served_bs_id is not None:
                self.m_u[ue_id] = served_bs_id

        # DPP-derived global reward using OLD queues
        term1 = float(sum(old_Q_u[u.ue_id] * served_rates[u.ue_id] for u in self.users))
        term2 = float(sum((old_Z_b[bs.bs_id] + self.lambda_E) * power_consumed[bs.bs_id] for bs in self.base_stations))
        term3 = float(sum(old_G_u[u.ue_id] * handover_u[u.ue_id] for u in self.users))
        
        global_reward = term1 - term2 - term3
        ue_team_reward = global_reward
        bs_team_reward = global_reward

        # Per-user logging
        ue_per_user_rewards = {
            u.ue_id: float(old_Q_u[u.ue_id] * served_rates[u.ue_id] - old_G_u[u.ue_id] * handover_u[u.ue_id])
            for u in self.users
        }

        # Per-BS logging
        bs_rewards = np.array([
            float(-(old_Z_b[bs.bs_id] + self.lambda_E) * power_consumed[bs.bs_id])
            for bs in self.base_stations
        ], dtype=np.float32)

        on_feats = self._get_bs_on_features()
        rho = self.power_budget_ratio

        # Move to next state
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
            "G_u": self.G_u.copy(),
            "m_u": self.m_u.copy(),

            "handover_u": handover_u.copy(),
            "served_bs_of_user": served_bs_of_user.copy(),
            "global_reward": float(global_reward),
            "ue_per_user_rewards": ue_per_user_rewards,
            "bs_rewards": bs_rewards.copy(),

            "bs_selections": bs_selections,
            "bs_requests": {bs_id: len(reqs) for bs_id, reqs in bs_requests.items()},
            "prev_req_ratio": self.prev_req_ratio.copy(),

            "total_QR_dummy": float(sum(old_Q_u[u.ue_id] * served_rates[u.ue_id] for u in self.users)),
            "total_ZP_dummy": float(sum(old_Z_b[bs.bs_id] * power_consumed[bs.bs_id] for bs in self.base_stations)),
            "total_GH_dummy": float(sum(old_G_u[u.ue_id] * handover_u[u.ue_id] for u in self.users)),
            "total_HO_count": float(sum(handover_u[u.ue_id] for u in self.users)),

            "no_coverage_count": int(self.no_coverage_count),
            "bs_on_used_in_window": self.bs_on_used_in_window.copy(),
            "window_step": int(self.window_step),
            "on_feats": on_feats,
            "rho": float(rho),

            "hard_constraint_enabled": bool(self.use_hard_constraint),
            "hard_on_limit": self.hard_on_limit.copy(),
        }

        done = False
        return local_obs, global_obs, info, done

    # =========================================================
    # Metric
    # =========================================================
    def calculate_jain_fairness(self, rate_history: List) -> float:
        """
        Jain's fairness computed from the most recent up to 100 slots.
        """
        recent = rate_history if len(rate_history) < 100 else rate_history[-100:]
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

class HAPPOTrainer:
    def __init__(
        self,
        env,
        lr_actor_ue: float = 3e-4,
        lr_actor_bs: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef_ue: float = 0.05,
        entropy_coef_bs: float = 0.05,
        value_coef: float = 0.5,
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
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.n_epochs = int(n_epochs)
        self.minibatch_size = int(minibatch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actors
        self.ue_actor = UEActorNetwork(env.local_obs_dim, env.action_dim).to(self.device)
        self.ue_actor_optim = optim.Adam(self.ue_actor.parameters(), lr=lr_actor_ue)

        self.bs_actor = BSActorNetwork(env.bs_obs_dim, env.bs_action_dim).to(self.device)
        self.bs_actor_optim = optim.Adam(self.bs_actor.parameters(), lr=lr_actor_bs)

        # Critic
        self.critic = CentralizedCritic(env.global_obs_dim).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Value normalization
        self.vn = ValueNorm(device=self.device)

        self.reset_rollout()

        print(f"[TRAINER] UE agents(shared actor): {len(env.users)}")
        print(f"[TRAINER] BS agents(shared actor): {len(env.base_stations)} | TopK={env.bs_top_k}")
        print(f"[TRAINER] Device: {self.device}")
        print(f"[TRAINER] PPO epochs: {self.n_epochs} | minibatch_size: {self.minibatch_size}")
        print(f"[TRAINER] Shared centralized critic: scalar V(s)")
        print(f"[TRAINER] Sequential policy update: UE actor -> BS actor\n")

    # =========================================================
    # Plot helper
    # =========================================================
    def _plot_qzg_100(
        self,
        Q_mean_history,
        Z_mean_history,
        G_mean_history,
        save_path: str,
        block: int = 100,
    ):
        """
        Plot 100-step block averages of Q, Z, and G after training.

        Q_mean_history: step-wise mean Q over all UEs
        Z_mean_history: step-wise mean Z over all BSs
        G_mean_history: step-wise mean G over all UEs
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        Q_arr = np.asarray(Q_mean_history, dtype=np.float32)
        Z_arr = np.asarray(Z_mean_history, dtype=np.float32)
        G_arr = np.asarray(G_mean_history, dtype=np.float32)

        T = min(len(Q_arr), len(Z_arr), len(G_arr))
        if T == 0:
            print("[WARN] Q/Z/G histories are empty. Skip QZG plot.")
            return

        Q_arr = Q_arr[:T]
        Z_arr = Z_arr[:T]
        G_arr = G_arr[:T]

        n_blocks = T // block

        if n_blocks > 0:
            Q_plot = Q_arr[:n_blocks * block].reshape(n_blocks, block).mean(axis=1)
            Z_plot = Z_arr[:n_blocks * block].reshape(n_blocks, block).mean(axis=1)
            G_plot = G_arr[:n_blocks * block].reshape(n_blocks, block).mean(axis=1)
            x = np.arange(1, n_blocks + 1) * block
        else:
            Q_plot = np.asarray([Q_arr.mean()], dtype=np.float32)
            Z_plot = np.asarray([Z_arr.mean()], dtype=np.float32)
            G_plot = np.asarray([G_arr.mean()], dtype=np.float32)
            x = np.asarray([T], dtype=np.int32)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(x, Q_plot, label="Q mean")
        ax.plot(x, Z_plot, label="Z mean")
        ax.plot(x, G_plot, label="G mean")

        ax.set_xlabel("Training step")
        ax.set_ylabel("100-step block average")
        ax.set_title("Training Queue Dynamics: Q, Z, G")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        print(f"✅ Saved Q/Z/G 100-step plot: {save_path}")

    def _default_qzg_plot_path(self, save_npz_path: Optional[str]) -> str:
        """
        If save_npz_path is provided:
            results/foo.npz -> results/foo_qzg_100.png
        Otherwise:
            train_qzg_100.png
        """
        if save_npz_path is None:
            return "train_qzg_100.png"

        root, _ = os.path.splitext(save_npz_path)
        return f"{root}_qzg_100.png"

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
            "critic": self.critic.state_dict(),
            "vn": self.vn.state_dict(),
        }

        if save_optim:
            payload.update({
                "ue_actor_optim": self.ue_actor_optim.state_dict(),
                "bs_actor_optim": self.bs_actor_optim.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
            })

        torch.save(payload, path)
        print(f"✅ Model saved: {path}")

    def load_model(self, path: str, load_optim: bool = False, map_location: Optional[str] = None):
        map_location = map_location if map_location is not None else str(self.device)
        payload = torch.load(path, map_location=map_location)

        self.ue_actor.load_state_dict(payload["ue_actor"])
        self.bs_actor.load_state_dict(payload["bs_actor"])
        self.critic.load_state_dict(payload["critic"])
        self.vn.load_state_dict(payload["vn"])

        if load_optim and ("ue_actor_optim" in payload):
            self.ue_actor_optim.load_state_dict(payload["ue_actor_optim"])
            self.bs_actor_optim.load_state_dict(payload["bs_actor_optim"])
            self.critic_opt.load_state_dict(payload["critic_opt"])

        self.ue_actor.eval()
        self.bs_actor.eval()
        self.critic.eval()

        print(f"✅ Model loaded: {path} (optim={load_optim})")

    def load_actor_only(self, path: str, map_location: Optional[str] = None):
        """
        Load only the shared UE/BS actors from a trained model.

        This is intentionally used for UE-count scalability tests. The actors
        only depend on BS-related local dimensions when the BS count is fixed,
        while the centralized critic depends on the number of UEs through the
        global observation dimension. Therefore, for U != train_U, loading the
        critic would fail or be meaningless.
        """
        map_location = map_location if map_location is not None else str(self.device)
        payload = torch.load(path, map_location=map_location)
        meta = payload.get("meta", {})

        # Actor dimensions must match. With a fixed BS count, these remain identical
        # even if the number of UEs changes.
        expected_pairs = [
            ("local_obs_dim", self.env.local_obs_dim),
            ("action_dim", self.env.action_dim),
            ("bs_obs_dim", self.env.bs_obs_dim),
            ("bs_action_dim", self.env.bs_action_dim),
        ]
        for key, current_value in expected_pairs:
            saved_value = meta.get(key, current_value)
            if int(saved_value) != int(current_value):
                raise ValueError(
                    f"Actor dimension mismatch for {key}: "
                    f"saved={saved_value}, current={current_value}. "
                    f"This usually means the BS count changed. For different BS counts, "
                    f"the code needs max-BS padding/masking architecture."
                )

        self.ue_actor.load_state_dict(payload["ue_actor"])
        self.bs_actor.load_state_dict(payload["bs_actor"])

        self.ue_actor.eval()
        self.bs_actor.eval()
        self.critic.eval()

        saved_users = meta.get("n_users", "unknown")
        saved_bs = meta.get("n_bs", "unknown")
        print(
            f"✅ Actor-only model loaded: {path} | "
            f"saved(B={saved_bs}, U={saved_users}) -> current(B={self.env.n_bs}, U={self.env.n_agents})"
        )

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

            "reward": [],
            "v_n": [],
            "nv_n": [],

            "dones": [],
        }

    @torch.no_grad()
    def select_actions(
        self,
        local_obs: Dict[int, np.ndarray],
        global_obs: np.ndarray,
        use_critic: bool = True,
    ):
        """
        Select UE and BS actions.

        use_critic=True is needed during training because PPO/GAE needs V(s).
        use_critic=False is used during scalability evaluation with a different
        number of UEs, because the centralized critic input dimension depends
        on the number of UEs. The shared UE/BS actors are still reusable when
        the BS count is fixed.
        """
        users = self.env.users

        if use_critic:
            global_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            v_n = self.critic(global_t).squeeze(0)
        else:
            v_n = torch.tensor(0.0, dtype=torch.float32, device=self.device)

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

            float(v_n.item()),
        )

    def store_step(
        self,
        local_obs, global_obs,
        ue_actions_dict, ue_logp_np, ue_masks_np,
        bs_actions_dict, bs_logp_np, bs_obs_np, bs_masks_np, cand_lists,
        reward: float,
        v_n: float, nv_n: float,
        done: bool
    ):
        users = self.env.users
        bss = self.env.base_stations

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

        self.rb["reward"].append(float(reward))
        self.rb["v_n"].append(float(v_n))
        self.rb["nv_n"].append(float(nv_n))

        self.rb["dones"].append(bool(done))

    def _iter_minibatches(self, N: int, batch_size: int):
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            yield idx[start:start + batch_size]

    # =========================================================
    # GAE
    # =========================================================
    def compute_gae(self, rewards, values_n, next_values_n, dones):
        T = len(rewards)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        v_n = torch.tensor(values_n, dtype=torch.float32, device=self.device)
        nv_n = torch.tensor(next_values_n, dtype=torch.float32, device=self.device)

        v = self.vn.denormalize(v_n)
        nv = self.vn.denormalize(nv_n)

        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            done_mask = 1.0 - float(dones[t])
            delta = r_t[t] + self.gamma * nv[t] * done_mask - v[t]
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

        global_obs = torch.tensor(
            np.stack(self.rb["global_obs"], axis=0),
            dtype=torch.float32,
            device=self.device
        )
        dones = self.rb["dones"]

        # GAE
        adv, ret_raw = self.compute_gae(
            rewards=self.rb["reward"],
            values_n=self.rb["v_n"],
            next_values_n=self.rb["nv_n"],
            dones=dones
        )

        with torch.no_grad():
            self.vn.update(ret_raw)

        ret_n = self.vn.normalize(ret_raw).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.detach()

        # UE tensors
        ue_local_obs = torch.tensor(
            np.stack(self.rb["local_obs"], axis=0),
            dtype=torch.float32,
            device=self.device
        )
        ue_masks = torch.tensor(
            np.stack(self.rb["ue_masks"], axis=0),
            dtype=torch.bool,
            device=self.device
        )
        ue_actions = torch.tensor(
            np.stack(self.rb["ue_actions"], axis=0),
            dtype=torch.long,
            device=self.device
        )
        ue_old_logp = torch.tensor(
            np.stack(self.rb["ue_logp"], axis=0),
            dtype=torch.float32,
            device=self.device
        )

        ue_local_f = ue_local_obs.reshape(T * N, -1)
        ue_masks_f = ue_masks.reshape(T * N, -1)
        ue_actions_f = ue_actions.reshape(T * N)
        ue_old_logp_f = ue_old_logp.reshape(T * N)
        ue_adv_f = adv.repeat_interleave(N)

        # BS tensors
        bs_obs = torch.tensor(
            np.stack(self.rb["bs_obs"], axis=0),
            dtype=torch.float32,
            device=self.device
        )
        bs_masks = torch.tensor(
            np.stack(self.rb["bs_masks"], axis=0),
            dtype=torch.bool,
            device=self.device
        )
        bs_actions = torch.tensor(
            np.stack(self.rb["bs_actions"], axis=0),
            dtype=torch.long,
            device=self.device
        )
        bs_old_logp = torch.tensor(
            np.stack(self.rb["bs_logp"], axis=0),
            dtype=torch.float32,
            device=self.device
        )

        bs_obs_f = bs_obs.reshape(T * B, -1)
        bs_masks_f = bs_masks.reshape(T * B, -1)
        bs_actions_f = bs_actions.reshape(T * B)
        bs_old_logp_f = bs_old_logp.reshape(T * B)
        bs_adv_f = adv.repeat_interleave(B)

        losses = {
            "critic": 0.0,
            "actor_ue": 0.0,
            "actor_bs": 0.0,
            "entropy_ue": 0.0,
            "entropy_bs": 0.0,
        }

        for _ in range(self.n_epochs):
            # Critic
            c_epoch, c_cnt = 0.0, 0
            critic_mb = max(32, min(self.minibatch_size, T))

            for mb in self._iter_minibatches(T, critic_mb):
                v_pred_n = self.critic(global_obs[mb])
                loss_v = F.mse_loss(v_pred_n, ret_n[mb])

                self.critic_opt.zero_grad()
                (self.value_coef * loss_v).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                c_epoch += float(loss_v.item())
                c_cnt += 1

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
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * ue_adv_f[mb]

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
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * bs_adv_f[mb]

                loss_pi = -torch.min(surr1, surr2).mean()
                loss_ent = -entropy.mean()

                self.bs_actor_optim.zero_grad()
                (loss_pi + self.entropy_coef_bs * loss_ent).backward()
                nn.utils.clip_grad_norm_(self.bs_actor.parameters(), self.max_grad_norm)
                self.bs_actor_optim.step()

                bs_epoch += float(loss_pi.item())
                bs_ent_epoch += float(loss_ent.item())
                bs_cnt += 1

            losses["critic"] += c_epoch / max(1, c_cnt)
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
        plot_qzg_path: Optional[str] = None,
        episode_len: Optional[int] = None,
        num_episodes: Optional[int] = None,
    ):
        if episode_len is not None:
            episode_len = int(episode_len)
            assert episode_len >= 1

            if num_episodes is None:
                assert int(n_steps) % episode_len == 0, (
                    "When episode_len is set and num_episodes is None, "
                    "n_steps must be divisible by episode_len."
                )
                num_episodes = int(n_steps) // episode_len
            else:
                num_episodes = int(num_episodes)
                n_steps = int(num_episodes * episode_len)
        else:
            n_steps = int(n_steps)
            num_episodes = 1

        print(f"\n{'='*100}")
        print(" HAPPO Training")
        print(f"{'='*100}")
        print(f"Total train steps: {n_steps}")
        if episode_len is not None:
            print(f"Episodic training: {num_episodes} episodes x {episode_len} steps/episode")
            print("At every episode boundary, env.reset() resets Q_u, Z_b, G_u, m_u, window counters, and channel/mobility state.")
        print(f"Update interval: {update_interval}")
        print(f"Hard constraint during training: {self.env.use_hard_constraint}")
        print(f"{'='*100}\n")

        throughput_history = []
        fairness_history = []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        slot_rates = []

        # Queue histories
        queue_history = {
            "Q_u": defaultdict(list),
            "Z_b": defaultdict(list),
            "G_u": defaultdict(list),
        }

        # Mean queue histories
        Q_mean_history = []
        Z_mean_history = []
        G_mean_history = []

        # Handover histories
        handover_count_history = []
        handover_ratio_history = []

        global_reward_hist = []
        ue_per_user_reward_hist = []

        # Loss histories, recorded per PPO update
        update_step_history = []
        critic_loss_history = []
        actor_ue_loss_history = []
        actor_bs_loss_history = []
        entropy_ue_history = []
        entropy_bs_history = []

        # Episode logging. Episode IDs start from 1.
        episode_id_history = []
        episode_step_history = []
        episode_end_steps = []

        local_obs, global_obs = self.env.reset()

        if episode_len is not None:
            print(f"[EPISODE START] Train episode 1/{num_episodes} | global_step=0")

        for step in range(n_steps):
            current_episode = (step // episode_len) + 1 if episode_len is not None else 1
            current_episode_step = (step % episode_len) + 1 if episode_len is not None else step + 1
            episode_done = bool(episode_len is not None and current_episode_step == episode_len)
            (
                ue_actions,
                ue_logp_np,
                ue_ent_np,
                ue_masks_np,
                bs_actions,
                bs_logp_np,
                bs_ent_np,
                bs_obs_np,
                bs_masks_np,
                cand_lists,
                v_n,
            ) = self.select_actions(local_obs, global_obs)

            next_local_obs, next_global_obs, info, env_done = self.env.step_joint(
                ue_actions=ue_actions,
                bs_actions=bs_actions,
                cand_lists=cand_lists
            )

            # For episodic training, force a terminal transition every episode_len
            # steps so GAE does not bootstrap across episode boundaries.
            done = bool(env_done or episode_done)

            with torch.no_grad():
                if done:
                    nv_n = np.asarray(0.0, dtype=np.float32)
                else:
                    next_global_t = torch.as_tensor(
                        next_global_obs,
                        dtype=torch.float32,
                        device=self.device
                    ).unsqueeze(0)

                    nv_n = self.critic(next_global_t).squeeze(0).detach().cpu().numpy().astype(np.float32)

            reward = float(info["global_reward"])

            self.store_step(
                local_obs=local_obs,
                global_obs=global_obs,
                ue_actions_dict=ue_actions,
                ue_logp_np=ue_logp_np,
                ue_masks_np=ue_masks_np,
                bs_actions_dict=bs_actions,
                bs_logp_np=bs_logp_np,
                bs_obs_np=bs_obs_np,
                bs_masks_np=bs_masks_np,
                cand_lists=cand_lists,
                reward=reward,
                v_n=float(v_n),
                nv_n=float(nv_n),
                done=done
            )

            # --------------------------------------------------
            # Main performance histories
            # --------------------------------------------------
            throughput_history.append(info["total_throughput"])
            episode_id_history.append(int(current_episode))
            episode_step_history.append(int(current_episode_step))

            rates_this_slot = [info["served_rates"][u.ue_id] for u in self.env.users]
            slot_rates.append(rates_this_slot)

            fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

            for bs_id, power in info["power_consumed"].items():
                power_history[bs_id].append(power)

            # --------------------------------------------------
            # Queue histories
            # --------------------------------------------------
            for ue_id, q_val in info["Q_u"].items():
                queue_history["Q_u"][ue_id].append(q_val)

            for bs_id, zb_val in info["Z_b"].items():
                queue_history["Z_b"][bs_id].append(zb_val)

            for ue_id, g_val in info["G_u"].items():
                queue_history["G_u"][ue_id].append(g_val)

            Q_mean_history.append(float(np.mean(list(info["Q_u"].values()))))
            Z_mean_history.append(float(np.mean(list(info["Z_b"].values()))))
            G_mean_history.append(float(np.mean(list(info["G_u"].values()))))

            # --------------------------------------------------
            # Handover histories
            # --------------------------------------------------
            ho_count = float(info["total_HO_count"])
            ho_ratio = ho_count / max(1, self.env.n_agents)

            handover_count_history.append(ho_count)
            handover_ratio_history.append(ho_ratio)

            # --------------------------------------------------
            # Reward histories
            # --------------------------------------------------
            global_reward_hist.append(reward)
            ue_per_user_reward_hist.append(
                [float(info["ue_per_user_rewards"][u.ue_id]) for u in self.env.users]
            )

            # Do not advance/reset observations here. For episodic training,
            # observation advancement is handled after PPO update/logging so
            # episode boundaries can call env.reset() cleanly.

            # --------------------------------------------------
            # PPO update
            # --------------------------------------------------
            should_update = ((step + 1) % update_interval == 0) or episode_done or bool(env_done)
            if should_update:
                losses = self.update()

                if losses:
                    update_step_history.append(step + 1)
                    critic_loss_history.append(float(losses["critic"]))
                    actor_ue_loss_history.append(float(losses["actor_ue"]))
                    actor_bs_loss_history.append(float(losses["actor_bs"]))
                    entropy_ue_history.append(float(losses["entropy_ue"]))
                    entropy_bs_history.append(float(losses["entropy_bs"]))

                    print(
                        f"[UPDATE] Step {step+1} | "
                        f"UE_Actor:{losses['actor_ue']:.4f} | "
                        f"BS_Actor:{losses['actor_bs']:.4f} | "
                        f"Critic:{losses['critic']:.4f} | "
                        f"Ent(UE):{losses['entropy_ue']:.4f} | "
                        f"Ent(BS):{losses['entropy_bs']:.4f}"
                    )

            # --------------------------------------------------
            # Training log every 100 steps
            # --------------------------------------------------
            if (step + 1) % 100 == 0:
                recent_thr = float(np.mean(throughput_history[-100:]))
                recent_fair = float(fairness_history[-1])
                global_rew_100 = float(np.mean(global_reward_hist[-100:]))

                Q_mean_100 = float(np.mean(Q_mean_history[-100:])) if len(Q_mean_history) > 0 else 0.0
                Z_mean_100 = float(np.mean(Z_mean_history[-100:])) if len(Z_mean_history) > 0 else 0.0
                G_mean_100 = float(np.mean(G_mean_history[-100:])) if len(G_mean_history) > 0 else 0.0

                ho_count_100 = float(np.mean(handover_count_history[-100:])) if len(handover_count_history) > 0 else 0.0
                ho_ratio_100 = float(np.mean(handover_ratio_history[-100:])) if len(handover_ratio_history) > 0 else 0.0

                on_parts = []
                for bs in self.env.base_stations:
                    hist = list(self.env.bs_on_hist[bs.bs_id])
                    on_ratio_100 = float(np.mean(hist[-100:])) if len(hist) > 0 else 0.0
                    on_parts.append(f"BS{bs.bs_id}:{on_ratio_100:.3f}")
                on_str = " ".join(on_parts)

                print(
                    f"Step {step+1:5d} | "
                    f"Ep:{current_episode}/{num_episodes} EpStep:{current_episode_step:5d} | "
                    f"Thr:{recent_thr:.3f} | "
                    f"Fair:{recent_fair:.3f} | "
                    f"ON(100): {on_str} | "
                    f"Q/Z/G(100): {Q_mean_100:.2f}/{Z_mean_100:.2f}/{G_mean_100:.2f} | "
                    f"HO(100): count={ho_count_100:.2f}, ratio={ho_ratio_100:.4f} | "
                    f"GlobalRew(100):{global_rew_100:.3f}"
                )

            # --------------------------------------------------
            # Episode boundary reset
            # --------------------------------------------------
            if episode_done:
                episode_end_steps.append(step + 1)

                ep_thr = float(np.mean(throughput_history[-episode_len:])) if episode_len is not None else float(np.mean(throughput_history))
                ep_rew = float(np.mean(global_reward_hist[-episode_len:])) if episode_len is not None else float(np.mean(global_reward_hist))

                print(
                    f"[EPISODE END] Train episode {current_episode}/{num_episodes} | "
                    f"global_step={step+1} | mean_thr={ep_thr:.4f} | mean_global_reward={ep_rew:.4f}"
                )

                if (step + 1) < n_steps:
                    print("[EPISODE RESET] Resetting Q_u/Z_b/G_u/m_u/window counters before next training episode.")
                    local_obs, global_obs = self.env.reset()
                    print(f"[EPISODE START] Train episode {current_episode + 1}/{num_episodes} | global_step={step+1}")
                else:
                    local_obs, global_obs = next_local_obs, next_global_obs
            else:
                local_obs, global_obs = next_local_obs, next_global_obs

        results = {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "queue_history": queue_history,

            "global_reward": global_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,

            # Handover metrics
            "handover_count_history": handover_count_history,
            "handover_ratio_history": handover_ratio_history,

            # Queue mean histories
            "Q_mean_history": Q_mean_history,
            "Z_mean_history": Z_mean_history,
            "G_mean_history": G_mean_history,

            # Episode metadata
            "episode_id_history": episode_id_history,
            "episode_step_history": episode_step_history,
            "episode_end_steps": episode_end_steps,
            "episode_len": episode_len,
            "num_episodes": num_episodes,

            # Loss curves
            "update_step_history": update_step_history,
            "critic_loss_history": critic_loss_history,
            "actor_ue_loss_history": actor_ue_loss_history,
            "actor_bs_loss_history": actor_bs_loss_history,
            "entropy_ue_history": entropy_ue_history,
            "entropy_bs_history": entropy_bs_history,
        }

        # --------------------------------------------------
        # Optional Q/Z/G plot after training
        # --------------------------------------------------
        # In the plot-NPZ experiment main below, plot_qzg_path is left as None
        # so only the requested compact NPZ files are saved.
        if plot_qzg_path is not None:
            self._plot_qzg_100(
                Q_mean_history=Q_mean_history,
                Z_mean_history=Z_mean_history,
                G_mean_history=G_mean_history,
                save_path=plot_qzg_path,
                block=100,
            )

        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="train")

        return results

    @torch.no_grad()
    def evaluate(self, n_steps: int, save_npz_path: Optional[str] = None):
        print(f"\n{'='*84}")
        print(" EVALUATION (No Learning)")
        print(f"{'='*84}")
        print(f"Total eval steps: {n_steps}")
        print(f"Evaluation episode: 1 episode x {n_steps} steps")
        print("Action selection: actor-only evaluation; centralized critic is not used.")
        print(f"Hard constraint during evaluation: {self.env.use_hard_constraint}\n")

        self.ue_actor.eval()
        self.bs_actor.eval()
        self.critic.eval()

        throughput_history = []
        fairness_history = []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        slot_rates = []
        global_reward_hist = []
        ue_per_user_reward_hist = []

        eval_on100_hist = {bs.bs_id: [] for bs in self.env.base_stations}

        # Handover metric histories
        handover_count_history = []
        handover_ratio_history = []

        # Optional QoE metric histories
        served_ratio_history = []
        outage_ratio_history = []

        # Queue mean histories
        Q_mean_history = []
        Z_mean_history = []
        G_mean_history = []

        local_obs, global_obs = self.env.reset()

        for step in range(n_steps):
            (
                ue_actions,
                ue_logp_np,
                ue_ent_np,
                ue_masks_np,
                bs_actions,
                bs_logp_np,
                bs_ent_np,
                bs_obs_np,
                bs_masks_np,
                cand_lists,
                v_n,
            ) = self.select_actions(local_obs, global_obs, use_critic=False)

            next_local_obs, next_global_obs, info, done = self.env.step_joint(
                ue_actions=ue_actions,
                bs_actions=bs_actions,
                cand_lists=cand_lists
            )

            throughput_history.append(info["total_throughput"])

            rates_this_slot = [info["served_rates"][u.ue_id] for u in self.env.users]
            slot_rates.append(rates_this_slot)

            fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

            # --------------------------------------------------
            # Handover metric
            # --------------------------------------------------
            ho_count = float(info["total_HO_count"])
            handover_count_history.append(ho_count)
            handover_ratio_history.append(ho_count / max(1, self.env.n_agents))

            # --------------------------------------------------
            # Optional QoE / outage metric
            # served user = rate > 0
            # --------------------------------------------------
            served_flags = np.array(
                [1.0 if info["served_rates"][u.ue_id] > 0.0 else 0.0 for u in self.env.users],
                dtype=np.float32
            )
            served_ratio = float(np.mean(served_flags))
            served_ratio_history.append(served_ratio)
            outage_ratio_history.append(1.0 - served_ratio)

            # --------------------------------------------------
            # Queue mean histories
            # --------------------------------------------------
            Q_mean_history.append(float(np.mean(list(info["Q_u"].values()))))
            Z_mean_history.append(float(np.mean(list(info["Z_b"].values()))))
            G_mean_history.append(float(np.mean(list(info["G_u"].values()))))

            for bs_id, power in info["power_consumed"].items():
                power_history[bs_id].append(power)

            global_reward_hist.append(float(info["global_reward"]))
            ue_per_user_reward_hist.append(
                [float(info["ue_per_user_rewards"][u.ue_id]) for u in self.env.users]
            )

            local_obs, global_obs = next_local_obs, next_global_obs

            if (step + 1) % 100 == 0:
                recent_thr = float(np.mean(throughput_history[-100:]))
                recent_fair = float(fairness_history[-1])

                G_mean_100 = float(np.mean(G_mean_history[-100:])) if len(G_mean_history) > 0 else 0.0
                ho_count_100 = float(np.mean(handover_count_history[-100:])) if len(handover_count_history) > 0 else 0.0
                ho_ratio_100 = float(np.mean(handover_ratio_history[-100:])) if len(handover_ratio_history) > 0 else 0.0

                on_parts = []
                for bs in self.env.base_stations:
                    hist = list(self.env.bs_on_hist[bs.bs_id])
                    on_ratio_100 = float(np.mean(hist[-100:])) if len(hist) > 0 else 0.0
                    eval_on100_hist[bs.bs_id].append(on_ratio_100)
                    on_parts.append(f"BS{bs.bs_id}:{on_ratio_100:.3f}")
                on_str = " ".join(on_parts)

                print(
                    f"[EVAL] Step {step+1:5d} | "
                    f"Thr:{recent_thr:.3f} | "
                    f"Fair:{recent_fair:.3f} | "
                    f"ON(100): {on_str} | "
                    f"G(100):{G_mean_100:.3f} | "
                    f"HO(100): count={ho_count_100:.2f}, ratio={ho_ratio_100:.4f}"
                )

            if (step + 1) % 1000 == 0:
                recent_rates_1000 = np.asarray(slot_rates[-1000:], dtype=np.float32)
                fair_1000 = _jain_fairness_zero_ok(recent_rates_1000)
                print(
                    f"[EVAL-1K] Step {step+1:5d} | "
                    f"Fairness(1000):{fair_1000:.4f}"
                )

            if (step + 1) % 10000 == 0:
                thr_10k_mean = float(np.mean(throughput_history[-10000:]))
                fair_10k_mean = float(np.mean(fairness_history[-10000:]))

                G_mean_10k = float(np.mean(G_mean_history[-10000:])) if len(G_mean_history) > 0 else 0.0
                ho_count_10k = float(np.mean(handover_count_history[-10000:])) if len(handover_count_history) > 0 else 0.0
                ho_ratio_10k = float(np.mean(handover_ratio_history[-10000:])) if len(handover_ratio_history) > 0 else 0.0

                on10k_parts = []
                for bs in self.env.base_stations:
                    recent_power = power_history[bs.bs_id][-10000:]
                    on10k_mean = float(np.mean(np.asarray(recent_power) > 0.0)) if len(recent_power) > 0 else 0.0
                    on10k_parts.append(f"BS{bs.bs_id}:{on10k_mean:.3f}")
                on10k_str = " ".join(on10k_parts)

                print(
                    f"[EVAL-10K] Step {step+1:5d} | "
                    f"ThroughputMean(10k):{thr_10k_mean:.3f} | "
                    f"Mean(step-wise Fair(100) over 10k):{fair_10k_mean:.3f} | "
                    f"ON(10k): {on10k_str} | "
                    f"GMean(10k):{G_mean_10k:.3f} | "
                    f"HO(10k): count={ho_count_10k:.2f}, ratio={ho_ratio_10k:.4f}"
                )

        results = {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "global_reward": global_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,

            "handover_count_history": handover_count_history,
            "handover_ratio_history": handover_ratio_history,

            "served_ratio_history": served_ratio_history,
            "outage_ratio_history": outage_ratio_history,

            "Q_mean_history": Q_mean_history,
            "Z_mean_history": Z_mean_history,
            "G_mean_history": G_mean_history,
        }

        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="eval")

        return results

    # =========================================================
    # NPZ save
    # =========================================================
    def save_results_npz(self, results: Dict, npz_path: str, tag: str = "run"):
        os.makedirs(os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".", exist_ok=True)

        thr = np.asarray(results.get("throughput_history", []), dtype=np.float32)
        fair = np.asarray(results.get("fairness_history", []), dtype=np.float32)

        global_reward = np.asarray(results.get("global_reward", []), dtype=np.float32)
        ue_per_user = np.asarray(results.get("ue_per_user_reward", []), dtype=np.float32)

        handover_count = np.asarray(results.get("handover_count_history", []), dtype=np.float32)
        handover_ratio = np.asarray(results.get("handover_ratio_history", []), dtype=np.float32)

        served_ratio = np.asarray(results.get("served_ratio_history", []), dtype=np.float32)
        outage_ratio = np.asarray(results.get("outage_ratio_history", []), dtype=np.float32)

        Q_mean = np.asarray(results.get("Q_mean_history", []), dtype=np.float32)
        Z_mean = np.asarray(results.get("Z_mean_history", []), dtype=np.float32)
        G_mean = np.asarray(results.get("G_mean_history", []), dtype=np.float32)

        episode_id = np.asarray(results.get("episode_id_history", []), dtype=np.int32)
        episode_step = np.asarray(results.get("episode_step_history", []), dtype=np.int32)
        episode_end_steps = np.asarray(results.get("episode_end_steps", []), dtype=np.int32)
        episode_len_saved = int(results.get("episode_len", 0) or 0)
        num_episodes_saved = int(results.get("num_episodes", 0) or 0)

        update_steps = np.asarray(results.get("update_step_history", []), dtype=np.int32)
        critic_loss = np.asarray(results.get("critic_loss_history", []), dtype=np.float32)
        actor_ue_loss = np.asarray(results.get("actor_ue_loss_history", []), dtype=np.float32)
        actor_bs_loss = np.asarray(results.get("actor_bs_loss_history", []), dtype=np.float32)
        entropy_ue = np.asarray(results.get("entropy_ue_history", []), dtype=np.float32)
        entropy_bs = np.asarray(results.get("entropy_bs_history", []), dtype=np.float32)

        if ue_per_user.ndim == 2 and ue_per_user.shape[0] > 0:
            mean_user_reward_step = ue_per_user.mean(axis=1).astype(np.float32)
            mean_user_reward_ma100 = moving_avg(mean_user_reward_step, 100)
        else:
            mean_user_reward_step = np.asarray([], dtype=np.float32)
            mean_user_reward_ma100 = np.asarray([], dtype=np.float32)

        # Block average for reward plotting
        block = 500

        reward_x_500, global_reward_500 = (
            block_avg_1d(global_reward, block)
            if global_reward.size > 0 else
            (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32))
        )

        user_reward_x_500, user_mean_reward_500 = (
            block_avg_1d(mean_user_reward_step, block)
            if mean_user_reward_step.size > 0 else
            (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32))
        )

        power_hist = results.get("power_history", {})
        bs_ids_sorted = sorted(list(power_hist.keys())) if isinstance(power_hist, dict) else []

        power_mat = []
        for bs_id in bs_ids_sorted:
            power_mat.append(np.asarray(power_hist[bs_id], dtype=np.float32))

        power_mat = (
            np.stack(power_mat, axis=0)
            if len(power_mat) > 0 else
            np.zeros((0, len(thr)), dtype=np.float32)
        )

        # --------------------------------------------------
        # Energy constraint metrics
        # --------------------------------------------------
        # power_mat shape: [n_bs, T]
        # P_max is used when BS is ON, 0 when OFF
        if power_mat.size > 0:
            bs_on_mat = (power_mat > 0.0).astype(np.float32)
            bs_on_ratio_per_bs = bs_on_mat.mean(axis=1)
            bs_on_ratio_mean = np.asarray([bs_on_ratio_per_bs.mean()], dtype=np.float32)

            energy_budget_ratio = np.asarray([self.env.power_budget_ratio], dtype=np.float32)

            energy_violation_per_bs = np.maximum(
                0.0,
                bs_on_ratio_per_bs - self.env.power_budget_ratio
            ).astype(np.float32)

            energy_violation_mean = np.asarray(
                [energy_violation_per_bs.mean()],
                dtype=np.float32
            )

            energy_violation_ratio = np.asarray(
                [float(np.mean(bs_on_ratio_per_bs > self.env.power_budget_ratio))],
                dtype=np.float32
            )
        else:
            bs_on_ratio_per_bs = np.asarray([], dtype=np.float32)
            bs_on_ratio_mean = np.asarray([], dtype=np.float32)
            energy_budget_ratio = np.asarray([self.env.power_budget_ratio], dtype=np.float32)
            energy_violation_per_bs = np.asarray([], dtype=np.float32)
            energy_violation_mean = np.asarray([], dtype=np.float32)
            energy_violation_ratio = np.asarray([], dtype=np.float32)

        # --------------------------------------------------
        # Handover constraint metrics
        # --------------------------------------------------
        # handover_ratio is slot-wise: total HO per slot / num_users
        if handover_ratio.size > 0:
            handover_ratio_mean = np.asarray(
                [float(np.mean(handover_ratio))],
                dtype=np.float32
            )

            handover_budget_ratio = np.asarray(
                [float(self.env.kappa)],
                dtype=np.float32
            )

            handover_violation_mean = np.asarray(
                [float(max(0.0, np.mean(handover_ratio) - self.env.kappa))],
                dtype=np.float32
            )

            handover_violation_flag = np.asarray(
                [float(np.mean(handover_ratio) > self.env.kappa)],
                dtype=np.float32
            )
        else:
            handover_ratio_mean = np.asarray([], dtype=np.float32)
            handover_budget_ratio = np.asarray([float(self.env.kappa)], dtype=np.float32)
            handover_violation_mean = np.asarray([], dtype=np.float32)
            handover_violation_flag = np.asarray([], dtype=np.float32)

        np.savez_compressed(
            npz_path,
            tag=str(tag),
            n_users=int(self.env.n_agents),
            n_bs=int(self.env.n_bs),
            episode_len=int(episode_len_saved),
            num_episodes=int(num_episodes_saved),
            episode_id=episode_id,
            episode_step=episode_step,
            episode_end_steps=episode_end_steps,

            throughput=thr,
            fairness=fair,

            global_reward=global_reward,
            global_reward_step=global_reward,
            ue_per_user_reward=ue_per_user,
            mean_user_reward_step=mean_user_reward_step,
            mean_user_reward_ma100=mean_user_reward_ma100,

            reward_x_500=reward_x_500,
            global_reward_500=global_reward_500,
            user_reward_x_500=user_reward_x_500,
            user_mean_reward_500=user_mean_reward_500,

            bs_ids=np.asarray(bs_ids_sorted, dtype=np.int32),
            power_mat=power_mat,

            # Performance
            handover_count=handover_count,
            handover_ratio=handover_ratio,
            handover_ratio_mean=handover_ratio_mean,

            bs_on_ratio_per_bs=bs_on_ratio_per_bs,
            bs_on_ratio_mean=bs_on_ratio_mean,

            # Constraint
            energy_budget_ratio=energy_budget_ratio,
            energy_violation_per_bs=energy_violation_per_bs,
            energy_violation_mean=energy_violation_mean,
            energy_violation_ratio=energy_violation_ratio,

            handover_budget_ratio=handover_budget_ratio,
            handover_violation_mean=handover_violation_mean,
            handover_violation_flag=handover_violation_flag,

            # Optional QoE
            served_ratio=served_ratio,
            outage_ratio=outage_ratio,

            # Queue
            Q_mean=Q_mean,
            Z_mean=Z_mean,
            G_mean=G_mean,

            # Losses
            update_steps=update_steps,
            critic_loss=critic_loss,
            actor_ue_loss=actor_ue_loss,
            actor_bs_loss=actor_bs_loss,
            entropy_ue=entropy_ue,
            entropy_bs=entropy_bs,
        )

        print(f"✅ Saved results npz: {npz_path}")


# =========================================================
# Experiment config: UE=20 fixed, independent training per kappa,
# followed by 10 independent 10,000-step evaluation episodes.
# All kappa values share the exact same 10 evaluation seeds.
# =========================================================
TRAIN_NUM_USERS = 20
EVAL_NUM_USERS = 20

# Train one independent model for every kappa value.
KAPPA_VALUES = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18]

EPISODE_LEN = 10000
TRAIN_EPISODES = 10
TRAIN_STEPS = TRAIN_EPISODES * EPISODE_LEN

# Evaluation is repeated with 10 distinct seeds. The seed list is generated
# only once per program execution and then shared by every kappa value.
EVAL_RUNS = 10
EVAL_STEPS_PER_RUN = EPISODE_LEN
EVAL_TOTAL_STEPS = EVAL_RUNS * EVAL_STEPS_PER_RUN

HARD_WINDOW_LEN = EPISODE_LEN
POWER_BUDGET_RATIO = 0.6
LAMBDA_E = 0.0

# BS=9 deployment config
N_BS = 9
AREA_SIZE = 100
COVERAGE_RADIUS = 35
GRID_MARGIN = 15.0
GRID_POINTS_PER_AXIS = 3

# Training remains exactly as before: every kappa is trained from scratch
# using the same training seed, making the kappa comparison controlled.
BASE_SEED = 3
TRAIN_SEED = BASE_SEED + TRAIN_NUM_USERS

# The 10 evaluation seeds are generated from OS entropy once at program start.
# They are distinct from one another and the identical list is reused for every
# kappa, which gives a paired and controlled kappa comparison. The generated
# list is saved so every evaluation can be reproduced later.
EVAL_SEED_MIN = 1
EVAL_SEED_MAX = 2_147_483_646

# A timestamped run directory is created at every execution, so previous
# experiment files are never overwritten.
RESULT_ROOT = "results_compare_bs9_kappa_sweep_10eval"


def generate_random_eval_seeds(num_runs: int) -> List[int]:
    """
    Generate one unique, non-deterministic evaluation-seed list using OS entropy.

    Call this function exactly once before the kappa loop. The returned list is
    then reused for every kappa value, so evaluation run i always uses the same
    environment/policy random seed across all kappa models. The seeds are saved
    with the results so each episode can be reproduced later.
    """
    num_runs = int(num_runs)
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")

    population_size = EVAL_SEED_MAX - EVAL_SEED_MIN + 1
    if num_runs > population_size:
        raise ValueError("Requested more unique seeds than the available range")

    rng = secrets.SystemRandom()
    return rng.sample(range(EVAL_SEED_MIN, EVAL_SEED_MAX + 1), num_runs)


def generate_grid_coverage_9(
    area_size: float = AREA_SIZE,
    margin: float = GRID_MARGIN,
    points_per_axis: int = GRID_POINTS_PER_AXIS,
) -> list:
    """
    Generate 9 small-cell BS positions arranged as a 3x3 grid.

    For area_size=100 and margin=15:
      SBS1: (15.00, 15.00)   SBS2: (50.00, 15.00)   SBS3: (85.00, 15.00)
      SBS4: (15.00, 50.00)   SBS5: (50.00, 50.00)   SBS6: (85.00, 50.00)
      SBS7: (15.00, 85.00)   SBS8: (50.00, 85.00)   SBS9: (85.00, 85.00)

    The margin=15 setting keeps the outer BSs inside the 100x100 area and gives
    a 35-unit spacing between neighboring grid points, matching COVERAGE_RADIUS=35.
    """
    if int(points_per_axis) != 3:
        raise ValueError("This helper is intended for a 3x3 BS deployment.")
    if not (0.0 <= float(margin) < float(area_size) / 2.0):
        raise ValueError("margin must lie in [0, area_size/2).")

    coords = np.linspace(
        float(margin),
        float(area_size) - float(margin),
        int(points_per_axis),
    )

    positions = []
    for y in coords:
        for x in coords:
            positions.append((float(x), float(y)))

    return positions


# =========================================================
# Environment / Trainer builders
# =========================================================
def make_env(seed, lambda_E, num_users, kappa, use_hard_constraint=False):
    set_seed(seed)

    sbs_positions = generate_grid_coverage_9(
        area_size=AREA_SIZE,
        margin=GRID_MARGIN,
        points_per_axis=GRID_POINTS_PER_AXIS,
    )

    assert len(sbs_positions) == N_BS, (
        f"Expected {N_BS} BSs, got {len(sbs_positions)}"
    )

    sbs_list = [
        SmallCellBaseStation(
            bs_id=i + 1,
            position=pos,
            beam_limit=10,
            coverage_radius=COVERAGE_RADIUS,
        )
        for i, pos in enumerate(sbs_positions)
    ]

    print("[BS Layout] 9-BS 3x3 grid deployment")
    for bs in sbs_list:
        print(
            f"  SBS{bs.bs_id}: "
            f"position=({bs.position[0]:.2f}, {bs.position[1]:.2f})"
        )

    users = [
        UserEquipment(
            i + 1,
            (np.random.uniform(10, 90), np.random.uniform(10, 90))
        )
        for i in range(num_users)
    ]

    env = HAPPOEnvironment(
        base_stations=sbs_list,
        users=users,
        V=5.0,
        power_budget_ratio=POWER_BUDGET_RATIO,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=100,
        bs_top_k=5,
        hard_window_len=HARD_WINDOW_LEN,
        bs_over_penalty=100.0,
        eta_q=1.0,
        alpha_rate=3.0,
        beta_z=1.0,
        use_hard_constraint=use_hard_constraint,
        lambda_E=lambda_E,
        kappa=float(kappa),
    )
    return env


def make_trainer(env):
    return HAPPOTrainer(
        env=env,
        lr_actor_ue=3e-4,
        lr_actor_bs=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef_ue=0.05,
        entropy_coef_bs=0.05,
        value_coef=0.5,
        n_epochs=4,
        minibatch_size=256,
    )



# =========================================================
# Plot-only NPZ helpers
# =========================================================
def _jain_fairness_zero_ok(rate_array) -> float:
    """
    Jain fairness from a [window, n_users] rate array.
    If the whole window is zero-throughput, return 0.0 instead of NaN.
    """
    arr = np.asarray(rate_array, dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2:
        return 0.0

    per_user_avg = arr.mean(axis=0)
    denom = float(np.sum(per_user_avg ** 2))
    if denom < 1e-12:
        return 0.0

    num_users = int(per_user_avg.shape[0])
    return float((np.sum(per_user_avg) ** 2) / (num_users * denom))


def _block_jain_fairness(slot_rates, block: int = 1000):
    """
    Compute one Jain fairness value per non-overlapping block.
    x value is the ending step of each block: 1000, 2000, ...
    """
    arr = np.asarray(slot_rates, dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

    xs, ys = [], []
    T = arr.shape[0]
    for start in range(0, T, int(block)):
        end = min(start + int(block), T)
        chunk = arr[start:end]
        if chunk.shape[0] == 0:
            continue
        xs.append(end)
        ys.append(_jain_fairness_zero_ok(chunk))

    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32)


def _power_history_to_on_matrix(power_history: Dict, expected_T: int):
    """
    Convert power_history dict {bs_id: [power_t]} into:
      bs_ids: [B]
      bs_on_mat: [B, T], 1 if BS is ON else 0
    """
    if not isinstance(power_history, dict) or len(power_history) == 0:
        return np.asarray([], dtype=np.int32), np.zeros((0, int(expected_T)), dtype=np.float32)

    bs_ids = sorted(list(power_history.keys()))
    rows = []
    for bs_id in bs_ids:
        arr = np.asarray(power_history[bs_id], dtype=np.float32).reshape(-1)
        if arr.size < expected_T:
            arr = np.pad(arr, (0, expected_T - arr.size), mode="constant", constant_values=0.0)
        elif arr.size > expected_T:
            arr = arr[:expected_T]
        rows.append((arr > 0.0).astype(np.float32))

    return np.asarray(bs_ids, dtype=np.int32), np.stack(rows, axis=0).astype(np.float32)


def _block_mean_matrix_time(mat: np.ndarray, block: int = 100):
    """
    mat: [B, T]
    return:
      x: [num_blocks]
      y: [num_blocks, B]
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.size == 0 or mat.ndim != 2:
        return np.asarray([], dtype=np.int32), np.zeros((0, 0), dtype=np.float32)

    T = mat.shape[1]
    xs, rows = [], []
    for start in range(0, T, int(block)):
        end = min(start + int(block), T)
        chunk = mat[:, start:end]
        if chunk.shape[1] == 0:
            continue
        xs.append(end)
        rows.append(chunk.mean(axis=1))

    return np.asarray(xs, dtype=np.int32), np.stack(rows, axis=0).astype(np.float32)


def save_train_plot_npz(
    results: Dict,
    npz_path: str,
    env,
    tag: str,
    episode_len: int,
    num_episodes: int,
):
    """
    Save only the requested training plot metrics:
      - global reward every 100 steps
      - handover ratio every 100 steps
    Each 100-step value is a non-overlapping block mean.
    """
    os.makedirs(os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".", exist_ok=True)

    global_reward = np.asarray(results.get("global_reward", []), dtype=np.float32).reshape(-1)
    handover_ratio = np.asarray(results.get("handover_ratio_history", []), dtype=np.float32).reshape(-1)

    x_reward_100, global_reward_100 = block_avg_1d(global_reward, 100)
    x_ho_100, handover_ratio_100 = block_avg_1d(handover_ratio, 100)

    # These should match, but save both defensively.
    x_100 = x_reward_100 if x_reward_100.size > 0 else x_ho_100
    episode_index_100 = ((x_100 - 1) // int(episode_len) + 1).astype(np.int32) if x_100.size > 0 else np.asarray([], dtype=np.int32)
    episode_step_100 = ((x_100 - 1) % int(episode_len) + 1).astype(np.int32) if x_100.size > 0 else np.asarray([], dtype=np.int32)

    np.savez_compressed(
        npz_path,
        tag=str(tag),
        algorithm=np.asarray(["LyMARL"]),
        mode=np.asarray(["train"]),
        n_users=int(env.n_agents),
        n_bs=int(env.n_bs),
        train_steps=int(len(global_reward)),
        episode_len=int(episode_len),
        num_episodes=int(num_episodes),
        kappa=np.asarray([float(env.kappa)], dtype=np.float32),
        handover_budget_ratio=np.asarray([float(env.kappa)], dtype=np.float32),
        x_100=x_100,
        global_reward_x_100=x_reward_100,
        global_reward_100=global_reward_100,
        handover_ratio_x_100=x_ho_100,
        handover_ratio_100=handover_ratio_100,
        episode_index_100=episode_index_100,
        episode_step_100=episode_step_100,
    )

    print(f"✅ Saved train plot npz: {npz_path}")
    print(f"   keys: x_100, global_reward_100, handover_ratio_100, episode_index_100, episode_step_100")


def save_eval_plot_npz(
    results: Dict,
    npz_path: str,
    env,
    tag: str,
    train_num_users: int,
    model_path: str,
    episode_len: int,
):
    """
    Save only the requested evaluation plot metrics:
      - fairness every 1000 steps
      - throughput every 100 steps
      - total mean throughput
      - handover ratio every 100 steps
      - total mean handover ratio
      - each BS ON-ratio every 100 steps
    """
    os.makedirs(os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".", exist_ok=True)

    throughput = np.asarray(results.get("throughput_history", []), dtype=np.float32).reshape(-1)
    slot_rates = np.asarray(results.get("slot_rates", []), dtype=np.float32)
    T = int(throughput.size)

    x_thr_100, throughput_100 = block_avg_1d(throughput, 100)
    throughput_mean = np.asarray([float(np.mean(throughput)) if T > 0 else 0.0], dtype=np.float32)

    # Evaluation handover ratio
    # Per-step definition: number of UEs that handed over in the slot / total number of UEs.
    # The evaluate() method already records this in handover_ratio_history.
    handover_ratio = np.asarray(
        results.get("handover_ratio_history", []),
        dtype=np.float32
    ).reshape(-1)
    handover_count = np.asarray(
        results.get("handover_count_history", []),
        dtype=np.float32
    ).reshape(-1)

    x_ho_100, handover_ratio_100 = block_avg_1d(handover_ratio, 100)
    handover_ratio_mean = np.asarray(
        [float(np.mean(handover_ratio)) if handover_ratio.size > 0 else 0.0],
        dtype=np.float32
    )

    # Count is also saved for debugging/interpretation.
    x_ho_count_100, handover_count_100 = block_avg_1d(handover_count, 100)
    handover_count_mean = np.asarray(
        [float(np.mean(handover_count)) if handover_count.size > 0 else 0.0],
        dtype=np.float32
    )

    fairness_x_1000, fairness_1000 = _block_jain_fairness(slot_rates, block=1000)
    fairness_1000_mean = np.asarray(
        [float(np.mean(fairness_1000)) if fairness_1000.size > 0 else 0.0],
        dtype=np.float32,
    )

    bs_ids, bs_on_mat = _power_history_to_on_matrix(results.get("power_history", {}), expected_T=T)
    on_x_100, bs_on_ratio_100 = _block_mean_matrix_time(bs_on_mat, block=100)

    if bs_on_mat.size > 0:
        bs_on_ratio_per_bs = bs_on_mat.mean(axis=1).astype(np.float32)
        bs_on_ratio_mean = np.asarray([float(bs_on_mat.mean())], dtype=np.float32)
    else:
        bs_on_ratio_per_bs = np.zeros((int(env.n_bs),), dtype=np.float32)
        bs_on_ratio_mean = np.asarray([0.0], dtype=np.float32)

    np.savez_compressed(
        npz_path,
        tag=str(tag),
        algorithm=np.asarray(["LyMARL"]),
        mode=np.asarray(["eval"]),
        train_n_users=int(train_num_users),
        test_n_users=int(env.n_agents),
        n_users=int(env.n_agents),
        n_bs=int(env.n_bs),
        eval_steps=int(T),
        episode_len=int(episode_len),
        model_path=np.asarray([str(model_path)]),
        x_100=x_thr_100,
        throughput_x_100=x_thr_100,
        throughput_100=throughput_100,
        throughput_mean=throughput_mean,
        handover_ratio_x_100=x_ho_100,
        handover_ratio_100=handover_ratio_100,
        handover_ratio_mean=handover_ratio_mean,
        handover_count_x_100=x_ho_count_100,
        handover_count_100=handover_count_100,
        handover_count_mean=handover_count_mean,
        fairness_x_1000=fairness_x_1000,
        fairness_1000=fairness_1000,
        fairness_1000_mean=fairness_1000_mean,
        bs_ids=bs_ids,
        bs_on_ratio_x_100=on_x_100,
        bs_on_ratio_100=bs_on_ratio_100,
        bs_on_ratio_per_bs=bs_on_ratio_per_bs,
        bs_on_ratio_mean=bs_on_ratio_mean,
        energy_budget_ratio=np.asarray([float(POWER_BUDGET_RATIO)], dtype=np.float32),
        kappa=np.asarray([float(env.kappa)], dtype=np.float32),
        handover_budget_ratio=np.asarray([float(env.kappa)], dtype=np.float32),
    )

    print(f"✅ Saved eval plot npz: {npz_path}")
    print(
        "   keys: throughput_100, throughput_mean, "
        "handover_ratio_100, handover_ratio_mean, "
        "handover_count_100, handover_count_mean, "
        "fairness_1000, bs_on_ratio_100"
    )
    print(f"   throughput mean: {float(throughput_mean[0]):.6f} Gbps")
    print(f"   fairness mean from 1000-step blocks: {float(fairness_1000_mean[0]):.6f}")
    print(f"   overall BS ON-ratio mean: {float(bs_on_ratio_mean[0]):.6f}")
    print(f"   eval mean handover ratio: {float(handover_ratio_mean[0]):.6f}")

# =========================================================
# Robust metric extraction helpers
# =========================================================
def _safe_get_attr(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _extract_from_npz(npz_path, candidate_keys):
    if npz_path is None or not os.path.exists(npz_path):
        return None

    data = np.load(npz_path, allow_pickle=True)
    for key in candidate_keys:
        if key in data:
            return data[key]
    return None


def _get_throughput_per_step(env, raw_eval_npz_path=None):
    # 1) env 내부 history 우선
    val = _safe_get_attr(
        env,
        [
            "throughput_history",
            "throughput_per_step",
            "eval_throughput_history",
            "total_throughput_history",
        ],
        default=None,
    )

    if val is not None and len(val) > 0:
        return np.asarray(val, dtype=np.float32)

    # 2) trainer.evaluate가 저장한 raw npz에서 후보 key 탐색
    val = _extract_from_npz(
        raw_eval_npz_path,
        [
            "throughput_per_step",
            "throughput",
            "throughput_history",
            "eval_throughput",
            "total_throughput",
        ],
    )

    if val is not None:
        return np.asarray(val, dtype=np.float32).reshape(-1)

    raise RuntimeError(
        "Cannot find throughput history. "
        "Check trainer.evaluate() output keys or env.throughput_history."
    )


def _get_slot_rates(env, raw_eval_npz_path=None):
    val = _safe_get_attr(
        env,
        ["slot_rates", "eval_slot_rates", "user_rates_per_step", "rate_history"],
        default=None,
    )

    if val is not None and len(val) > 0:
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 2:
            return arr

    val = _extract_from_npz(
        raw_eval_npz_path,
        ["slot_rates", "eval_slot_rates", "user_rates_per_step", "rate_history"],
    )

    if val is not None:
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 2:
            return arr

    return None


def _get_bs_status_history(env, raw_eval_npz_path=None):
    val = _safe_get_attr(
        env,
        ["bs_status_history", "eval_bs_status_history", "bs_on_history"],
        default=None,
    )

    if val is not None and len(val) > 0:
        return val

    val = _extract_from_npz(
        raw_eval_npz_path,
        ["bs_status_history", "eval_bs_status_history", "bs_on_history"],
    )

    if val is not None:
        return val

    return None


def _get_power_history(env, raw_eval_npz_path=None):
    val = _safe_get_attr(
        env,
        ["power_history", "eval_power_history"],
        default=None,
    )

    if val is not None:
        return val

    val = _extract_from_npz(raw_eval_npz_path, ["power_mat", "power_history"])
    if val is not None:
        return val

    return None


def compute_jain_fairness_from_rate_array(rate_array):
    """
    rate_array shape: [window, n_users]
    all-zero slots should already be removed before calling this function.
    """
    if rate_array is None or rate_array.size == 0:
        return np.nan

    if rate_array.ndim != 2:
        return np.nan

    per_user_avg = rate_array.mean(axis=0)
    sum_rates = per_user_avg.sum()
    sum_squared = np.sum(per_user_avg ** 2)
    n_users = len(per_user_avg)

    if sum_squared < 1e-12:
        return np.nan

    return float((sum_rates ** 2) / (n_users * sum_squared))


def build_fairness_100(slot_rates, throughput_per_step, window=100):
    """
    100-step fairness array 생성.
    - 100-step window 전체가 zero-throughput이면 np.nan 저장.
    - 나중에 np.nanmean으로 평균을 내면 hard constraint all-zero 구간이 제외됨.
    """
    T = len(throughput_per_step)
    fairness_100 = []
    throughput_100 = []
    zero_mask = []
    x_100 = []

    for start in range(0, T, window):
        end = min(start + window, T)
        thr_chunk = throughput_per_step[start:end]

        if len(thr_chunk) == 0:
            continue

        zero_window = bool(np.sum(thr_chunk) <= 1e-12)
        throughput_100.append(float(np.mean(thr_chunk)))
        zero_mask.append(zero_window)
        x_100.append(end)

        if zero_window:
            fairness_100.append(np.nan)
            continue

        if slot_rates is None:
            # slot_rates가 없으면 fairness를 재계산할 수 없음.
            # 이 경우 raw eval npz나 env에 이미 저장된 fairness 후보를 써야 함.
            fairness_100.append(np.nan)
            continue

        rate_chunk = np.asarray(slot_rates[start:end], dtype=np.float32)
        active_mask = np.sum(rate_chunk, axis=1) > 1e-12
        active_chunk = rate_chunk[active_mask]
        fairness_100.append(compute_jain_fairness_from_rate_array(active_chunk))

    return (
        np.asarray(x_100, dtype=np.int32),
        np.asarray(throughput_100, dtype=np.float32),
        np.asarray(fairness_100, dtype=np.float32),
        np.asarray(zero_mask, dtype=bool),
    )


def build_bs_on_mat(env, throughput_per_step, raw_eval_npz_path=None):
    """
    BS ON/OFF matrix 생성.
    return shape: [n_bs, T]
    """
    T = len(throughput_per_step)
    bs_ids = sorted([bs.bs_id for bs in env.base_stations])

    # 1) raw npz에 이미 bs_on_mat이 있으면 그대로 사용
    raw_bs_on_mat = _extract_from_npz(raw_eval_npz_path, ["bs_on_mat"])
    if raw_bs_on_mat is not None:
        arr = np.asarray(raw_bs_on_mat, dtype=np.float32)
        if arr.ndim == 2:
            return arr, bs_ids

    # 2) env.bs_status_history에서 생성
    bs_status_history = _get_bs_status_history(env, raw_eval_npz_path)
    if bs_status_history is not None and len(bs_status_history) > 0:
        bs_on_mat = np.zeros((len(bs_ids), T), dtype=np.float32)
        for t in range(min(T, len(bs_status_history))):
            status = bs_status_history[t]
            for i, bs_id in enumerate(bs_ids):
                if isinstance(status, dict):
                    bs_on_mat[i, t] = 1.0 if status.get(bs_id, 0) == 1 else 0.0
                else:
                    # list/array 형태일 가능성까지 대비
                    try:
                        bs_on_mat[i, t] = float(status[i])
                    except Exception:
                        bs_on_mat[i, t] = 0.0
        return bs_on_mat, bs_ids

    # 3) power_history 또는 power_mat에서 생성
    power_history = _get_power_history(env, raw_eval_npz_path)
    if power_history is not None:
        if isinstance(power_history, dict):
            power_mat = []
            for bs_id in bs_ids:
                power_mat.append(np.asarray(power_history[bs_id], dtype=np.float32))
            power_mat = np.stack(power_mat, axis=0)
        else:
            power_mat = np.asarray(power_history, dtype=np.float32)

        if power_mat.ndim == 2:
            # power_mat shape이 [T, B]이면 transpose
            if power_mat.shape[0] == T and power_mat.shape[1] == len(bs_ids):
                power_mat = power_mat.T
            return (power_mat > 0.0).astype(np.float32), bs_ids

    raise RuntimeError(
        "Cannot build bs_on_mat. "
        "Need env.bs_status_history, env.power_history, or raw npz bs_on_mat/power_mat."
    )


def build_queue_mean(env, name, n_steps):
    """
    Optional queue mean trajectory extraction.
    LyMARL env에 Q/Z/G history 이름이 다를 수 있으므로 가능한 경우만 저장.
    """
    candidates = {
        "Q": ["Q_mean", "q_mean", "Q_mean_history", "q_mean_history"],
        "Z": ["Z_mean", "z_mean", "Z_mean_history", "z_mean_history"],
        "G": ["G_mean", "g_mean", "G_mean_history", "g_mean_history"],
    }

    for attr in candidates.get(name, []):
        if hasattr(env, attr):
            val = getattr(env, attr)
            if val is not None and len(val) > 0:
                arr = np.asarray(val, dtype=np.float32).reshape(-1)
                return arr[:n_steps]

    # queue_history dict 구조 대비
    if hasattr(env, "queue_history"):
        qh = getattr(env, "queue_history")
        if isinstance(qh, dict) and name in qh:
            sub = qh[name]
            vals = []
            for t in range(n_steps):
                one_t = []
                if isinstance(sub, dict):
                    for _, hist in sub.items():
                        if len(hist) > t:
                            one_t.append(hist[t])
                if len(one_t) > 0:
                    vals.append(float(np.mean(one_t)))
            if len(vals) > 0:
                return np.asarray(vals, dtype=np.float32)

    return np.full((n_steps,), np.nan, dtype=np.float32)


def save_common_lymarl_eval_npz(env,
                                raw_eval_npz_path,
                                common_npz_path,
                                tag,
                                num_users,
                                lambda_E,
                                model_path):
    """
    Max-SNR/DDPP와 같은 key 구조로 LyMARL evaluation 결과 저장.

    핵심 평균 정의:
    - throughput_mean: 100,000-step 전체 평균, zero-throughput slot 포함.
    - fairness_mean_exclude_zero: 100-step fairness 평균, all-zero 100-step window 제외.
    - bs_on_ratio_first10k_per_bs: 첫 10,000 step에서 BS별 ON-ratio.
    """
    os.makedirs(os.path.dirname(common_npz_path), exist_ok=True)

    throughput_per_step = _get_throughput_per_step(env, raw_eval_npz_path)
    T = len(throughput_per_step)
    throughput_mean = np.asarray([float(np.mean(throughput_per_step))], dtype=np.float32)

    slot_rates = _get_slot_rates(env, raw_eval_npz_path)

    x_100, throughput_100, fairness_100, zero_throughput_100_mask = build_fairness_100(
        slot_rates=slot_rates,
        throughput_per_step=throughput_per_step,
        window=100,
    )

    # slot_rates가 없어서 fairness_100 재계산이 불가능한 경우,
    # raw eval npz에 저장된 fairness 후보를 fallback으로 사용.
    if np.all(np.isnan(fairness_100)):
        raw_fair = _extract_from_npz(
            raw_eval_npz_path,
            ["fairness_100", "fairness", "fairness_history", "eval_fairness"],
        )
        if raw_fair is not None:
            raw_fair = np.asarray(raw_fair, dtype=np.float32).reshape(-1)
            if len(raw_fair) == len(fairness_100):
                fairness_100 = raw_fair.copy()
                fairness_100[zero_throughput_100_mask] = np.nan
            elif len(raw_fair) == T:
                # per-step fairness로 저장되어 있으면 100번째 값마다 샘플링
                sampled = []
                for end in x_100:
                    sampled.append(raw_fair[end - 1])
                fairness_100 = np.asarray(sampled, dtype=np.float32)
                fairness_100[zero_throughput_100_mask] = np.nan

    if fairness_100.size > 0:
        fairness_mean_exclude_zero = np.asarray([float(np.nanmean(fairness_100))], dtype=np.float32)
    else:
        fairness_mean_exclude_zero = np.asarray([np.nan], dtype=np.float32)

    bs_on_mat, bs_ids = build_bs_on_mat(env, throughput_per_step, raw_eval_npz_path)

    bs_on_ratio_total_per_bs = bs_on_mat.mean(axis=1).astype(np.float32)
    bs_on_ratio_total_mean = np.asarray(
        [float(bs_on_ratio_total_per_bs.mean())],
        dtype=np.float32,
    )

    first10k_len = min(10000, bs_on_mat.shape[1])
    bs_on_ratio_first10k_per_bs = bs_on_mat[:, :first10k_len].mean(axis=1).astype(np.float32)
    bs_on_ratio_first10k_mean = np.asarray(
        [float(bs_on_ratio_first10k_per_bs.mean())],
        dtype=np.float32,
    )

    # 10,000-step throughput block summaries
    throughput_10k_x = []
    throughput_10k_avg = []
    throughput_10k_sum = []
    throughput_10k_zero_slots = []

    for start in range(0, T, HARD_WINDOW_LEN):
        chunk = throughput_per_step[start:start + HARD_WINDOW_LEN]
        if len(chunk) == 0:
            continue
        throughput_10k_x.append(int(start + len(chunk)))
        throughput_10k_avg.append(float(np.mean(chunk)))
        throughput_10k_sum.append(float(np.sum(chunk)))
        throughput_10k_zero_slots.append(int(np.sum(chunk <= 1e-12)))

    throughput_10k_x = np.asarray(throughput_10k_x, dtype=np.int32)
    throughput_10k_avg = np.asarray(throughput_10k_avg, dtype=np.float32)
    throughput_10k_sum = np.asarray(throughput_10k_sum, dtype=np.float32)
    throughput_10k_zero_slots = np.asarray(throughput_10k_zero_slots, dtype=np.int32)

    Q_mean = build_queue_mean(env, "Q", T)
    Z_mean = build_queue_mean(env, "Z", T)
    G_mean = build_queue_mean(env, "G", T)

    # Optional handover extraction
    handover_ratio = _safe_get_attr(env, ["handover_ratio_history", "eval_handover_ratio_history"], default=None)
    if handover_ratio is None:
        handover_ratio = _extract_from_npz(raw_eval_npz_path, ["handover_ratio", "handover_ratio_history"])
    handover_ratio = (
        np.asarray(handover_ratio, dtype=np.float32).reshape(-1)
        if handover_ratio is not None else np.asarray([], dtype=np.float32)
    )

    handover_count = _safe_get_attr(env, ["handover_count_history", "eval_handover_count_history"], default=None)
    if handover_count is None:
        handover_count = _extract_from_npz(raw_eval_npz_path, ["handover_count", "handover_count_history"])
    handover_count = (
        np.asarray(handover_count, dtype=np.float32).reshape(-1)
        if handover_count is not None else np.asarray([], dtype=np.float32)
    )

    if slot_rates is None:
        slot_rates_to_save = np.asarray([], dtype=np.float32)
    else:
        slot_rates_to_save = np.asarray(slot_rates, dtype=np.float32)

    np.savez_compressed(
        common_npz_path,
        algorithm=np.asarray(["LyMARL"]),
        tag=str(tag),
        n_users=int(num_users),
        n_bs=int(len(env.base_stations)),
        eval_steps=int(T),
        train_steps=int(TRAIN_STEPS),
        model_path=np.asarray([str(model_path)]),

        # Final comparison keys
        throughput_per_step=throughput_per_step,
        throughput_mean=throughput_mean,
        fairness_100=fairness_100,
        fairness_mean_exclude_zero=fairness_mean_exclude_zero,
        throughput_100=throughput_100,
        zero_throughput_100_mask=zero_throughput_100_mask,
        x_100=x_100,

        bs_on_mat=bs_on_mat,
        bs_ids=np.asarray(bs_ids, dtype=np.int32),
        bs_on_ratio_total_per_bs=bs_on_ratio_total_per_bs,
        bs_on_ratio_total_mean=bs_on_ratio_total_mean,
        bs_on_ratio_first10k_per_bs=bs_on_ratio_first10k_per_bs,
        bs_on_ratio_first10k_mean=bs_on_ratio_first10k_mean,

        throughput_10k_x=throughput_10k_x,
        throughput_10k_avg=throughput_10k_avg,
        throughput_10k_sum=throughput_10k_sum,
        throughput_10k_zero_slots=throughput_10k_zero_slots,

        # Backward-compatible / extra keys
        throughput=throughput_per_step,
        fairness=fairness_100,
        bs_on_ratio_per_bs=bs_on_ratio_total_per_bs,
        bs_on_ratio_mean=bs_on_ratio_total_mean,
        Q_mean=Q_mean,
        Z_mean=Z_mean,
        G_mean=G_mean,
        slot_rates=slot_rates_to_save,
        handover_ratio=handover_ratio,
        handover_count=handover_count,
        handover_budget_ratio=np.asarray([float(env.kappa)], dtype=np.float32),
        energy_budget_ratio=np.asarray([float(POWER_BUDGET_RATIO)], dtype=np.float32),
        lambda_E=np.asarray([float(lambda_E)], dtype=np.float32),
    )

    print(f"✅ Saved LyMARL common eval npz: {common_npz_path}")
    print(f"   throughput_mean including zero slots = {float(throughput_mean[0]):.4f} Gbps")
    print(f"   fairness_mean excluding all-zero 100-step windows = {float(fairness_mean_exclude_zero[0]):.4f}")
    print(f"   first10k BS ON-ratio per BS = {bs_on_ratio_first10k_per_bs}")
    print(f"   first10k BS ON-ratio mean = {float(bs_on_ratio_first10k_mean[0]):.4f}")


# =========================================================
# Kappa sweep summary helpers
# =========================================================
def _kappa_file_tag(kappa: float) -> str:
    """Convert 0.03 -> '0p03' for safe, readable file/folder names."""
    return f"{float(kappa):.2f}".replace(".", "p")


def summarize_eval_results(results: Dict, env) -> Dict:
    """
    Return the four aggregate evaluation metrics requested for one kappa:
      1) mean throughput over all evaluation slots
      2) mean of non-overlapping 1000-step Jain fairness values
      3) mean ON-ratio over all BSs and all evaluation slots
      4) mean per-slot handover ratio
    """
    throughput = np.asarray(results.get("throughput_history", []), dtype=np.float32).reshape(-1)
    slot_rates = np.asarray(results.get("slot_rates", []), dtype=np.float32)
    handover_ratio = np.asarray(
        results.get("handover_ratio_history", []), dtype=np.float32
    ).reshape(-1)

    _, fairness_1000 = _block_jain_fairness(slot_rates, block=1000)
    _, bs_on_mat = _power_history_to_on_matrix(
        results.get("power_history", {}), expected_T=int(throughput.size)
    )

    throughput_mean = float(np.mean(throughput)) if throughput.size > 0 else 0.0
    fairness_1000_mean = float(np.mean(fairness_1000)) if fairness_1000.size > 0 else 0.0
    on_ratio_mean = float(np.mean(bs_on_mat)) if bs_on_mat.size > 0 else 0.0
    handover_ratio_mean = float(np.mean(handover_ratio)) if handover_ratio.size > 0 else 0.0

    if bs_on_mat.size > 0:
        on_ratio_per_bs = bs_on_mat.mean(axis=1).astype(np.float32)
    else:
        on_ratio_per_bs = np.zeros((int(env.n_bs),), dtype=np.float32)

    return {
        "kappa": float(env.kappa),
        "throughput_mean": throughput_mean,
        "fairness_1000_mean": fairness_1000_mean,
        "on_ratio_mean": on_ratio_mean,
        "handover_ratio_mean": handover_ratio_mean,
        "on_ratio_per_bs": on_ratio_per_bs,
        "fairness_1000": fairness_1000.astype(np.float32),
    }



def _extract_eval_plot_metrics(results: Dict, n_bs: int) -> Dict:
    """
    Convert one 10,000-step evaluation result into the exact arrays that are
    saved for plotting. This is called once per shared-seed evaluation run.
    """
    throughput = np.asarray(
        results.get("throughput_history", []), dtype=np.float32
    ).reshape(-1)
    slot_rates = np.asarray(results.get("slot_rates", []), dtype=np.float32)
    T = int(throughput.size)

    throughput_x_100, throughput_100 = block_avg_1d(throughput, 100)
    throughput_mean = float(np.mean(throughput)) if T > 0 else 0.0

    handover_ratio = np.asarray(
        results.get("handover_ratio_history", []), dtype=np.float32
    ).reshape(-1)
    handover_count = np.asarray(
        results.get("handover_count_history", []), dtype=np.float32
    ).reshape(-1)

    handover_ratio_x_100, handover_ratio_100 = block_avg_1d(
        handover_ratio, 100
    )
    handover_count_x_100, handover_count_100 = block_avg_1d(
        handover_count, 100
    )
    handover_ratio_mean = (
        float(np.mean(handover_ratio)) if handover_ratio.size > 0 else 0.0
    )
    handover_count_mean = (
        float(np.mean(handover_count)) if handover_count.size > 0 else 0.0
    )

    fairness_x_1000, fairness_1000 = _block_jain_fairness(
        slot_rates, block=1000
    )
    fairness_1000_mean = (
        float(np.mean(fairness_1000)) if fairness_1000.size > 0 else 0.0
    )

    bs_ids, bs_on_mat = _power_history_to_on_matrix(
        results.get("power_history", {}), expected_T=T
    )
    bs_on_ratio_x_100, bs_on_ratio_100 = _block_mean_matrix_time(
        bs_on_mat, block=100
    )

    if bs_on_mat.size > 0:
        bs_on_ratio_per_bs = bs_on_mat.mean(axis=1).astype(np.float32)
        bs_on_ratio_mean = float(bs_on_mat.mean())
    else:
        bs_ids = np.arange(1, int(n_bs) + 1, dtype=np.int32)
        bs_on_ratio_per_bs = np.zeros((int(n_bs),), dtype=np.float32)
        bs_on_ratio_mean = 0.0

    return {
        "eval_steps": T,
        "throughput_x_100": throughput_x_100,
        "throughput_100": throughput_100.astype(np.float32),
        "throughput_mean": float(throughput_mean),
        "handover_ratio_x_100": handover_ratio_x_100,
        "handover_ratio_100": handover_ratio_100.astype(np.float32),
        "handover_ratio_mean": float(handover_ratio_mean),
        "handover_count_x_100": handover_count_x_100,
        "handover_count_100": handover_count_100.astype(np.float32),
        "handover_count_mean": float(handover_count_mean),
        "fairness_x_1000": fairness_x_1000,
        "fairness_1000": fairness_1000.astype(np.float32),
        "fairness_1000_mean": float(fairness_1000_mean),
        "bs_ids": bs_ids.astype(np.int32),
        "bs_on_ratio_x_100": bs_on_ratio_x_100,
        "bs_on_ratio_100": bs_on_ratio_100.astype(np.float32),
        "bs_on_ratio_per_bs": bs_on_ratio_per_bs.astype(np.float32),
        "bs_on_ratio_mean": float(bs_on_ratio_mean),
    }


def _stack_equal_arrays(metrics: List[Dict], key: str) -> np.ndarray:
    """Stack one metric from all runs and fail clearly if shapes differ."""
    arrays = [np.asarray(m[key]) for m in metrics]
    if not arrays:
        return np.asarray([], dtype=np.float32)

    expected_shape = arrays[0].shape
    for run_idx, arr in enumerate(arrays, start=1):
        if arr.shape != expected_shape:
            raise ValueError(
                f"Evaluation metric shape mismatch for '{key}' at run {run_idx}: "
                f"expected {expected_shape}, got {arr.shape}"
            )
    return np.stack(arrays, axis=0)


def aggregate_evaluation_runs(
    eval_results_list: List[Dict],
    eval_seeds: List[int],
    env,
) -> Dict:
    """
    Aggregate 10 independent evaluation episodes.

    Important interpretation:
      - Every curve is first computed independently for each episode.
      - The saved curve is then the element-wise mean across episodes.
      - Scalar means are the mean of the 10 episode-level scalar means.

    Therefore fairness is averaged across the 10 independently computed
    fairness curves; it is not recomputed from an averaged rate trajectory.
    """
    if len(eval_results_list) == 0:
        raise ValueError("eval_results_list is empty")
    if len(eval_results_list) != len(eval_seeds):
        raise ValueError(
            "Number of evaluation results and number of evaluation seeds differ"
        )

    # Accept either raw evaluate() result dictionaries or already extracted
    # compact metric dictionaries. The main loop passes compact metrics so the
    # ten full 10,000-step trajectories do not remain in memory together.
    metrics = []
    for item in eval_results_list:
        if "eval_steps" in item and "throughput_100" in item:
            metrics.append(item)
        else:
            metrics.append(
                _extract_eval_plot_metrics(item, n_bs=int(env.n_bs))
            )

    expected_steps = int(metrics[0]["eval_steps"])
    for run_idx, metric in enumerate(metrics, start=1):
        if int(metric["eval_steps"]) != expected_steps:
            raise ValueError(
                f"Evaluation length mismatch at run {run_idx}: "
                f"expected {expected_steps}, got {metric['eval_steps']}"
            )

    throughput_100_runs = _stack_equal_arrays(metrics, "throughput_100").astype(np.float32)
    handover_ratio_100_runs = _stack_equal_arrays(metrics, "handover_ratio_100").astype(np.float32)
    handover_count_100_runs = _stack_equal_arrays(metrics, "handover_count_100").astype(np.float32)
    fairness_1000_runs = _stack_equal_arrays(metrics, "fairness_1000").astype(np.float32)
    bs_on_ratio_100_runs = _stack_equal_arrays(metrics, "bs_on_ratio_100").astype(np.float32)
    bs_on_ratio_per_bs_runs = _stack_equal_arrays(metrics, "bs_on_ratio_per_bs").astype(np.float32)

    throughput_mean_runs = np.asarray(
        [m["throughput_mean"] for m in metrics], dtype=np.float32
    )
    handover_ratio_mean_runs = np.asarray(
        [m["handover_ratio_mean"] for m in metrics], dtype=np.float32
    )
    handover_count_mean_runs = np.asarray(
        [m["handover_count_mean"] for m in metrics], dtype=np.float32
    )
    fairness_1000_mean_runs = np.asarray(
        [m["fairness_1000_mean"] for m in metrics], dtype=np.float32
    )
    bs_on_ratio_mean_runs = np.asarray(
        [m["bs_on_ratio_mean"] for m in metrics], dtype=np.float32
    )

    def mean_axis0(arr: np.ndarray) -> np.ndarray:
        return np.mean(arr, axis=0).astype(np.float32)

    def std_axis0(arr: np.ndarray) -> np.ndarray:
        return np.std(arr, axis=0, ddof=0).astype(np.float32)

    return {
        "kappa": float(env.kappa),
        "num_eval_runs": int(len(eval_results_list)),
        "eval_steps_per_run": expected_steps,
        "eval_seeds": np.asarray(eval_seeds, dtype=np.int64),

        "throughput_x_100": np.asarray(metrics[0]["throughput_x_100"], dtype=np.int32),
        "throughput_100_runs": throughput_100_runs,
        "throughput_100_mean": mean_axis0(throughput_100_runs),
        "throughput_100_std": std_axis0(throughput_100_runs),
        "throughput_mean_runs": throughput_mean_runs,
        "throughput_mean": float(np.mean(throughput_mean_runs)),
        "throughput_mean_std": float(np.std(throughput_mean_runs, ddof=0)),

        "handover_ratio_x_100": np.asarray(metrics[0]["handover_ratio_x_100"], dtype=np.int32),
        "handover_ratio_100_runs": handover_ratio_100_runs,
        "handover_ratio_100_mean": mean_axis0(handover_ratio_100_runs),
        "handover_ratio_100_std": std_axis0(handover_ratio_100_runs),
        "handover_ratio_mean_runs": handover_ratio_mean_runs,
        "handover_ratio_mean": float(np.mean(handover_ratio_mean_runs)),
        "handover_ratio_mean_std": float(np.std(handover_ratio_mean_runs, ddof=0)),

        "handover_count_x_100": np.asarray(metrics[0]["handover_count_x_100"], dtype=np.int32),
        "handover_count_100_runs": handover_count_100_runs,
        "handover_count_100_mean": mean_axis0(handover_count_100_runs),
        "handover_count_100_std": std_axis0(handover_count_100_runs),
        "handover_count_mean_runs": handover_count_mean_runs,
        "handover_count_mean": float(np.mean(handover_count_mean_runs)),
        "handover_count_mean_std": float(np.std(handover_count_mean_runs, ddof=0)),

        "fairness_x_1000": np.asarray(metrics[0]["fairness_x_1000"], dtype=np.int32),
        "fairness_1000_runs": fairness_1000_runs,
        "fairness_1000_mean_curve": mean_axis0(fairness_1000_runs),
        "fairness_1000_std_curve": std_axis0(fairness_1000_runs),
        "fairness_1000_mean_runs": fairness_1000_mean_runs,
        "fairness_1000_mean": float(np.mean(fairness_1000_mean_runs)),
        "fairness_1000_mean_std": float(np.std(fairness_1000_mean_runs, ddof=0)),

        "bs_ids": np.asarray(metrics[0]["bs_ids"], dtype=np.int32),
        "bs_on_ratio_x_100": np.asarray(metrics[0]["bs_on_ratio_x_100"], dtype=np.int32),
        "bs_on_ratio_100_runs": bs_on_ratio_100_runs,
        "bs_on_ratio_100_mean": mean_axis0(bs_on_ratio_100_runs),
        "bs_on_ratio_100_std": std_axis0(bs_on_ratio_100_runs),
        "bs_on_ratio_per_bs_runs": bs_on_ratio_per_bs_runs,
        "bs_on_ratio_per_bs": mean_axis0(bs_on_ratio_per_bs_runs),
        "bs_on_ratio_per_bs_std": std_axis0(bs_on_ratio_per_bs_runs),
        "bs_on_ratio_mean_runs": bs_on_ratio_mean_runs,
        "bs_on_ratio_mean": float(np.mean(bs_on_ratio_mean_runs)),
        "bs_on_ratio_mean_std": float(np.std(bs_on_ratio_mean_runs, ddof=0)),
    }


def save_multi_eval_plot_npz(
    aggregate: Dict,
    npz_path: str,
    env,
    tag: str,
    train_num_users: int,
    model_path: str,
    episode_len: int,
):
    """
    Save the 10-evaluation aggregate.

    Backward-compatible plotting keys contain the 10-run mean:
      throughput_100, throughput_mean,
      handover_ratio_100, handover_ratio_mean,
      handover_count_100, handover_count_mean,
      fairness_1000, fairness_1000_mean,
      bs_on_ratio_100, bs_on_ratio_per_bs, bs_on_ratio_mean.

    The corresponding *_runs and *_std keys are also saved.
    """
    os.makedirs(
        os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".",
        exist_ok=True,
    )

    np.savez_compressed(
        npz_path,
        tag=str(tag),
        algorithm=np.asarray(["LyMARL"]),
        mode=np.asarray(["eval_10run_mean"]),
        train_n_users=int(train_num_users),
        test_n_users=int(env.n_agents),
        n_users=int(env.n_agents),
        n_bs=int(env.n_bs),
        eval_steps=int(aggregate["eval_steps_per_run"]),
        eval_steps_per_run=int(aggregate["eval_steps_per_run"]),
        num_eval_runs=int(aggregate["num_eval_runs"]),
        total_eval_steps=int(
            aggregate["num_eval_runs"] * aggregate["eval_steps_per_run"]
        ),
        episode_len=int(episode_len),
        model_path=np.asarray([str(model_path)]),
        eval_seeds=np.asarray(aggregate["eval_seeds"], dtype=np.int64),

        # Throughput: legacy keys are the 10-run means.
        x_100=np.asarray(aggregate["throughput_x_100"], dtype=np.int32),
        throughput_x_100=np.asarray(aggregate["throughput_x_100"], dtype=np.int32),
        throughput_100=np.asarray(aggregate["throughput_100_mean"], dtype=np.float32),
        throughput_100_mean=np.asarray(aggregate["throughput_100_mean"], dtype=np.float32),
        throughput_100_std=np.asarray(aggregate["throughput_100_std"], dtype=np.float32),
        throughput_100_runs=np.asarray(aggregate["throughput_100_runs"], dtype=np.float32),
        throughput_mean=np.asarray([aggregate["throughput_mean"]], dtype=np.float32),
        throughput_mean_std=np.asarray([aggregate["throughput_mean_std"]], dtype=np.float32),
        throughput_mean_runs=np.asarray(aggregate["throughput_mean_runs"], dtype=np.float32),

        # Handover ratio/count: legacy keys are the 10-run means.
        handover_ratio_x_100=np.asarray(aggregate["handover_ratio_x_100"], dtype=np.int32),
        handover_ratio_100=np.asarray(aggregate["handover_ratio_100_mean"], dtype=np.float32),
        handover_ratio_100_mean=np.asarray(aggregate["handover_ratio_100_mean"], dtype=np.float32),
        handover_ratio_100_std=np.asarray(aggregate["handover_ratio_100_std"], dtype=np.float32),
        handover_ratio_100_runs=np.asarray(aggregate["handover_ratio_100_runs"], dtype=np.float32),
        handover_ratio_mean=np.asarray([aggregate["handover_ratio_mean"]], dtype=np.float32),
        handover_ratio_mean_std=np.asarray([aggregate["handover_ratio_mean_std"]], dtype=np.float32),
        handover_ratio_mean_runs=np.asarray(aggregate["handover_ratio_mean_runs"], dtype=np.float32),

        handover_count_x_100=np.asarray(aggregate["handover_count_x_100"], dtype=np.int32),
        handover_count_100=np.asarray(aggregate["handover_count_100_mean"], dtype=np.float32),
        handover_count_100_mean=np.asarray(aggregate["handover_count_100_mean"], dtype=np.float32),
        handover_count_100_std=np.asarray(aggregate["handover_count_100_std"], dtype=np.float32),
        handover_count_100_runs=np.asarray(aggregate["handover_count_100_runs"], dtype=np.float32),
        handover_count_mean=np.asarray([aggregate["handover_count_mean"]], dtype=np.float32),
        handover_count_mean_std=np.asarray([aggregate["handover_count_mean_std"]], dtype=np.float32),
        handover_count_mean_runs=np.asarray(aggregate["handover_count_mean_runs"], dtype=np.float32),

        # Fairness: average the 10 independently computed 1000-step curves.
        fairness_x_1000=np.asarray(aggregate["fairness_x_1000"], dtype=np.int32),
        fairness_1000=np.asarray(aggregate["fairness_1000_mean_curve"], dtype=np.float32),
        fairness_1000_mean_curve=np.asarray(aggregate["fairness_1000_mean_curve"], dtype=np.float32),
        fairness_1000_std_curve=np.asarray(aggregate["fairness_1000_std_curve"], dtype=np.float32),
        fairness_1000_runs=np.asarray(aggregate["fairness_1000_runs"], dtype=np.float32),
        fairness_1000_mean=np.asarray([aggregate["fairness_1000_mean"]], dtype=np.float32),
        fairness_1000_mean_std=np.asarray([aggregate["fairness_1000_mean_std"]], dtype=np.float32),
        fairness_1000_mean_runs=np.asarray(aggregate["fairness_1000_mean_runs"], dtype=np.float32),

        # BS ON ratio: [100-step block, BS] after averaging 10 runs.
        bs_ids=np.asarray(aggregate["bs_ids"], dtype=np.int32),
        bs_on_ratio_x_100=np.asarray(aggregate["bs_on_ratio_x_100"], dtype=np.int32),
        bs_on_ratio_100=np.asarray(aggregate["bs_on_ratio_100_mean"], dtype=np.float32),
        bs_on_ratio_100_mean=np.asarray(aggregate["bs_on_ratio_100_mean"], dtype=np.float32),
        bs_on_ratio_100_std=np.asarray(aggregate["bs_on_ratio_100_std"], dtype=np.float32),
        bs_on_ratio_100_runs=np.asarray(aggregate["bs_on_ratio_100_runs"], dtype=np.float32),
        bs_on_ratio_per_bs=np.asarray(aggregate["bs_on_ratio_per_bs"], dtype=np.float32),
        bs_on_ratio_per_bs_std=np.asarray(aggregate["bs_on_ratio_per_bs_std"], dtype=np.float32),
        bs_on_ratio_per_bs_runs=np.asarray(aggregate["bs_on_ratio_per_bs_runs"], dtype=np.float32),
        bs_on_ratio_mean=np.asarray([aggregate["bs_on_ratio_mean"]], dtype=np.float32),
        bs_on_ratio_mean_std=np.asarray([aggregate["bs_on_ratio_mean_std"]], dtype=np.float32),
        bs_on_ratio_mean_runs=np.asarray(aggregate["bs_on_ratio_mean_runs"], dtype=np.float32),

        energy_budget_ratio=np.asarray([float(POWER_BUDGET_RATIO)], dtype=np.float32),
        kappa=np.asarray([float(env.kappa)], dtype=np.float32),
        handover_budget_ratio=np.asarray([float(env.kappa)], dtype=np.float32),
    )

    print(f"✅ Saved 10-run mean eval plot npz: {npz_path}")
    print(f"   shared evaluation seeds: {aggregate['eval_seeds'].tolist()}")
    print(
        f"   throughput mean over {aggregate['num_eval_runs']} runs: "
        f"{aggregate['throughput_mean']:.6f} ± "
        f"{aggregate['throughput_mean_std']:.6f} Gbps"
    )
    print(
        f"   fairness mean over {aggregate['num_eval_runs']} runs: "
        f"{aggregate['fairness_1000_mean']:.6f} ± "
        f"{aggregate['fairness_1000_mean_std']:.6f}"
    )
    print(
        f"   BS ON-ratio mean over {aggregate['num_eval_runs']} runs: "
        f"{aggregate['bs_on_ratio_mean']:.6f} ± "
        f"{aggregate['bs_on_ratio_mean_std']:.6f}"
    )
    print(
        f"   handover ratio mean over {aggregate['num_eval_runs']} runs: "
        f"{aggregate['handover_ratio_mean']:.6f} ± "
        f"{aggregate['handover_ratio_mean_std']:.6f}"
    )


def save_kappa_sweep_summary(summary_rows: List[Dict], run_dir: str):
    """Save the 10-run mean/std trend for all kappa values to NPZ and CSV."""
    if not summary_rows:
        print("[WARN] No kappa summary rows to save.")
        return

    kappas = np.asarray([row["kappa"] for row in summary_rows], dtype=np.float32)

    def col(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in summary_rows], dtype=np.float32)

    throughput_mean = col("throughput_mean")
    throughput_mean_std = col("throughput_mean_std")
    fairness_1000_mean = col("fairness_1000_mean")
    fairness_1000_mean_std = col("fairness_1000_mean_std")
    on_ratio_mean = col("bs_on_ratio_mean")
    on_ratio_mean_std = col("bs_on_ratio_mean_std")
    handover_ratio_mean = col("handover_ratio_mean")
    handover_ratio_mean_std = col("handover_ratio_mean_std")

    on_ratio_per_bs = np.stack(
        [np.asarray(row["bs_on_ratio_per_bs"], dtype=np.float32) for row in summary_rows],
        axis=0,
    )
    on_ratio_per_bs_std = np.stack(
        [np.asarray(row["bs_on_ratio_per_bs_std"], dtype=np.float32) for row in summary_rows],
        axis=0,
    )
    eval_seed_matrix = np.stack(
        [np.asarray(row["eval_seeds"], dtype=np.int64) for row in summary_rows],
        axis=0,
    )

    summary_npz_path = os.path.join(run_dir, "kappa_sweep_eval_10run_summary.npz")
    np.savez_compressed(
        summary_npz_path,
        algorithm=np.asarray(["LyMARL"]),
        n_bs=int(N_BS),
        train_n_users=int(TRAIN_NUM_USERS),
        eval_n_users=int(EVAL_NUM_USERS),
        train_steps=int(TRAIN_STEPS),
        num_eval_runs=int(EVAL_RUNS),
        eval_steps_per_run=int(EVAL_STEPS_PER_RUN),
        total_eval_steps_per_kappa=int(EVAL_TOTAL_STEPS),
        eval_seed_matrix=eval_seed_matrix,
        kappa=kappas,
        throughput_mean=throughput_mean,
        throughput_mean_std=throughput_mean_std,
        fairness_1000_mean=fairness_1000_mean,
        fairness_1000_mean_std=fairness_1000_mean_std,
        on_ratio_mean=on_ratio_mean,
        on_ratio_mean_std=on_ratio_mean_std,
        handover_ratio_mean=handover_ratio_mean,
        handover_ratio_mean_std=handover_ratio_mean_std,
        on_ratio_per_bs=on_ratio_per_bs,
        on_ratio_per_bs_std=on_ratio_per_bs_std,
        energy_budget_ratio=np.asarray([float(POWER_BUDGET_RATIO)], dtype=np.float32),
        lambda_E=np.asarray([float(LAMBDA_E)], dtype=np.float32),
    )

    summary_csv_path = os.path.join(run_dir, "kappa_sweep_eval_10run_summary.csv")
    fieldnames = [
        "kappa",
        "num_eval_runs",
        "eval_steps_per_run",
        "throughput_mean_gbps",
        "throughput_std_gbps",
        "fairness_1000_mean",
        "fairness_1000_std",
        "on_ratio_mean",
        "on_ratio_std",
        "handover_ratio_mean",
        "handover_ratio_std",
        "evaluation_seeds",
    ]
    for i in range(int(N_BS)):
        fieldnames.extend([f"bs{i + 1}_on_ratio", f"bs{i + 1}_on_ratio_std"])

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            csv_row = {
                "kappa": row["kappa"],
                "num_eval_runs": row["num_eval_runs"],
                "eval_steps_per_run": row["eval_steps_per_run"],
                "throughput_mean_gbps": row["throughput_mean"],
                "throughput_std_gbps": row["throughput_mean_std"],
                "fairness_1000_mean": row["fairness_1000_mean"],
                "fairness_1000_std": row["fairness_1000_mean_std"],
                "on_ratio_mean": row["bs_on_ratio_mean"],
                "on_ratio_std": row["bs_on_ratio_mean_std"],
                "handover_ratio_mean": row["handover_ratio_mean"],
                "handover_ratio_std": row["handover_ratio_mean_std"],
                "evaluation_seeds": ";".join(
                    str(int(seed)) for seed in row["eval_seeds"]
                ),
            }
            for bi, value in enumerate(row["bs_on_ratio_per_bs"]):
                csv_row[f"bs{bi + 1}_on_ratio"] = float(value)
                csv_row[f"bs{bi + 1}_on_ratio_std"] = float(
                    row["bs_on_ratio_per_bs_std"][bi]
                )
            writer.writerow(csv_row)

    print(f"✅ Saved combined 10-run kappa summary NPZ: {summary_npz_path}")
    print(f"✅ Saved combined 10-run kappa summary CSV: {summary_csv_path}")


# =========================================================
# Main experiment loop
# =========================================================
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULT_ROOT, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=False)

    print("\n" + "#" * 108)
    print("# LyMARL kappa sweep: independent training + 10 shared-seed evaluations per kappa")
    print(f"# BS={N_BS}, training UE={TRAIN_NUM_USERS}, evaluation UE={EVAL_NUM_USERS}")
    print(f"# KAPPA_VALUES={KAPPA_VALUES}")
    print(
        f"# TRAIN: {TRAIN_EPISODES} episodes x {EPISODE_LEN} steps "
        f"= {TRAIN_STEPS} steps per kappa"
    )
    print(
        f"# EVAL : {EVAL_RUNS} independent episodes x {EVAL_STEPS_PER_RUN} steps "
        f"= {EVAL_TOTAL_STEPS} total eval steps per kappa"
    )
    print("# Training hard constraint: OFF / Evaluation hard constraint: ON")
    print(f"# Training seed is fixed at {TRAIN_SEED} for every kappa.")
    print("# One random list of 10 distinct evaluation seeds is generated once and shared by every kappa.")
    print("# Saved plotting arrays are element-wise averages across the 10 evaluation episodes.")
    print(f"# Unique output directory: {run_dir}")
    print("#" * 108 + "\n")

    # --------------------------------------------------
    # Generate the evaluation seeds ONCE.
    # The same ordered list is reused for every kappa.
    # --------------------------------------------------
    shared_eval_seeds = generate_random_eval_seeds(EVAL_RUNS)

    # Defensive check: evaluation seeds must not accidentally equal the
    # training seed. This is extremely unlikely, but keeping them disjoint
    # makes the experimental protocol explicit.
    while TRAIN_SEED in shared_eval_seeds:
        shared_eval_seeds = generate_random_eval_seeds(EVAL_RUNS)

    seed_npz_path = os.path.join(run_dir, "experiment_seeds.npz")
    seed_txt_path = os.path.join(run_dir, "experiment_seeds.txt")
    np.savez_compressed(
        seed_npz_path,
        train_seed=np.asarray(TRAIN_SEED, dtype=np.int64),
        shared_eval_seeds=np.asarray(shared_eval_seeds, dtype=np.int64),
        kappa_values=np.asarray(KAPPA_VALUES, dtype=np.float32),
        eval_runs=np.asarray(EVAL_RUNS, dtype=np.int32),
        eval_steps_per_run=np.asarray(EVAL_STEPS_PER_RUN, dtype=np.int32),
    )
    with open(seed_txt_path, "w", encoding="utf-8") as f:
        f.write(f"TRAIN_SEED={TRAIN_SEED}\n")
        f.write("SHARED_EVAL_SEEDS=" + ",".join(map(str, shared_eval_seeds)) + "\n")
        f.write("KAPPA_VALUES=" + ",".join(map(str, KAPPA_VALUES)) + "\n")

    print(f"[SEED SETUP] Fixed training seed : {TRAIN_SEED}")
    print(f"[SEED SETUP] Shared eval seeds  : {shared_eval_seeds}")
    print(f"[SEED SETUP] Saved seed metadata: {seed_npz_path}")

    summary_rows = []

    for sweep_index, kappa in enumerate(KAPPA_VALUES, start=1):
        kappa = float(kappa)
        kappa_tag = _kappa_file_tag(kappa)
        kappa_dir = os.path.join(run_dir, f"kappa_{kappa_tag}")
        train_dir = os.path.join(kappa_dir, "train")
        eval_dir = os.path.join(kappa_dir, "eval")
        model_dir = os.path.join(kappa_dir, "model")
        os.makedirs(train_dir, exist_ok=False)
        os.makedirs(eval_dir, exist_ok=False)
        os.makedirs(model_dir, exist_ok=False)

        train_plot_npz_path = os.path.join(
            train_dir,
            f"LyMARL_B{N_BS}_U{TRAIN_NUM_USERS}_kappa_{kappa_tag}_train_"
            f"ep{TRAIN_EPISODES}x{EPISODE_LEN}_lambda_{LAMBDA_E}.npz",
        )
        model_path = os.path.join(
            model_dir,
            f"LyMARL_B{N_BS}_U{TRAIN_NUM_USERS}_kappa_{kappa_tag}_train_"
            f"ep{TRAIN_EPISODES}x{EPISODE_LEN}_lambda_{LAMBDA_E}.pt",
        )
        eval_plot_npz_path = os.path.join(
            eval_dir,
            f"LyMARL_B{N_BS}_trainU{TRAIN_NUM_USERS}_testU{EVAL_NUM_USERS}_"
            f"kappa_{kappa_tag}_eval_mean{EVAL_RUNS}x{EVAL_STEPS_PER_RUN}_"
            f"sharedseed_lambda_{LAMBDA_E}.npz",
        )

        print("\n" + "=" * 108)
        print(
            f"KAPPA SWEEP {sweep_index}/{len(KAPPA_VALUES)} | "
            f"kappa={kappa:.2f} | train from scratch -> "
            f"evaluate {EVAL_RUNS} times"
        )
        print("=" * 108 + "\n")

        # --------------------------------------------------
        # 1) Independent training from scratch at this kappa
        # --------------------------------------------------
        train_env = make_env(
            seed=TRAIN_SEED,
            lambda_E=LAMBDA_E,
            num_users=TRAIN_NUM_USERS,
            kappa=kappa,
            use_hard_constraint=False,
        )
        train_trainer = make_trainer(train_env)

        train_results = train_trainer.train(
            n_steps=TRAIN_STEPS,
            update_interval=128,
            save_npz_path=None,
            plot_qzg_path=None,
            episode_len=EPISODE_LEN,
            num_episodes=TRAIN_EPISODES,
        )

        save_train_plot_npz(
            results=train_results,
            npz_path=train_plot_npz_path,
            env=train_trainer.env,
            tag=f"LyMARL_B{N_BS}_U{TRAIN_NUM_USERS}_kappa_{kappa:.2f}_train",
            episode_len=EPISODE_LEN,
            num_episodes=TRAIN_EPISODES,
        )
        train_trainer.save_model(model_path)

        del train_results
        del train_trainer
        del train_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --------------------------------------------------
        # 2) Ten independent 10,000-step evaluations
        #    Reuse the exact same ordered seed list for every kappa.
        # --------------------------------------------------
        eval_seeds = list(shared_eval_seeds)
        print(f"[KAPPA {kappa:.2f}] Shared evaluation seeds: {eval_seeds}")

        eval_metrics_list = []
        last_eval_env = None

        for eval_index, eval_seed in enumerate(eval_seeds, start=1):
            print("\n" + "-" * 108)
            print(
                f"[KAPPA {kappa:.2f}] EVALUATION EPISODE "
                f"{eval_index}/{EVAL_RUNS} | seed={eval_seed} | "
                f"steps={EVAL_STEPS_PER_RUN}"
            )
            print("-" * 108)

            eval_env = make_env(
                seed=eval_seed,
                lambda_E=LAMBDA_E,
                num_users=EVAL_NUM_USERS,
                kappa=kappa,
                use_hard_constraint=True,
            )
            eval_trainer = make_trainer(eval_env)
            eval_trainer.load_actor_only(model_path)

            eval_results = eval_trainer.evaluate(
                n_steps=EVAL_STEPS_PER_RUN,
                save_npz_path=None,
            )
            run_summary = summarize_eval_results(eval_results, eval_trainer.env)
            eval_metrics_list.append(
                _extract_eval_plot_metrics(eval_results, n_bs=int(eval_trainer.env.n_bs))
            )
            print(
                f"[EVAL EPISODE {eval_index:02d}/{EVAL_RUNS}] "
                f"seed={eval_seed} | "
                f"Thr={run_summary['throughput_mean']:.6f} Gbps | "
                f"Fair(1k)={run_summary['fairness_1000_mean']:.6f} | "
                f"ON={run_summary['on_ratio_mean']:.6f} | "
                f"HO={run_summary['handover_ratio_mean']:.6f}"
            )

            if last_eval_env is not None:
                del last_eval_env
            last_eval_env = eval_env

            # Only compact plot metrics are retained across runs.
            del eval_results
            del eval_trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if last_eval_env is None:
            raise RuntimeError("No evaluation environment was created")

        aggregate = aggregate_evaluation_runs(
            eval_results_list=eval_metrics_list,
            eval_seeds=eval_seeds,
            env=last_eval_env,
        )

        save_multi_eval_plot_npz(
            aggregate=aggregate,
            npz_path=eval_plot_npz_path,
            env=last_eval_env,
            tag=(
                f"LyMARL_B{N_BS}_trainU{TRAIN_NUM_USERS}_testU{EVAL_NUM_USERS}_"
                f"kappa_{kappa:.2f}_mean_of_{EVAL_RUNS}_shared_seed_evals"
            ),
            train_num_users=TRAIN_NUM_USERS,
            model_path=model_path,
            episode_len=EPISODE_LEN,
        )

        aggregate["model_path"] = model_path
        aggregate["eval_npz_path"] = eval_plot_npz_path
        summary_rows.append(aggregate)

        print("\n" + "=" * 108)
        print(
            f"KAPPA={kappa:.2f} | FINAL AVERAGE AFTER "
            f"{EVAL_RUNS} EVALUATION EPISODES"
        )
        print(f"  Evaluation seeds              : {eval_seeds}")
        print(
            f"  Throughput mean               : "
            f"{aggregate['throughput_mean']:.6f} ± "
            f"{aggregate['throughput_mean_std']:.6f} Gbps"
        )
        print(
            f"  Fairness mean (1000-step)     : "
            f"{aggregate['fairness_1000_mean']:.6f} ± "
            f"{aggregate['fairness_1000_mean_std']:.6f}"
        )
        print(
            f"  BS ON-ratio mean              : "
            f"{aggregate['bs_on_ratio_mean']:.6f} ± "
            f"{aggregate['bs_on_ratio_mean_std']:.6f}"
        )
        print(
            f"  Handover ratio mean           : "
            f"{aggregate['handover_ratio_mean']:.6f} ± "
            f"{aggregate['handover_ratio_mean_std']:.6f}"
        )
        print(f"  Per-BS ON-ratio mean          : {aggregate['bs_on_ratio_per_bs']}")
        print(f"  Per-BS ON-ratio std           : {aggregate['bs_on_ratio_per_bs_std']}")
        print(f"  Saved averaged evaluation NPZ : {eval_plot_npz_path}")
        print("=" * 108 + "\n")

        del eval_metrics_list
        del aggregate
        del last_eval_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------
    # 3) Save one combined trend table for all kappa values
    # --------------------------------------------------
    save_kappa_sweep_summary(summary_rows, run_dir)

    print("\n" + "=" * 124)
    print(f"FINAL KAPPA SWEEP SUMMARY: MEAN ± STD ACROSS {EVAL_RUNS} SHARED-SEED EVALUATIONS")
    print("=" * 124)
    print(
        f"{'kappa':>8} | {'throughput mean±std':>24} | "
        f"{'fairness mean±std':>22} | {'ON mean±std':>20} | "
        f"{'HO mean±std':>20}"
    )
    print("-" * 124)
    for row in summary_rows:
        print(
            f"{row['kappa']:8.2f} | "
            f"{row['throughput_mean']:10.6f}±{row['throughput_mean_std']:<10.6f} | "
            f"{row['fairness_1000_mean']:9.6f}±{row['fairness_1000_mean_std']:<9.6f} | "
            f"{row['bs_on_ratio_mean']:8.6f}±{row['bs_on_ratio_mean_std']:<8.6f} | "
            f"{row['handover_ratio_mean']:8.6f}±{row['handover_ratio_mean_std']:<8.6f}"
        )
    print("=" * 124)
    print(f"\n✅ Completed all kappa experiments. Results are in: {run_dir}\n")


if __name__ == "__main__":
    main()