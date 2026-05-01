import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque

from env.basestation import BaseStation
from env.user_equipment import UserEquipment


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
    def calculate_jain_fairness(self, rate_history, window: int = 100, exclude_all_zero_slots: bool = True):
        recent = rate_history if len(rate_history) < window else rate_history[-window:]

        if not recent:
            return 0.0

        rate_array = np.asarray(recent, dtype=np.float32)

        if rate_array.ndim != 2:
            return 0.0

        if exclude_all_zero_slots:
            active_slot_mask = np.sum(rate_array, axis=1) > 1e-12
            rate_array = rate_array[active_slot_mask]

            if rate_array.shape[0] == 0:
                return 0.0

        per_user_avg = rate_array.mean(axis=0)

        sum_rates = per_user_avg.sum()
        sum_squared = (per_user_avg ** 2).sum()
        n_users = len(per_user_avg)

        if sum_squared < 1e-12:
            return 0.0

        return float((sum_rates ** 2) / (n_users * sum_squared))