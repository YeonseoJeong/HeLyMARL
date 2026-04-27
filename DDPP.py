import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import defaultdict

from basestation import BaseStation, SmallCellBaseStation
from user_equipment import UserEquipment
from core import generate_triangle_coverage

################################################
## BASELINE DDPP ALGORITHM IMPLEMENTATION
################################################

class DDPPAlgorithm:
    """
    Q_u(t+1) = [Q_u(t) + r*(t) - R_u(t)]_+
    Z_b(t+1) = [Z_b(t) + P_b(t) - P_bar]_+
    G_u(t+1)
    Score (UE->BS request) = Q_u × R - Z_b × P_max
    """

    def __init__(self,
                 base_stations: List[BaseStation],
                 users: List[UserEquipment],
                 V: float = 20.0,
                 power_budget_ratio: float = 0.7,
                 max_slots: int = 1000,
                 enable_mobility: bool = True,
                 enable_channel_variation: bool = True,
                 seed: int = None,
                 use_hard_constraint: bool = False,
                 hard_window_len: int = 10000,
                 lambda_E: float = 1.0):

        self.users = users
        self.base_stations = [bs for bs in base_stations if bs.bs_id != 0]
        self.V = V
        self.power_budget_ratio = power_budget_ratio
        self.max_slots = max_slots
        self.enable_mobility = enable_mobility
        self.enable_channel_variation = enable_channel_variation
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_hard_constraint = bool(use_hard_constraint)
        self.hard_window_len = int(hard_window_len)
        self.lambda_E = float(lambda_E)

        self.hard_on_limit = {
            bs.bs_id: int(np.floor(self.hard_window_len * self.power_budget_ratio)) for bs in self.base_stations
        }
        self.bs_on_used_in_window = {bs.bs_id: 0 for bs in self.base_stations}
        self.window_step = 0
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        self.P_max = {
            bs.bs_id: 10 ** (bs.tx_power_dbm / 10) / 1000  # [W] # 20 dBm -> 0.1 W
            for bs in self.base_stations
        }

        self.P_bar = {
            bs.bs_id: self.power_budget_ratio * self.P_max[bs.bs_id]  # [W]
            for bs in self.base_stations
        }

        self.Q_u = {ue.ue_id: 0.1 for ue in users}
        self.Z_b = {bs.bs_id: 0.01 for bs in self.base_stations}
        self.gamma_max = {ue.ue_id: 5.0 for ue in users}

        self.G_u = {ue.ue_id: 0.0 for ue in users}
        self.kappa = 0.1

        self.m_u = {ue.ue_id: None for ue in users}

        # ==========================================
        # Tracking
        # ==========================================
        self.associations_history = []
        self.bs_status_history = []
        self.throughput_history = []
        self.power_history = defaultdict(list)
        self.queue_history = {
            'Q': defaultdict(list), 
            'Z': defaultdict(list),
            'G': defaultdict(list)
        }
        self.handover_count_history = []
        self.handover_ratio_history = []
        self.G_mean_history = []

        self.user_rate_history = defaultdict(list)
        self.fairness_history = []
        self.slot_rates = []

        # ==========================================
        # Environment
        # ==========================================
        self.noise_dbm = -174 + 10 * np.log10(500e6) + 5
        self.noise_watts = 10 ** (self.noise_dbm / 10) / 1000
        self.mobility_speed = 1.0
        self.area_size = 100
        self.channel_gains = defaultdict(dict)
        self.fading_std = 4.0

    # ==========================================
    # Environment Dynamics
    # ==========================================
    def set_hard_constraint(self, enabled:bool):
        self.use_hard_constraint = bool(enabled)

    def compute_handover_indicator(self, ue_id: int, candidate_bs_id) -> float:
        if candidate_bs_id is None:
            return 0.0

        prev_bs = self.m_u.get(ue_id, None)

        if prev_bs is None:
            return 0.0

        return 1.0 if candidate_bs_id != prev_bs else 0.0
    
    def compute_handover(self, scheduled_users: dict) -> tuple:
        current_serving_bs = {ue.ue_id: None for ue in self.users}

        for bs_id, ue_id in scheduled_users.items():
            if ue_id is not None:
                current_serving_bs[ue_id] = bs_id

        h_u = {}
        handover_count = 0.0

        for ue in self.users:
            ue_id = ue.ue_id
            curr_bs = current_serving_bs[ue_id]

            h = self.compute_handover_indicator(ue_id, curr_bs)

            h_u[ue_id] = h
            handover_count += h

        for ue in self.users:
            ue_id = ue.ue_id
            served_bs = current_serving_bs[ue_id]
            if served_bs is not None:
                self.m_u[ue_id] = served_bs

        return h_u, handover_count
    
    def update_user_positions(self):
        if not self.enable_mobility:
            return
        for user in self.users:
            dx, dy = self.rng.normal(0, self.mobility_speed, 2)
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
                    fading_db = self.rng.normal(0, self.fading_std)
                else:
                    prev_db = 10 * np.log10(self.channel_gains[u.ue_id][bs.bs_id] + 1e-10)
                    fading_db = 0.9 * prev_db + self.rng.normal(0, self.fading_std * np.sqrt(1 - 0.9**2))
                self.channel_gains[u.ue_id][bs.bs_id] = 10 ** (fading_db / 10)

    # ==========================================
    # PHY Layer
    # ==========================================
    def calculate_achievable_rate(self, user_id: int, bs_id: int, interferer_status: dict = None) -> float:
        """ Returns: rate [Gbps] """
        user = next(u for u in self.users if u.ue_id == user_id)
        bs = next(b for b in self.base_stations if b.bs_id == bs_id)

        dist = max(1.0, bs.distance_to(user.position))
        rx_dbm = bs.receive_power(dist)

        # small-scale fading gain (linear) 
        gain = self.channel_gains.get(user_id, {}).get(bs_id, 1.0)
        rx_dbm += 10 * np.log10(gain + 1e-12)

        # dBm -> W
        rx_watts = 10 ** (rx_dbm / 10) / 1000

        # interference
        interference = 0.0
        for other_bs in self.base_stations:
            if other_bs.bs_id == bs_id:
                continue

            # 간섭원 선택 기준
            if interferer_status is None:
                # 이전 슬롯 ON만 간섭원
                is_on = (self.prev_power.get(other_bs.bs_id, 0.0) > 0.0)
            else:
                # 현재 슬롯 ON만 간섭원
                is_on = (interferer_status.get(other_bs.bs_id, 0) == 1)

            if not is_on:
                continue

            other_dist = max(1.0, other_bs.distance_to(user.position))
            other_rx_dbm = other_bs.receive_power(other_dist)
            # interference 쪽에는 channel gain 안 붙임
            interference += 10 ** (other_rx_dbm / 10) / 1000

        sinr = rx_watts / (self.noise_watts + interference)
        rate_bps = bs.bandwidth * np.log2(1.0 + sinr)
        return max(0.0, rate_bps / 1e9)

    # ==========================================
    # DPP Core
    # ==========================================
    def compute_aux_rate(self, u_id: int) -> float:
        """r* = min{gamma_max, V/Q_u} [Gbps]"""
        Q_u = self.Q_u[u_id]
        return min(self.gamma_max[u_id], self.V / max(Q_u, 1e-6))  # 상한 gamma_max

    def user_association(self, t: int) -> dict:
        """Score = Q_u × R_tilde - (Z_b + lambda_E) × P_max - G_u × h_candidate"""
        associations = {}

        for user in self.users:
            best_bs = None
            best_score = -np.inf

            for bs in self.base_stations:
                # 결정 단계는 prev_power 기반 간섭 -> R_tilde 
                R_tilde = self.calculate_achievable_rate(user.ue_id, bs.bs_id)
                power = self.P_max[bs.bs_id]
                h_candidate = self.compute_handover_indicator(user.ue_id, bs.bs_id)

                score = (
                    self.Q_u[user.ue_id] * R_tilde 
                    - (self.Z_b[bs.bs_id] + self.lambda_E) * power
                    - self.G_u[user.ue_id] * h_candidate
                )

                if score > best_score:
                    best_score = score
                    best_bs = bs.bs_id

            associations[user.ue_id] = best_bs if best_score > 0 else None

        return associations

    def bs_scheduling(self, associations: dict) -> tuple:
        """
        R = Q_u × R_tilde - G_u × h_candidate
        threshold = (Z_b + lambda_E) × P_max
        if R > threshold => ON and serve best UE
        """
        bs_status = {}
        scheduled_users = {}

        proposers = defaultdict(list)
        for ue_id, bs_id in associations.items():
            if bs_id is not None:
                proposers[bs_id].append(ue_id)

        for bs in self.base_stations:
            if not proposers[bs.bs_id]:
                bs_status[bs.bs_id] = 0
                scheduled_users[bs.bs_id] = None
                continue

            best_score_qr = 0.0
            best_ue = None

            for ue_id in proposers[bs.bs_id]:
                # 결정 단계는 prev_power 기반 간섭 -> R_tilde 
                R_tilde = self.calculate_achievable_rate(ue_id, bs.bs_id)
                h_candidate = self.compute_handover_indicator(ue_id, bs.bs_id)

                score_qr = (
                    self.Q_u[ue_id] * R_tilde
                    - self.G_u[ue_id] * h_candidate
                )

                if score_qr > best_score_qr:
                    best_score_qr = score_qr
                    best_ue = ue_id

            power = self.P_max[bs.bs_id]
            threshold = (self.Z_b[bs.bs_id] + self.lambda_E) * power

            if best_score_qr > threshold:
                bs_status[bs.bs_id] = 1
                scheduled_users[bs.bs_id] = best_ue
            else:
                bs_status[bs.bs_id] = 0
                scheduled_users[bs.bs_id] = None

        return bs_status, scheduled_users


    def update_queues(self, scheduled_users: dict, bs_status: dict) -> dict:
        """Queue updates + returns actual served rates R(SINR)"""
        actual_rates = {u.ue_id: 0.0 for u in self.users}

        for bs_id, ue_id in scheduled_users.items():
            if ue_id is not None and bs_status[bs_id] == 1:
                # actual rate는 "현재 슬롯 ON(bs_status)"을 간섭원으로 사용
                actual_rate = self.calculate_achievable_rate(ue_id, bs_id, interferer_status=bs_status)
                actual_rates[ue_id] = actual_rate
                self.user_rate_history[ue_id].append(actual_rate)

        for user in self.users:
            aux_rate = self.compute_aux_rate(user.ue_id)
            served_rate = actual_rates[user.ue_id]
            self.Q_u[user.ue_id] = max(1e-12, self.Q_u[user.ue_id] + (aux_rate - served_rate))

        for bs in self.base_stations:
            power = self.P_max[bs.bs_id] if bs_status[bs.bs_id] == 1 else 0.0
            budget = self.P_bar[bs.bs_id]
            self.Z_b[bs.bs_id] = max(0.001, self.Z_b[bs.bs_id] + (power - budget))

        h_u, handover_count = self.compute_handover(scheduled_users)
        for ue in self.users:
            self.G_u[ue.ue_id] = max(0.0, self.G_u[ue.ue_id] + h_u[ue.ue_id] - self.kappa)
        
        return actual_rates, h_u, handover_count

    def update_max_rates(self):
        for user in self.users:
            max_R_tilde = 0.0
            for bs in self.base_stations:
                # R_max는 결정/관측과 같은 기준(prev 간섭)으로 계산
                R_tilde = self.calculate_achievable_rate(user.ue_id, bs.bs_id)
                max_R_tilde = max(max_R_tilde, R_tilde)
            self.gamma_max[user.ue_id] = max_R_tilde if max_R_tilde > 0 else 1.0

    # ==========================================
    # Metrics
    # ==========================================
    def calculate_jain_fairness(self, window: int = 100, exclude_all_zero_slots: bool = True) -> float:
        recent_slots = self.slot_rates if len(self.slot_rates) < window else self.slot_rates[-window:]

        if not recent_slots:
            return np.nan

        rate_array = np.asarray(recent_slots, dtype=np.float32)

        if rate_array.ndim != 2:
            return np.nan

        if exclude_all_zero_slots:
            active_mask = np.sum(rate_array, axis=1) > 1e-12
            rate_array = rate_array[active_mask]

            # 중요: 최근 window 전체가 all-zero면 0이 아니라 nan으로 둬야 평균에서 제외 가능
            if rate_array.shape[0] == 0:
                return np.nan

        per_user_avg = rate_array.mean(axis=0)

        sum_rates = per_user_avg.sum()
        sum_squared = np.sum(per_user_avg ** 2)
        n_users = len(per_user_avg)

        if sum_squared < 1e-12:
            return np.nan

        return float((sum_rates ** 2) / (n_users * sum_squared))
    
    # ==========================================
    # Simulation
    # ==========================================
    def run_slot(self, t: int):
        # NOTE: 결정(association/scheduling/gamma_max)은 prev_power(t-1) 간섭 기준 -> R_tilde
        # queue update(실제 서비스): current bs_status(t) 간섭 기반 -> R
        self.update_user_positions()
        self.update_channel_gains(t)
        self.update_max_rates()

        associations = self.user_association(t)
        bs_status, scheduled = self.bs_scheduling(associations)
        if self.use_hard_constraint:
            for bs in self.base_stations:
                used = self.bs_on_used_in_window[bs.bs_id]
                limit = self.hard_on_limit[bs.bs_id]

                if used >= limit:
                    bs_status[bs.bs_id] = 0
                    scheduled[bs.bs_id] = None

        actual_rates, h_u, handover_count = self.update_queues(scheduled, bs_status)
        self.window_step += 1

        for bs in self.base_stations:
            if bs_status.get(bs.bs_id, 0) == 1:
                self.bs_on_used_in_window[bs.bs_id] += 1

        if self.window_step % self.hard_window_len == 0:
            self.bs_on_used_in_window = {
                bs.bs_id: 0 for bs in self.base_stations
            }
        # prev_power 업데이트 -> 다음 슬롯 R_tilde 계산에 사용
        self.prev_power = {bs_id: (bs_status.get(bs_id, 0) * self.P_max[bs_id]) for bs_id in self.P_max}

        self.associations_history.append(scheduled)
        self.bs_status_history.append(bs_status)
        self.throughput_history.append(sum(actual_rates.values()))

        for bs_id, status in bs_status.items():
            power_watts = status * self.P_max[bs_id]
            self.power_history[bs_id].append(power_watts)

        for ue_id in self.Q_u:
            self.queue_history['Q'][ue_id].append(self.Q_u[ue_id])
        for bs_id in self.Z_b:
            self.queue_history['Z'][bs_id].append(self.Z_b[bs_id])
        for ue_id in self.G_u:
            self.queue_history['G'][ue_id].append(self.G_u[ue_id])

        self.handover_count_history.append(handover_count)
        self.handover_ratio_history.append(handover_count / max(1, len(self.users)))
        self.G_mean_history.append(np.mean(list(self.G_u.values())))
        self.slot_rates.append([actual_rates.get(u.ue_id, 0.0) for u in self.users])
        self.fairness_history.append(self.calculate_jain_fairness(window=100))

    def run_simulation(self):
        print(f"\n{'='*60}")
        print(f"  Pure DPP Algorithm (Decision: prev ON, Actual: current ON)")
        print(f"{'='*60}")
        print(f"  V = {self.V}")
        print(f"  Power budget ratio = {self.power_budget_ratio}")
        print(f"  Lambda E = {self.lambda_E}")
        print(f"  Total slots = {self.max_slots}")
        print(f"{'='*60}\n")
        self.recent_fair_list = []

        for t in range(self.max_slots):

            if hasattr(self, "V_schedule_fn"):
                self.V = self.V_schedule_fn(t)
            if t in [0, 10000, 20000, 30000]:
                print(f"[DDPP] t={t:6d} | V={self.V} | lambda_E={self.lambda_E}")
            self.run_slot(t)
            
            if (t + 1) % 100 == 0:
                recent_thr = float(np.mean(self.throughput_history[-100:]))
                recent_fair = float(self.calculate_jain_fairness(window=100))
                if not np.isnan(recent_fair):
                    self.recent_fair_list.append(recent_fair)

                on_ratios = {}
                for bs in self.base_stations:
                    on_count = sum(1 for s in self.bs_status_history[-100:] if s.get(bs.bs_id, 0) == 1)
                    on_ratios[bs.bs_id] = on_count / 100

                ratio_str = ', '.join([f'BS{b}:{r:.2f}' for b, r in on_ratios.items()])
                ho_count_100 = float(np.mean(self.handover_count_history[-100:]))
                ho_ratio_100 = float(np.mean(self.handover_ratio_history[-100:]))
                G_mean_now = float(self.G_mean_history[-1]) if len(self.G_mean_history) > 0 else 0.0

                fair_str = "nan" if np.isnan(recent_fair) else f"{recent_fair:.3f}"
                print(f"Slot {t+1:4d} | Thr: {recent_thr:.3f} Gbps | "
                    f"Fair(JFI@100): {fair_str} | ON: [{ratio_str}] | "
                    f"HO(100): count={ho_count_100:.3f} ratio={ho_ratio_100:.4f}/{self.kappa:.4f} | "
                    f"Gmean:{G_mean_now:.3f}")
        
        print(f"\n{'='*60}")
        overall_thr = float(np.mean(self.throughput_history))
        overall_fair = float(np.nanmean(self.recent_fair_list)) if len(self.recent_fair_list) > 0 else np.nan  # ep 전체 슬롯 기준 JFI
        print(f"  Avg Throughput: {overall_thr:.3f} Gbps")
        print(f"  JFI (avg over 100 slots): {overall_fair:.4f}")

        print(f"\n  Power Budget Check:")
        for bs in self.base_stations:
            avg_power = np.mean(self.power_history[bs.bs_id])
            budget = self.P_bar[bs.bs_id]
            on_ratio = sum(1 for p in self.power_history[bs.bs_id] if p > 0) / len(self.power_history[bs.bs_id])
            print(f"    BS {bs.bs_id}: {avg_power:.4f}W / {budget:.4f}W | "
                  f"ON={on_ratio:.3f} (target={self.power_budget_ratio})")

        print(f"\n  Queue Value Ranges:")
        q_vals = [self.Q_u[u.ue_id] for u in self.users]
        z_vals = [self.Z_b[bs.bs_id] for bs in self.base_stations]
        rtilde_vals = []
        w_vals = []
        for u in self.users:
            for bs in self.base_stations:
                R_tilde = self.calculate_achievable_rate(u.ue_id, bs.bs_id)
                rtilde_vals.append(R_tilde)
                w_vals.append(self.Q_u[u.ue_id] * R_tilde - self.Z_b[bs.bs_id] * self.P_max[bs.bs_id])
        print(f"    Q_u: [{min(q_vals):.4f}, {max(q_vals):.4f}]")
        print(f"    Z_b: [{min(z_vals):.6f}, {max(z_vals):.6f}]")
        print(f"    R_tilde: [{min(rtilde_vals):.4f}, {max(rtilde_vals):.4f}] Gbps")
        print(f"    W: [{min(w_vals):.4f}, {max(w_vals):.4f}]")
        print(f"{'='*60}\n")
        
    def plot_results(self):
        bs_ids = sorted([bs.bs_id for bs in self.base_stations])
        T = len(self.bs_status_history)
        if T == 0:
            print("No simulation data to plot.")
            return

        fig, axes = plt.subplots(4, 1, figsize=(12, 12))

        # =========================================================
        # (1) ON ratio per 10,000 slots 
        # =========================================================
        ax = axes[0]
        block = 10000

        x_points = []
        block_ratios = {bs_id: [] for bs_id in bs_ids}

        for start in range(0, T, block):
            chunk = self.bs_status_history[start:start + block]
            if len(chunk) == 0:
                continue
            end_step = start + len(chunk)
            x_points.append(end_step)

            for bs_id in bs_ids:
                on_count = sum(1 for s in chunk if s.get(bs_id, 0) == 1)
                block_ratios[bs_id].append(on_count / len(chunk))

        for bs_id in bs_ids:
            ax.plot(x_points, block_ratios[bs_id], label=f'BS{bs_id}')

        ax.set_title(f'ON Ratio per BS (every {block} slots)', fontweight='bold')
        ax.set_xlabel('Slot (end of each block)')
        ax.set_ylabel('ON ratio')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

        # =========================================================
        # (2) Local ON ratio per BS (last 10,000 slots) 
        # =========================================================
        ax = axes[1]
        window = 1000
        last_window = 10000

        start_idx = max(0, T - last_window)
        status_slice = self.bs_status_history[start_idx:]
        T_slice = len(status_slice)

        for bs_id in bs_ids:
            local_ratios = []
            x_pts = []
            for i in range(0, T_slice, window):
                chunk = status_slice[i:i + window]
                if len(chunk) == 0:
                    continue
                on_count = sum(1 for s in chunk if s.get(bs_id, 0) == 1)
                ratio = on_count / len(chunk)
                local_ratios.append(ratio)
                global_step = start_idx + i + len(chunk)
                x_pts.append(global_step)

            ax.plot(x_pts, local_ratios, label=f'BS{bs_id}')

        ax.set_title(f'Local ON Ratio per BS (last {last_window} slots, window={window})', fontweight='bold')
        ax.set_xlabel('Slot')
        ax.set_ylabel('ON ratio (per 1000 slots)')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

        # =========================================================
        # (3) Throughput Trend (last 10,000 slots) 
        # =========================================================
        ax = axes[2]
        T_thr = len(self.throughput_history)
        start_idx_thr = max(0, T_thr - last_window)
        thr_slice = self.throughput_history[start_idx_thr:]

        avg_vals = []
        x_pts = []
        for i in range(0, len(thr_slice), window):
            chunk = thr_slice[i:i + window]
            if len(chunk) == 0:
                continue
            avg_vals.append(np.mean(chunk))
            global_step = start_idx_thr + i + len(chunk)
            x_pts.append(global_step)

        ax.plot(x_pts, avg_vals, linewidth=2, color='orange', label=f'Avg Throughput (window={window})')

        ax.set_title(f'Throughput Trend (last {last_window} slots)', fontweight='bold')
        ax.set_xlabel('Slot')
        ax.set_ylabel('Throughput [Gbps]')
        ax.grid(alpha=0.3)
        ax.legend()

        # =========================================================
        # (4) Mean Queue Trajectories: Q, Z, G
        # =========================================================
        ax = axes[3]

        Q_mean = []
        Z_mean = []
        G_mean = []

        for t in range(T):
            q_vals = [self.queue_history['Q'][ue.ue_id][t] for ue in self.users]
            z_vals = [self.queue_history['Z'][bs.bs_id][t] for bs in self.base_stations]
            g_vals = [self.queue_history['G'][ue.ue_id][t] for ue in self.users]

            Q_mean.append(np.mean(q_vals))
            Z_mean.append(np.mean(z_vals))
            G_mean.append(np.mean(g_vals))

        window_q = 1000
        x_q = []
        Q_block = []
        Z_block = []
        G_block = []

        for i in range(0, T, window_q):
            q_chunk = Q_mean[i:i + window_q]
            z_chunk = Z_mean[i:i + window_q]
            g_chunk = G_mean[i:i + window_q]

            if len(q_chunk) == 0:
                continue

            x_q.append(i + len(q_chunk))
            Q_block.append(np.mean(q_chunk))
            Z_block.append(np.mean(z_chunk))
            G_block.append(np.mean(g_chunk))

        ax.plot(x_q, Q_block, linewidth=2, label='Q')
        ax.plot(x_q, Z_block, linewidth=2, label='Z')
        ax.plot(x_q, G_block, linewidth=2, label='G')

        ax.set_title(f'Mean Queue Trajectories (window={window_q})', fontweight='bold')
        ax.set_xlabel('Slot')
        ax.set_ylabel('Mean Queue Value')
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig('dpp_summary_modified.png', dpi=300)
        plt.show()

    def save_results_npz(self, npz_path: str, tag: str = "DDPP"):
        import os
        os.makedirs(os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".", exist_ok=True)

        throughput = np.asarray(self.throughput_history, dtype=np.float32)
        fairness = np.asarray(self.fairness_history, dtype=np.float32)
        handover_ratio = np.asarray(self.handover_ratio_history, dtype=np.float32)

        bs_ids = sorted([bs.bs_id for bs in self.base_stations])

        power_mat = []
        for bs_id in bs_ids:
            power_mat.append(np.asarray(self.power_history[bs_id], dtype=np.float32))
        power_mat = np.stack(power_mat, axis=0) if len(power_mat) > 0 else np.zeros((0, len(throughput)), dtype=np.float32)

        if power_mat.size > 0:
            bs_on_mat = (power_mat > 0.0).astype(np.float32)
            bs_on_ratio_per_bs = bs_on_mat.mean(axis=1)
            bs_on_ratio_mean = np.asarray([float(bs_on_ratio_per_bs.mean())], dtype=np.float32)
        else:
            bs_on_ratio_per_bs = np.asarray([], dtype=np.float32)
            bs_on_ratio_mean = np.asarray([np.nan], dtype=np.float32)

        # Queue mean trajectories
        T = len(throughput)
        Q_mean = []
        Z_mean = []
        G_mean = []

        for t in range(T):
            q_vals = [self.queue_history["Q"][ue.ue_id][t] for ue in self.users]
            z_vals = [self.queue_history["Z"][bs.bs_id][t] for bs in self.base_stations]
            g_vals = [self.queue_history["G"][ue.ue_id][t] for ue in self.users]

            Q_mean.append(float(np.mean(q_vals)))
            Z_mean.append(float(np.mean(z_vals)))
            G_mean.append(float(np.mean(g_vals)))

        Q_mean = np.asarray(Q_mean, dtype=np.float32)
        Z_mean = np.asarray(Z_mean, dtype=np.float32)
        G_mean = np.asarray(G_mean, dtype=np.float32)

        np.savez_compressed(
            npz_path,
            tag=str(tag),
            n_users=int(len(self.users)),
            n_bs=int(len(self.base_stations)),

            throughput=throughput,
            fairness=fairness,

            power_mat=power_mat,
            bs_ids=np.asarray(bs_ids, dtype=np.int32),
            bs_on_ratio_per_bs=bs_on_ratio_per_bs,
            bs_on_ratio_mean=bs_on_ratio_mean,

            handover_ratio=handover_ratio,
            handover_count=np.asarray(self.handover_count_history, dtype=np.float32),
            handover_budget_ratio=np.asarray([float(self.kappa)], dtype=np.float32),
            energy_budget_ratio=np.asarray([float(self.power_budget_ratio)], dtype=np.float32),
            lambda_E=np.asarray([float(self.lambda_E)], dtype=np.float32),

            Q_mean=Q_mean,
            Z_mean=Z_mean,
            G_mean=G_mean,
        )

        print(f"✅ Saved DDPP results npz: {npz_path}")

if __name__ == "__main__":
    area_size = 100
    num_users = 20
    lambda_E = 30.0

    sbs_positions = generate_triangle_coverage(area_size, 35)
    sbs_list = [SmallCellBaseStation(i + 1, pos, 10, 35) for i, pos in enumerate(sbs_positions)]
    users = [UserEquipment(i + 1, (np.random.uniform(10, 90), np.random.uniform(10, 90)))
             for i in range(num_users)]

    dpp = DDPPAlgorithm(sbs_list, users, V=5.0, power_budget_ratio=0.6, max_slots=50000, use_hard_constraint=True, hard_window_len=10000, lambda_E=lambda_E)
    dpp.run_simulation()
    dpp.plot_results()
    dpp.save_results_npz(f"results_compare/DDPP_eval_lambda_{lambda_E}.npz", tag=f"DDPP_{lambda_E}")
