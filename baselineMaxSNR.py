import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import defaultdict

from env.basestation import BaseStation, SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage, generate_five_bs_coverage


################################################
## MAX-SNR BASELINE WITH ENERGY BUDGET
################################################

class MaxSNRBaseline:
    """
    Max-SNR / Max-Rate baseline with energy budget.

    Policy:
    1) Each UE requests the BS with the highest achievable rate.
    2) Each BS schedules one UE with the highest achievable rate among requesters.
    3) BS activation is constrained by power_budget_ratio using a hard window budget.
    """

    def __init__(self,
                 base_stations: List[BaseStation],
                 users: List[UserEquipment],
                 power_budget_ratio: float = 0.6,
                 max_slots: int = 50000,
                 enable_mobility: bool = True,
                 enable_channel_variation: bool = True,
                 seed: int = None,
                 hard_window_len: int = 10000,
                 lambda_E: float = 1.0,
                 kappa: float = 0.1):

        self.users = users
        self.base_stations = [bs for bs in base_stations if bs.bs_id != 0]

        self.power_budget_ratio = float(power_budget_ratio)
        self.max_slots = int(max_slots)
        self.enable_mobility = enable_mobility
        self.enable_channel_variation = enable_channel_variation
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.hard_window_len = int(hard_window_len)
        self.lambda_E = float(lambda_E)
        self.kappa = float(kappa)

        self.avg_rate_pf = {ue.ue_id: 1e-3 for ue in users}
        self.pf_beta = 0.01
        self.pf_eps = 1e-6

        self.P_max = {
            bs.bs_id: 10 ** ((bs.tx_power_dbm - 30) / 10)
            for bs in self.base_stations
        }

        self.P_bar = {
            bs.bs_id: self.power_budget_ratio * self.P_max[bs.bs_id]
            for bs in self.base_stations
        }

        # Hard budget state
        self.hard_on_limit = {
            bs.bs_id: int(np.floor(self.hard_window_len * self.power_budget_ratio))
            for bs in self.base_stations
        }
        self.bs_on_used_in_window = {
            bs.bs_id: 0 for bs in self.base_stations
        }
        self.window_step = 0

        # Previous serving BS for handover
        self.m_u = {ue.ue_id: None for ue in users}
        self.G_u = {ue.ue_id: 0.0 for ue in users}

        # Previous power for interference during decision
        self.prev_power = {bs.bs_id: 0.0 for bs in self.base_stations}

        # Tracking
        self.associations_history = []
        self.bs_status_history = []
        self.throughput_history = []
        self.power_history = defaultdict(list)

        self.handover_count_history = []
        self.handover_ratio_history = []
        self.G_mean_history = []

        self.user_rate_history = defaultdict(list)
        self.slot_rates = []
        self.fairness_history = []

        self.queue_history = {
            "G": defaultdict(list)
        }

        # PHY/environment
        self.noise_dbm = -174 + 10 * np.log10(500e6) + 5
        self.noise_watts = 10 ** (self.noise_dbm / 10) / 1000
        self.mobility_speed = 1.0
        self.area_size = 100
        self.channel_gains = defaultdict(dict)
        self.fading_std = 4.0

    # =========================================================
    # Environment dynamics
    # =========================================================
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
                    prev_db = 10 * np.log10(
                        self.channel_gains[u.ue_id][bs.bs_id] + 1e-10
                    )
                    fading_db = (
                        0.9 * prev_db
                        + self.rng.normal(0, self.fading_std * np.sqrt(1 - 0.9 ** 2))
                    )

                self.channel_gains[u.ue_id][bs.bs_id] = 10 ** (fading_db / 10)

    # =========================================================
    # PHY layer
    # =========================================================    
    def calculate_achievable_rate(self,
                                  user_id: int,
                                  bs_id: int,
                                  interferer_status: dict = None) -> float:
        """
        Returns achievable rate [Gbps] based on SNR only.
        Inter-cell interference is ignored for the Max-SNR baseline.
        """
        user = next(u for u in self.users if u.ue_id == user_id)
        bs = next(b for b in self.base_stations if b.bs_id == bs_id)

        dist = max(1.0, bs.distance_to(user.position))
        rx_dbm = bs.receive_power(dist)

        gain = self.channel_gains.get(user_id, {}).get(bs_id, 1.0)
        rx_dbm += 10 * np.log10(gain + 1e-12)

        rx_watts = 10 ** (rx_dbm / 10) / 1000

        interference = 0.0
        for other_bs in self.base_stations:
            if other_bs.bs_id == bs_id:
                continue

            if interferer_status is None:
                is_on = self.prev_power.get(other_bs.bs_id, 0.0) > 0.0
            else:
                is_on = interferer_status.get(other_bs.bs_id, 0) == 1

            if not is_on:
                continue

            other_dist = max(1.0, other_bs.distance_to(user.position))
            other_rx_dbm = other_bs.receive_power(other_dist)
            interference += 10 ** (other_rx_dbm / 10) / 1000

        sinr = rx_watts / (self.noise_watts + interference)
        snr = rx_watts / self.noise_watts
        rate_bps = bs.bandwidth * np.log2(1.0 + sinr)

        return max(0.0, rate_bps / 1e9)

    def compute_pf_score(self, ue_id: int, bs_id: int) -> float:
        rate = self.calculate_achievable_rate(ue_id, bs_id)
        avg_rate = self.avg_rate_pf.get(ue_id, 1e-3)

        return rate / (avg_rate + self.pf_eps)
    # =========================================================
    # Max-SNR decision
    # =========================================================
    def user_association(self, t: int) -> dict:
        """
        Each UE requests the BS with the highest achievable rate.
        In this implementation, achievable rate is used as the Max-SNR proxy.
        """
        associations = {}

        for user in self.users:
            ue_id = user.ue_id

            best_bs = None
            best_rate = -np.inf

            for bs in self.base_stations:
                bs_id = bs.bs_id
                rate = self.calculate_achievable_rate(ue_id, bs_id)

                if rate > best_rate:
                    best_rate = rate
                    best_bs = bs_id

            associations[ue_id] = best_bs

        return associations

    def bs_scheduling(self, associations: dict) -> tuple:
        """
        Each BS schedules one UE with the highest achievable rate among requesters.
        """
        bs_status = {}
        scheduled_users = {}

        proposers = defaultdict(list)
        for ue_id, bs_id in associations.items():
            if bs_id is not None:
                proposers[bs_id].append(ue_id)

        for bs in self.base_stations:
            bs_id = bs.bs_id

            if not proposers[bs_id]:
                bs_status[bs_id] = 0
                scheduled_users[bs_id] = None
                continue

            best_ue = None
            best_rate = -np.inf

            for ue_id in proposers[bs_id]:
                rate = self.calculate_achievable_rate(ue_id, bs_id)

                if rate > best_rate:
                    best_rate = rate
                    best_ue = ue_id

            threshold = self.lambda_E * self.P_max[bs_id]

            if best_rate >= threshold:
                bs_status[bs_id] = 1
                scheduled_users[bs_id] = best_ue
            else:
                bs_status[bs_id] = 0
                scheduled_users[bs_id] = None

        return bs_status, scheduled_users

    def apply_energy_budget(self, bs_status: dict, scheduled_users: dict) -> tuple:
        """
        Hard window energy budget:
        If a BS already used its ON budget within the current window, force it OFF.
        """
        for bs in self.base_stations:
            bs_id = bs.bs_id

            used = self.bs_on_used_in_window[bs_id]
            limit = self.hard_on_limit[bs_id]

            if used >= limit:
                bs_status[bs_id] = 0
                scheduled_users[bs_id] = None

        return bs_status, scheduled_users

    # =========================================================
    # Handover
    # =========================================================
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

    # =========================================================
    # Metrics
    # =========================================================
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

            if rate_array.shape[0] == 0:
                return np.nan

        per_user_avg = rate_array.mean(axis=0)

        sum_rates = per_user_avg.sum()
        sum_squared = np.sum(per_user_avg ** 2)
        n_users = len(per_user_avg)

        if sum_squared < 1e-12:
            return np.nan

        return float((sum_rates ** 2) / (n_users * sum_squared))

    # =========================================================
    # Simulation
    # =========================================================
    def run_slot(self, t: int):
        self.update_user_positions()
        self.update_channel_gains(t)

        associations = self.user_association(t)
        bs_status, scheduled_users = self.bs_scheduling(associations)

        bs_status, scheduled_users = self.apply_energy_budget(bs_status, scheduled_users)

        actual_rates = {u.ue_id: 0.0 for u in self.users}

        for bs_id, ue_id in scheduled_users.items():
            if ue_id is not None and bs_status.get(bs_id, 0) == 1:
                actual_rate = self.calculate_achievable_rate(
                    ue_id,
                    bs_id,
                    interferer_status=bs_status
                )
                actual_rates[ue_id] = actual_rate
                self.user_rate_history[ue_id].append(actual_rate)

        for ue in self.users:
            ue_id = ue.ue_id
            self.avg_rate_pf[ue_id] = (
                (1.0 - self.pf_beta) * self.avg_rate_pf[ue_id]
                + self.pf_beta * actual_rates.get(ue_id, 0.0)
            )

        h_u, handover_count = self.compute_handover(scheduled_users)

        for ue in self.users:
            self.G_u[ue.ue_id] = max(
                0.0,
                self.G_u[ue.ue_id] + h_u[ue.ue_id] - self.kappa
            )

        self.window_step += 1

        for bs in self.base_stations:
            bs_id = bs.bs_id
            if bs_status.get(bs_id, 0) == 1:
                self.bs_on_used_in_window[bs_id] += 1

        if self.window_step % self.hard_window_len == 0:
            self.bs_on_used_in_window = {
                bs.bs_id: 0 for bs in self.base_stations
            }

        self.prev_power = {
            bs_id: bs_status.get(bs_id, 0) * self.P_max[bs_id]
            for bs_id in self.P_max
        }

        self.associations_history.append(scheduled_users)
        self.bs_status_history.append(bs_status)
        self.throughput_history.append(sum(actual_rates.values()))

        for bs_id, status in bs_status.items():
            power_watts = status * self.P_max[bs_id]
            self.power_history[bs_id].append(power_watts)

        for ue_id in self.G_u:
            self.queue_history["G"][ue_id].append(self.G_u[ue_id])

        self.handover_count_history.append(handover_count)
        self.handover_ratio_history.append(handover_count / max(1, len(self.users)))
        self.G_mean_history.append(np.mean(list(self.G_u.values())))

        self.slot_rates.append([
            actual_rates.get(u.ue_id, 0.0) for u in self.users
        ])

        self.fairness_history.append(self.calculate_jain_fairness(window=100))

    def run_simulation(self):
        print(f"\n{'=' * 70}")
        print("  Max-SNR Baseline with Energy Budget")
        print(f"{'=' * 70}")
        print(f"  Power budget ratio = {self.power_budget_ratio}")
        print(f"  Lambda E = {self.lambda_E}")
        print(f"  Handover budget kappa = {self.kappa}")
        print(f"  Total slots = {self.max_slots}")
        print(f"{'=' * 70}\n")

        self.recent_fair_list = []

        for t in range(self.max_slots):
            self.run_slot(t)

            if (t + 1) % 100 == 0:
                recent_thr = float(np.mean(self.throughput_history[-100:]))
                recent_fair = float(self.calculate_jain_fairness(window=100))

                if not np.isnan(recent_fair):
                    self.recent_fair_list.append(recent_fair)

                on_ratios = {}
                for bs in self.base_stations:
                    on_count = sum(
                        1 for s in self.bs_status_history[-100:]
                        if s.get(bs.bs_id, 0) == 1
                    )
                    on_ratios[bs.bs_id] = on_count / 100

                ratio_str = ", ".join([
                    f"BS{b}:{r:.2f}" for b, r in on_ratios.items()
                ])

                ho_count_100 = float(np.mean(self.handover_count_history[-100:]))
                ho_ratio_100 = float(np.mean(self.handover_ratio_history[-100:]))

                G_vals = np.array(list(self.G_u.values()), dtype=float)
                fair_str = "nan" if np.isnan(recent_fair) else f"{recent_fair:.3f}"

                print(
                    f"Slot {t + 1:6d} | "
                    f"Thr: {recent_thr:.3f} Gbps | "
                    f"Fair(JFI@100): {fair_str} | "
                    f"ON: [{ratio_str}] | "
                    f"HO(100): count={ho_count_100:.3f} "
                    f"ratio={ho_ratio_100:.4f}/{self.kappa:.4f} | "
                    f"G mean/max: {np.mean(G_vals):5.3f}/{np.max(G_vals):5.3f}"
                )

        print(f"\n{'=' * 70}")
        overall_thr = float(np.mean(self.throughput_history))
        overall_fair = (
            float(np.nanmean(self.recent_fair_list))
            if len(self.recent_fair_list) > 0 else np.nan
        )

        print(f"  Avg Throughput: {overall_thr:.3f} Gbps")
        print(f"  JFI (avg over 100 slots): {overall_fair:.4f}")

        print(f"\n  Power Budget Check:")
        for bs in self.base_stations:
            avg_power = np.mean(self.power_history[bs.bs_id])
            budget = self.P_bar[bs.bs_id]
            on_ratio = (
                sum(1 for p in self.power_history[bs.bs_id] if p > 0)
                / len(self.power_history[bs.bs_id])
            )
            print(
                f"    BS {bs.bs_id}: {avg_power:.4f}W / {budget:.4f}W | "
                f"ON={on_ratio:.3f} target={self.power_budget_ratio}"
            )

        print(f"{'=' * 70}\n")

    # =========================================================
    # Save results
    # =========================================================
    def save_results_npz(self, npz_path: str, tag: str = "MaxSNR"):
        os.makedirs(
            os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".",
            exist_ok=True
        )

        throughput = np.asarray(self.throughput_history, dtype=np.float32)
        fairness = np.asarray(self.fairness_history, dtype=np.float32)

        handover_ratio_all = np.asarray(self.handover_ratio_history, dtype=np.float32)
        handover_count_all = np.asarray(self.handover_count_history, dtype=np.float32)

        bs_ids = sorted([bs.bs_id for bs in self.base_stations])

        power_mat = []
        for bs_id in bs_ids:
            power_mat.append(np.asarray(self.power_history[bs_id], dtype=np.float32))

        power_mat = (
            np.stack(power_mat, axis=0)
            if len(power_mat) > 0
            else np.zeros((0, len(throughput)), dtype=np.float32)
        )

        if power_mat.size > 0:
            bs_on_mat = (power_mat > 0.0).astype(np.float32)
            bs_on_ratio_per_bs = bs_on_mat.mean(axis=1)
            bs_on_ratio_mean = np.asarray(
                [float(bs_on_ratio_per_bs.mean())],
                dtype=np.float32
            )

            # 0인 shutdown 구간 제외한 HO ratio 저장
            bs_on_any = np.any(bs_on_mat > 0.0, axis=0)

            if np.any(bs_on_any):
                handover_ratio = handover_ratio_all[bs_on_any]
                handover_count = handover_count_all[bs_on_any]
            else:
                handover_ratio = np.asarray([], dtype=np.float32)
                handover_count = np.asarray([], dtype=np.float32)
        else:
            bs_on_ratio_per_bs = np.asarray([], dtype=np.float32)
            bs_on_ratio_mean = np.asarray([np.nan], dtype=np.float32)
            handover_ratio = np.asarray([], dtype=np.float32)
            handover_count = np.asarray([], dtype=np.float32)

        # =========================================================
        # Eq. (11) Objective / EA-PF Utility over last window
        # J = sum_u log(avg_t R_u(t)) - lambda_E * avg_t sum_b e_b y_b(t)
        # =========================================================
        slot_rates = np.asarray(self.slot_rates, dtype=np.float32)
        eps = 1e-12
        obj_window = min(10000, len(throughput))

        if slot_rates.ndim == 2 and slot_rates.shape[0] > 0:
            recent_slot_rates = slot_rates[-obj_window:]
            avg_user_rates = np.mean(recent_slot_rates, axis=0)
            pf_utility_value = float(np.sum(np.log(avg_user_rates + eps)))
        else:
            recent_slot_rates = np.asarray([], dtype=np.float32)
            avg_user_rates = np.asarray([], dtype=np.float32)
            pf_utility_value = np.nan

        if power_mat.size > 0:
            recent_power_mat = power_mat[:, -obj_window:]
            energy_per_slot = np.sum(recent_power_mat, axis=0)
            avg_energy_cost_value = float(np.mean(energy_per_slot))
        else:
            energy_per_slot = np.asarray([], dtype=np.float32)
            avg_energy_cost_value = np.nan

        if np.isnan(pf_utility_value) or np.isnan(avg_energy_cost_value):
            ea_pf_utility_value = np.nan
        else:
            ea_pf_utility_value = (
                pf_utility_value - self.lambda_E * avg_energy_cost_value
            )

        pf_utility = np.asarray([pf_utility_value], dtype=np.float32)
        avg_energy_cost = np.asarray([avg_energy_cost_value], dtype=np.float32)
        ea_pf_utility = np.asarray([ea_pf_utility_value], dtype=np.float32)
        avg_user_rates = np.asarray(avg_user_rates, dtype=np.float32)
        energy_per_slot = np.asarray(energy_per_slot, dtype=np.float32)

        # G mean trajectory
        T = len(throughput)
        G_mean = []

        for t in range(T):
            g_vals = [
                self.queue_history["G"][ue.ue_id][t]
                for ue in self.users
            ]
            G_mean.append(float(np.mean(g_vals)))

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
            handover_count=handover_count,
            handover_budget_ratio=np.asarray([float(self.kappa)], dtype=np.float32),
            energy_budget_ratio=np.asarray([float(self.power_budget_ratio)], dtype=np.float32),
            lambda_E=np.asarray([float(self.lambda_E)], dtype=np.float32),

            G_mean=G_mean,

            slot_rates=slot_rates,
            avg_user_rates=avg_user_rates,
            energy_per_slot=energy_per_slot,
            pf_utility=pf_utility,
            avg_energy_cost=avg_energy_cost,
            ea_pf_utility=ea_pf_utility,
            performance_metric=ea_pf_utility,
        )

        print(f"✅ Saved Max-SNR results npz: {npz_path}")


if __name__ == "__main__":
    area_size = 100
    num_users = 20
    lambda_E = 20.0

    sbs_positions = generate_triangle_coverage(area_size, 35)
    # sbs_positions = generate_five_bs_coverage(area_size, 35)

    sbs_list = [
        SmallCellBaseStation(i + 1, pos, 10, 35)
        for i, pos in enumerate(sbs_positions)
    ]

    users = [
        UserEquipment(
            i + 1,
            (np.random.uniform(10, 90), np.random.uniform(10, 90))
        )
        for i in range(num_users)
    ]

    maxsnr = MaxSNRBaseline(
        sbs_list,
        users,
        power_budget_ratio=0.6,
        max_slots=50000,
        enable_mobility=True,
        enable_channel_variation=True,
        seed=0,
        hard_window_len=10000,
        lambda_E=lambda_E,
        kappa=0.1,
    )

    maxsnr.run_simulation()
    maxsnr.save_results_npz(
        f"results_compare/MaxSNR_eval_lambda_{lambda_E}.npz",
        tag=f"MaxSNR_{lambda_E}"
    )