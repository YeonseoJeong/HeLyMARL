"""
Constrained HAPPO baselines: J-HAPPO and PF-HAPPO, no Lyapunov virtual queues.

UE obs   : [{rate_b, RemE_b} for b; RemH_u; prev_bs_u]       dim = 2*n_bs + 2
BS obs   : [RemE_b, {rate_k, RemH_k} for k]                  dim = 1 + 2*bs_top_k
Global   : [rate_{u,b} for u,b; RemE_b for b; RemH_u for u; prev_bs_u for u]
Actions  : UE selects requested BS; BS selects OFF or one UE from Top-K requested candidates

J-HAPPO reward  : sum_u log(R_u(t) + eps) - sum_b mu_E_b*cost_b(t) - sum_u nu_H_u*h_u(t)
PF-HAPPO reward : sum_u R_u(t)/(Rbar_u(t-1)+eps) - sum_b mu_E_b*cost_b(t) - sum_u nu_H_u*h_u(t)

cost_b(t)       : y_b(t) if use_dimensionless=True, else e_bar_b*y_b(t) or power_consumed_b(t)
Dual update     : mu_E_b, nu_H_u are fixed within an episode and updated only at episode end
                  using C_E_b = mean_t y_b(t) - rho, C_H_u = mean_t h_u(t) - kappa.
"""

import numpy as np
from typing import Dict, List, Optional
from HeLyMARL.env_happo import HAPPOEnvironment

class ConstrainedHAPPOEnvironment(HAPPOEnvironment):

    def __init__(
        self,
        *args,
        eta_mu: float = 0.01,
        eta_nu: float = 0.01,
        mu_max: float = 100.0,
        nu_max: float = 100.0,
        use_dimensionless: bool = True,
        pf_avg_beta: float = 0.99,
        episode_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.global_obs_dim = self.n_agents * self.n_bs + self.n_bs + 2 * self.n_agents
        self.local_obs_dim = 2 * self.n_bs + 2
        self.bs_obs_dim = 1 + 2 * self.bs_top_k

        self.eta_mu = float(eta_mu)
        self.eta_nu = float(eta_nu)
        self.mu_max = float(mu_max)
        self.nu_max = float(nu_max)

        self.rate_eps = 1e-6
        self.episode_length = int(episode_length) if episode_length is not None else int(self.hard_window_len)
        self.use_dimensionless = bool(use_dimensionless)
        self.pf_avg_beta = float(pf_avg_beta)

        # Dual variables
        self.mu_E_b = {bs.bs_id: 0.0 for bs in self.base_stations}
        self.nu_H_u = {u.ue_id: 0.0 for u in self.users}

        # Episode-level constraint statistics
        self.episode_on_hist = {
            bs.bs_id: []
            for bs in self.base_stations
        }

        self.episode_ho_hist = {
            u.ue_id: []
            for u in self.users
        }

        # PF running average rate, Rbar_u(t-1)
        self.avg_rate_u = {
            u.ue_id: 1.0
            for u in self.users
        }

        self.episode_idx = 0

    def reset(self):
        obs = super().reset()

        self.episode_on_hist = {
            bs.bs_id: []
            for bs in self.base_stations
        }

        self.episode_ho_hist = {
            u.ue_id: []
            for u in self.users
        }

        self.avg_rate_u = {
            u.ue_id: 1.0
            for u in self.users
        }

        return obs
        
    def reset_dual_variables(self):
        self.mu_E_b = {bs.bs_id: 0.0 for bs in self.base_stations}
        self.nu_H_u = {u.ue_id: 0.0 for u in self.users}
    
    def _compute_utility(self, served_rates: Dict[int, float]) -> float:
        """
        Child classes override this.
        """
        raise NotImplementedError

    
    def _bs_candidate_score(self, ue_id: int, rate: float) -> float:
        """
        Default score for Jensen-HAPPO.
        PF-HAPPO overrides this.
        """
        return float(rate)
    
    def _remaining_handover_ratio(self, ue_id: int) -> float:
        hist = self.episode_ho_hist[ue_id]
        if len(hist) == 0:
            return 1.0
        used_ho = float(np.sum(hist))
        max_ho = max(1.0, self.kappa * self.episode_length)

        return float(np.clip((max_ho - used_ho) / max_ho, 0.0, 1.0))

    def _remaining_energy_ratio(self, bs_id: int) -> float:
        hist = self.episode_on_hist[bs_id]
        if len(hist) == 0:
            return 1.0
        used_on = float(np.sum(hist))
        max_on = max(1.0, self.power_budget_ratio * self.episode_length)

        return float(np.clip((max_on - used_on) / max_on, 0.0, 1.0))
    
    def _get_local_observation_by_index(self, ui: int) -> np.ndarray:
        ue = self.users[ui]
        ue_id = ue.ue_id

        obs = []
        for bi, bs in enumerate(self.base_stations):
            rate = float(self._rate_cache[ui, bi])
            rem_e = float(self._remaining_energy_ratio(bs.bs_id))
            obs.extend([rate, rem_e])

        obs.append(float(self._remaining_handover_ratio(ue_id)))
        obs.append(float(self.m_u.get(ue_id, 0)))
        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.local_obs_dim, (
            f"local_obs_dim mismatch: got {len(result)}, expected {self.local_obs_dim}"
        )
        return result
    
    def _get_global_observation(self) -> np.ndarray:
        obs = []

        # 1) rate_{u,b}
        for u in self.users:
            ui = self.ue_id_to_index[u.ue_id]
            obs.extend(self._rate_cache[ui, :].tolist())

        # 2) RemE_b
        for bs in self.base_stations:
            obs.append(float(self._remaining_energy_ratio(bs.bs_id)))

        # 3) RemH_u
        for u in self.users:
            obs.append(float(self._remaining_handover_ratio(u.ue_id)))

        # 4) prev_bs_u / serving memory
        for u in self.users:
            ue_id = u.ue_id
            obs.append(float(self.m_u.get(ue_id, 0)))

        result = np.array(obs, dtype=np.float32)

        assert len(result) == self.global_obs_dim, (
            f"global_obs_dim mismatch: got {len(result)}, expected {self.global_obs_dim}"
        )
        return result
    
    def build_bs_decision_inputs(self, ue_actions):
        """
        Build BS observations for J-HAPPO / PF-HAPPO.

        BS obs:
        [RemE_b, rate_1, RemH_1, ..., rate_K, RemH_K]

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
            bs_id = bs.bs_id
            reqs = bs_requests[bs_id]

            scored = []
            for ue_id in reqs:
                ui = self.ue_id_to_index[ue_id]
                rate = float(self._rate_cache[ui, bi])

                if rate <= 0.0:
                    continue

                score = self._bs_candidate_score(ue_id, rate)
                scored.append((float(score), ue_id, rate))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:self.bs_top_k]

            cand = []
            feat_pairs = []

            for score, ue_id, rate in top:
                cand.append(ue_id)
                rem_h = self._remaining_handover_ratio(ue_id)
                feat_pairs.extend([
                    float(rate),
                    float(rem_h),
                ])

            while len(cand) < self.bs_top_k:
                cand.append(-1)
                feat_pairs.extend([0.0, 0.0])

            cand_lists.append(cand)

            rem_e = self._remaining_energy_ratio(bs_id)
            obs = [float(rem_e)]
            obs.extend(feat_pairs)
            bs_obs_batch[bi, :] = np.array(obs, dtype=np.float32)

            bs_mask_batch[bi, 0] = True
            for k in range(self.bs_top_k):
                bs_mask_batch[bi, k + 1] = cand[k] >= 0

        return bs_obs_batch, bs_mask_batch, cand_lists

    def step_joint(self, ue_actions, bs_actions, cand_lists):
        local_obs, global_obs, info, done = super().step_joint(
            ue_actions,
            bs_actions,
            cand_lists,
        )

        served_rates = info["served_rates"]
        power_consumed = info["power_consumed"]
        handover_u = info["handover_u"]

        # 1. episode-level cost 기록
        for bs in self.base_stations:
            bs_id = bs.bs_id
            y_b = float(power_consumed[bs_id] > 0.0)
            self.episode_on_hist[bs_id].append(y_b)

        for u in self.users:
            ue_id = u.ue_id
            self.episode_ho_hist[ue_id].append(float(handover_u[ue_id]))

        # 2. old dual variables 기록
        old_mu_E_b = self.mu_E_b.copy()
        old_nu_H_u = self.nu_H_u.copy()

        # 3. utility 계산
        utility = self._compute_utility(served_rates)

        # 4. Lagrangian penalty 계산
        if self.use_dimensionless:
            energy_penalty = sum(
                old_mu_E_b[bs.bs_id] * float(power_consumed[bs.bs_id] > 0.0)
                for bs in self.base_stations
            )
        else:
            energy_penalty = sum(
                old_mu_E_b[bs.bs_id] * float(power_consumed[bs.bs_id])
                for bs in self.base_stations
            )

        handover_penalty = sum(
            old_nu_H_u[u.ue_id] * float(handover_u[u.ue_id])
            for u in self.users
        )

        global_reward = float(utility - energy_penalty - handover_penalty)

        # 5. PF running average rate 업데이트
        for u in self.users:
            ue_id = u.ue_id
            r = float(served_rates[ue_id])
            self.avg_rate_u[ue_id] = (
                self.pf_avg_beta * self.avg_rate_u[ue_id]
                + (1.0 - self.pf_avg_beta) * r
            )

        # 6. episode 끝에서만 dual update   
        episode_done = done or ((self.timestep % self.episode_length) == 0)
        if episode_done:
            self._update_dual_variables_episode()

        info["global_reward"] = global_reward
        info["ue_team_reward"] = global_reward
        info["bs_team_reward"] = global_reward

        info["utility"] = float(utility)
        info["energy_penalty"] = float(energy_penalty)
        info["handover_penalty"] = float(handover_penalty)

        info["mu_E_b"] = self.mu_E_b.copy()
        info["nu_H_u"] = self.nu_H_u.copy()

        info["old_mu_E_b"] = old_mu_E_b
        info["old_nu_H_u"] = old_nu_H_u

        info["episode_done"] = bool(episode_done)

        return local_obs, global_obs, info, done
    
    def _update_dual_variables_episode(self):
        beta_mu_k = self.eta_mu / np.sqrt(self.episode_idx + 1)
        beta_nu_k = self.eta_nu / np.sqrt(self.episode_idx + 1)

        for bs in self.base_stations:
            bs_id = bs.bs_id

            if self.use_dimensionless:
                if len(self.episode_on_hist[bs_id]) == 0:
                    avg_cost = 0.0
                else:
                    avg_cost = float(np.mean(self.episode_on_hist[bs_id]))

                budget = float(self.power_budget_ratio)
            else:
                if len(self.episode_on_hist[bs_id]) == 0:
                    avg_cost = 0.0
                else:
                    avg_cost = float(np.mean(self.episode_on_hist[bs_id])) * float(self.P_max[bs_id])

                budget = float(self.P_bar[bs_id])

            C_E = avg_cost - budget

            self.mu_E_b[bs_id] = float(np.clip(
                self.mu_E_b[bs_id] + beta_mu_k * C_E,
                0.0,
                self.mu_max,
            ))

        for u in self.users:
            ue_id = u.ue_id

            if len(self.episode_ho_hist[ue_id]) == 0:
                ho_ratio = 0.0
            else:
                ho_ratio = float(np.mean(self.episode_ho_hist[ue_id]))

            C_H = ho_ratio - self.kappa

            self.nu_H_u[ue_id] = float(np.clip(
                self.nu_H_u[ue_id] + beta_nu_k * C_H,
                0.0,
                self.nu_max,
            ))

        self.episode_idx += 1

        self.episode_on_hist = {
            bs.bs_id: []
            for bs in self.base_stations
        }

        self.episode_ho_hist = {
            u.ue_id: []
            for u in self.users
        }
            

class JensenHAPPOEnvironment(ConstrainedHAPPOEnvironment):
    """
    Jensen-HAPPO baseline.

    Utility:
        U(t) = mean_u log(1 + R_u(t))

    This is a concave utility over rates.
    """

    def _compute_utility(self, served_rates: Dict[int, float]) -> float:
        """
        J-HAPPO reward:
            r_t = sum_u log(R_u(t) + eps)
        """
        utility = 0.0

        for u in self.users:
            ue_id = u.ue_id
            r = float(served_rates[ue_id])
            utility += np.log(r + self.rate_eps)

        return float(utility)


class PFHAPPOEnvironment(ConstrainedHAPPOEnvironment):
    """
    PF-HAPPO reward:
        r_t = sum_u R_u(t) / (Rbar_u(t-1) + eps)
    """

    def __init__(self, *args, pf_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.pf_eps = float(pf_eps)

    def _compute_utility(self, served_rates: Dict[int, float]) -> float:
        utility = 0.0

        for u in self.users:
            ue_id = u.ue_id

            r = float(served_rates[ue_id])
            avg_r_prev = float(self.avg_rate_u[ue_id])

            utility += r / (avg_r_prev + self.pf_eps)

        return float(utility)
    
    def _bs_candidate_score(self, ue_id: int, rate: float) -> float:
        avg_r = float(self.avg_rate_u[ue_id])
        return float(rate / (avg_r + self.pf_eps))

    