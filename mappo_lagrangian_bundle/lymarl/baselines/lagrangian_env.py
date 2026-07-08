"""
MAPPO-Lagrangian baseline: joint UE+BS PPO with primal-dual energy constraint.

Differs from J-MARL by:
  - per-BS Lagrange multiplier mu_b (projected sub-gradient ascent, end of episode)
  - BS reward = log(rate)*I_b - penalty(mu_b)  (coefficient 1 on log R, no (over)^2 term)
  - mu_b is added to global state and per-BS observation (NOT to UE obs)

Two key formulation knobs (defaults set for stable learning, not for paper fidelity):
  - use_dimensionless=True  : C^b = on_ratio_b - rho  (drops e_bar_b scaling).
                              Reward penalty becomes -mu_b * (y_b or EMA over-ratio).
                              Why: e_bar_b ~ 0.1 W squashed the constraint signal by ~10x
                              and made eta_mu non-portable across scenarios.
  - use_ema_penalty=True    : per-step BS reward uses -mu_b * max(0, on_ratio_ema_b(t) - rho)
                              rather than -mu_b * y_b(t).
                              Why: gives continuous gradient pressure that scales with how
                              much the BS is currently over-spending, instead of a uniform
                              cost per ON slot.
  - mu_max                  : projection upper bound (avoids primal-dual runaway).  0 <= mu_b <= mu_max

See docs/additional_lagrangian_marl.md.
"""
import numpy as np
from collections import deque

from lymarl.baselines.jmarl_env import JMARLEnvironment


class LagrangianEnvironment(JMARLEnvironment):

    def __init__(
        self,
        *args,
        eta_mu: float = 0.5,
        dual_update_interval: int = 100,
        mu_max: float = 50.0,
        use_dimensionless: bool = True,
        use_ema_penalty: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Why: doc Eq. (5)/(7) — mu_b enters global state and per-BS obs (not UE obs).
        self.global_obs_dim = self.n_agents * self.n_bs + 2 * self.n_bs
        self.bs_obs_dim = 1 + 1 + self.bs_top_k

        # Why: e_bar_b is only retained for the original (paper) penalty; the dimensionless
        # formulation drops it from both C^b and the reward.
        self.e_bar_b = np.array(
            [self.P_max[bs.bs_id] for bs in self.base_stations], dtype=np.float64
        )
        self.E_bar_b = np.array(
            [self.power_budget_ratio_per_bs[bs.bs_id] * self.P_max[bs.bs_id]
             for bs in self.base_stations],
            dtype=np.float64,
        )
        self.rho_b = np.array(
            [self.power_budget_ratio_per_bs[bs.bs_id] for bs in self.base_stations],
            dtype=np.float64,
        )

        self.mu_b = np.zeros(self.n_bs, dtype=np.float64)
        self.eta_mu = float(eta_mu)
        self.dual_update_interval = int(dual_update_interval)
        self.mu_max = float(mu_max)
        self.use_dimensionless = bool(use_dimensionless)
        self.use_ema_penalty = bool(use_ema_penalty)

        self._win_on_count = np.zeros(self.n_bs, dtype=np.float64)
        self._win_steps = 0

        # Why: EMA penalty needs a per-BS recent on-ratio with the same window the user uses
        # for soft-constraint book-keeping (on_window).
        self._on_window_buf = [deque(maxlen=self.on_window) for _ in range(self.n_bs)]

        self.mu_b_history = []
        self.C_b_history = []

        print(f"[Lagrangian] eta_mu={self.eta_mu} | dual_update_interval={self.dual_update_interval} "
              f"| mu_max={self.mu_max} | dimensionless={self.use_dimensionless} "
              f"| ema_penalty={self.use_ema_penalty}")
        print(f"[Lagrangian] rho_b={self.rho_b.tolist()} | e_bar_b={self.e_bar_b.tolist()}")
        print(f"[Lagrangian] global_obs_dim={self.global_obs_dim} | bs_obs_dim={self.bs_obs_dim}")

    # ---- observations -----------------------------------------------

    def _get_global_observation(self) -> np.ndarray:
        obs = []
        for ui in range(self.n_agents):
            obs.extend(self._rate_cache[ui, :].tolist())
        for bs in self.base_stations:
            obs.append(self._rem_b(bs.bs_id))
        obs.extend(self.mu_b.tolist())
        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.global_obs_dim
        return result

    # ---- BS decision inputs ----------------------------------------

    def build_bs_decision_inputs(self, ue_actions):
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}
        for ue_id, a in ue_actions.items():
            a = int(a)
            if a == self.no_request_action or not (0 <= a < self.n_bs):
                continue
            bs_requests[self.base_stations[a].bs_id].append(ue_id)

        bs_obs_batch = np.zeros((self.n_bs, self.bs_obs_dim), dtype=np.float32)
        bs_mask_batch = np.zeros((self.n_bs, self.bs_action_dim), dtype=bool)
        cand_lists = []

        for bi, bs in enumerate(self.base_stations):
            scored = []
            for ue_id in bs_requests[bs.bs_id]:
                ui = self.ue_id_to_index[ue_id]
                r = float(self._rate_cache[ui, bi])
                if r > 0.0:
                    scored.append((r, ue_id))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: self.bs_top_k]
            cand = [uid for (_, uid) in top]
            scores = [s for (s, _) in top]
            while len(cand) < self.bs_top_k:
                cand.append(-1)
                scores.append(0.0)
            cand_lists.append(cand)

            obs = [self._rem_b(bs.bs_id), float(self.mu_b[bi])] + [float(s) for s in scores]
            bs_obs_batch[bi, :] = np.array(obs, dtype=np.float32)
            for k in range(self.bs_top_k):
                bs_mask_batch[bi, k] = cand[k] >= 0
            bs_mask_batch[bi, self.bs_top_k] = True

        return bs_obs_batch, bs_mask_batch, cand_lists

    # ---- step: parent physics, override BS reward + dual update -----

    def step_joint(self, ue_actions, bs_actions, cand_lists):
        local_obs, global_obs, info, done = super().step_joint(
            ue_actions, bs_actions, cand_lists)

        eps = self._log_eps
        served_rates = info["served_rates"]

        # Compute current on-status and update EMA windows BEFORE shaping the reward,
        # so the penalty reflects state *including* this slot.
        on_now_vec = np.zeros(self.n_bs, dtype=np.float64)
        on_ratio_ema = np.zeros(self.n_bs, dtype=np.float64)
        for bi, bs in enumerate(self.base_stations):
            sel = info["bs_selections"][bs.bs_id]
            on_now = 1.0 if sel is not None else 0.0
            on_now_vec[bi] = on_now
            self._on_window_buf[bi].append(on_now)
            on_ratio_ema[bi] = float(np.mean(self._on_window_buf[bi]))

        bs_rewards = []
        for bi, bs in enumerate(self.base_stations):
            sel = info["bs_selections"][bs.bs_id]
            rate_bs = float(served_rates[sel]) if sel is not None else 0.0
            on_now = on_now_vec[bi]

            if self.use_ema_penalty:
                # Why: pressure scales with recent over-ratio, gated by on_now so an idle BS
                # doesn't pay a phantom cost.
                over = max(0.0, on_ratio_ema[bi] - float(self.rho_b[bi]))
                penalty = float(self.mu_b[bi]) * over * on_now
            elif self.use_dimensionless:
                penalty = float(self.mu_b[bi]) * on_now
            else:
                penalty = float(self.mu_b[bi]) * float(self.e_bar_b[bi]) * on_now

            # Why: doc puts coefficient 1 on log R; tuning lives in the dual penalty (mu_b).
            r_i = float(np.log(rate_bs + eps)) * on_now - penalty
            bs_rewards.append(float(r_i))

        bs_arr = np.array(bs_rewards, dtype=np.float32)
        info["bs_rewards"] = bs_arr
        info["bs_team_reward"] = float(bs_arr.mean())

        # ---- dual update --------------------------------------------
        self._win_on_count += on_now_vec
        self._win_steps += 1
        c_b_now = None
        if self._win_steps >= self.dual_update_interval:
            avg_on = self._win_on_count / float(self._win_steps)
            if self.use_dimensionless:
                c_b_now = avg_on - self.rho_b
            else:
                c_b_now = self.e_bar_b * avg_on - self.E_bar_b
            self.mu_b = np.clip(self.mu_b + self.eta_mu * c_b_now, 0.0, self.mu_max)
            self.mu_b_history.append(self.mu_b.copy())
            self.C_b_history.append(c_b_now.copy())
            self._win_on_count[:] = 0.0
            self._win_steps = 0

        info["mu_b"] = self.mu_b.copy()
        info["aux_metrics"] = {
            "lagrangian/mu_b_mean": float(self.mu_b.mean()),
            "lagrangian/mu_b_max": float(self.mu_b.max()),
            "lagrangian/on_ratio_ema_mean": float(on_ratio_ema.mean()),
        }
        if c_b_now is not None:
            info["aux_metrics"]["lagrangian/C_b_mean"] = float(c_b_now.mean())
            info["aux_metrics"]["lagrangian/C_b_max"] = float(c_b_now.max())

        return local_obs, global_obs, info, done
