"""
J-MARL baseline: both UE+BS agents, no Lyapunov virtual queues.

UE obs   : [Rem_b for b, rate_b for b]           dim = 2*n_bs
BS obs   : [Rem_b, score_1..score_K]             dim = 1 + bs_top_k  (same size as LyMARL)
Global   : [rate_{u,b} for u,b; Rem_b for b]     dim = n_agents*n_bs + n_bs
UE team reward : mean_u log(R_{u,a_u} + eps)
BS reward      : (log(R+eps) - alpha_energy * P^tx) * I_b   (doc fixed energy penalty)
"""
import numpy as np
from lymarl.env.mappo_env import MAPPOEnvironment


class JMARLEnvironment(MAPPOEnvironment):

    def __init__(self, *args, log_eps: float = 1e-6, alpha_energy: float = 1.0, **kwargs):
        self._log_eps = float(log_eps)
        self.alpha_energy = float(alpha_energy)
        super().__init__(*args, **kwargs)
        # Override dims (super().__init__ already set n_bs / n_agents)
        self.local_obs_dim = 2 * self.n_bs
        self.global_obs_dim = self.n_agents * self.n_bs + self.n_bs
        print(f"[JMARL] local_obs_dim={self.local_obs_dim} | global_obs_dim={self.global_obs_dim}")

    def _rem_b(self, bs_id: int) -> float:
        limit = max(1, self.hard_on_limit[bs_id])
        used  = self.bs_on_used_in_window[bs_id]
        return float(max(0.0, (limit - used) / limit))

    # ---- observations -----------------------------------------------

    def _get_local_observation_by_index(self, ui: int) -> np.ndarray:
        obs = [self._rem_b(bs.bs_id) for bs in self.base_stations]
        obs.extend(self._rate_cache[ui, :].tolist())
        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.local_obs_dim
        return result

    def _get_global_observation(self) -> np.ndarray:
        obs = []
        for ui in range(self.n_agents):
            obs.extend(self._rate_cache[ui, :].tolist())
        for bs in self.base_stations:
            obs.append(self._rem_b(bs.bs_id))
        result = np.array(obs, dtype=np.float32)
        assert len(result) == self.global_obs_dim
        return result

    # ---- BS decision inputs (score = rate, no Q_u weighting) ---------

    def build_bs_decision_inputs(self, ue_actions):
        bs_requests = {bs.bs_id: [] for bs in self.base_stations}
        for ue_id, a in ue_actions.items():
            a = int(a)
            if a == self.no_request_action or not (0 <= a < self.n_bs):
                continue
            bs_requests[self.base_stations[a].bs_id].append(ue_id)

        bs_obs_batch  = np.zeros((self.n_bs, self.bs_obs_dim),   dtype=np.float32)
        bs_mask_batch = np.zeros((self.n_bs, self.bs_action_dim), dtype=bool)
        cand_lists    = []

        for bi, bs in enumerate(self.base_stations):
            scored = []
            for ue_id in bs_requests[bs.bs_id]:
                ui = self.ue_id_to_index[ue_id]
                r  = float(self._rate_cache[ui, bi])
                if r > 0.0:
                    scored.append((r, ue_id))
            scored.sort(key=lambda x: x[0], reverse=True)
            top    = scored[: self.bs_top_k]
            cand   = [uid for (_, uid) in top]
            scores = [s   for (s,   _) in top]
            while len(cand) < self.bs_top_k:
                cand.append(-1); scores.append(0.0)
            cand_lists.append(cand)

            obs = [self._rem_b(bs.bs_id)] + [float(s) for s in scores]
            bs_obs_batch[bi, :] = np.array(obs, dtype=np.float32)
            for k in range(self.bs_top_k):
                bs_mask_batch[bi, k] = cand[k] >= 0
            bs_mask_batch[bi, self.bs_top_k] = True

        return bs_obs_batch, bs_mask_batch, cand_lists

    # ---- step: parent handles physics; we override rewards -----------

    def step_joint(self, ue_actions, bs_actions, cand_lists):
        local_obs, global_obs, info, done = super().step_joint(
            ue_actions, bs_actions, cand_lists)

        served_rates = info["served_rates"]
        eps          = self._log_eps

        # UE team reward
        ue_team = float(np.mean([np.log(served_rates[u.ue_id] + eps) for u in self.users]))
        info["ue_team_reward"]     = ue_team
        info["ue_per_user_rewards"] = {
            u.ue_id: float(np.log(served_rates[u.ue_id] + eps)) for u in self.users
        }

        # BS reward
        bs_rewards = []
        for bi, bs in enumerate(self.base_stations):
            sel = info["bs_selections"][bs.bs_id]
            if sel is not None:
                rate_bs = float(served_rates[sel]); on_now = 1.0
            else:
                rate_bs = 0.0; on_now = 0.0
            p_tx = float(self.P_max[bs.bs_id])
            # Why: doc J-MARL BS reward = (log R - alpha_energy * P^tx) * 1{on}, a fixed energy penalty.
            r_i  = (float(np.log(rate_bs + eps)) - self.alpha_energy * p_tx) * on_now
            bs_rewards.append(float(r_i))

        bs_arr = np.array(bs_rewards, dtype=np.float32)
        info["bs_rewards"]    = bs_arr
        info["bs_team_reward"] = float(np.mean(bs_arr))
        return local_obs, global_obs, info, done