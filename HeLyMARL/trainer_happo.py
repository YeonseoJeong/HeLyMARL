import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
from typing import Dict, Optional

from HeLyMARL.networks_happo import (
    UEActorNetwork,
    BSActorNetwork,
    CentralizedCritic,
    ValueNorm
)
from HeLyMARL.utils_happo import moving_avg, block_avg_1d, set_seed


class HAPPOTrainer:
    def __init__(
        self,
        env,
        eval_env=None,
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
        self.eval_env = eval_env
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

        # Critics
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
        env=None,
        deterministic: bool = False,
    ):
        run_env = self.env if env is None else env
        users = run_env.users

        global_t = torch.as_tensor(
            global_obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        v_n = self.critic(global_t).squeeze(0)

        # ==================================================
        # UE actions
        # ==================================================
        obs_batch = np.stack(
            [
                local_obs[u.ue_id]
                for u in users
            ],
            axis=0,
        ).astype(np.float32)

        ue_mask_batch = np.stack(
            [
                run_env._get_action_mask(u.ue_id)
                for u in users
            ],
            axis=0,
        ).astype(bool)

        obs_t = torch.as_tensor(
            obs_batch,
            dtype=torch.float32,
            device=self.device,
        )

        ue_mask_t = torch.as_tensor(
            ue_mask_batch,
            dtype=torch.bool,
            device=self.device,
        )

        ue_logits = self.ue_actor(obs_t).masked_fill(
            ~ue_mask_t,
            float("-inf"),
        )

        ue_dist = Categorical(logits=ue_logits)

        if deterministic:
            ue_actions_t = torch.argmax(
                ue_logits,
                dim=-1,
            )
        else:
            ue_actions_t = ue_dist.sample()

        ue_logp_t = ue_dist.log_prob(ue_actions_t)
        ue_ent_t = ue_dist.entropy()

        ue_actions = {
            u.ue_id: int(ue_actions_t[i].item())
            for i, u in enumerate(users)
        }

        # ==================================================
        # BS actions
        # ==================================================
        bs_obs_batch, bs_mask_batch, cand_lists = (
            run_env.build_bs_decision_inputs(ue_actions)
        )

        bs_obs_t = torch.as_tensor(
            bs_obs_batch,
            dtype=torch.float32,
            device=self.device,
        )

        bs_mask_t = torch.as_tensor(
            bs_mask_batch,
            dtype=torch.bool,
            device=self.device,
        )

        bs_logits = self.bs_actor(bs_obs_t).masked_fill(
            ~bs_mask_t,
            float("-inf"),
        )

        bs_dist = Categorical(logits=bs_logits)

        if deterministic:
            bs_actions_t = torch.argmax(
                bs_logits,
                dim=-1,
            )
        else:
            bs_actions_t = bs_dist.sample()

        bs_logp_t = bs_dist.log_prob(bs_actions_t)
        bs_ent_t = bs_dist.entropy()

        bs_actions = {
            bs.bs_id: int(bs_actions_t[i].item())
            for i, bs in enumerate(run_env.base_stations)
        }

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
    
    @torch.no_grad()
    def evaluate_current_policy_objective(
        self,
        n_eval_episodes: int,
        steps_per_episode: int,
        objective_eps: float = 1e-12,
        eval_seeds=None,
        deterministic: bool = False,
    ):
        if self.eval_env is None:
            raise ValueError(
                "eval_env가 설정되지 않았습니다. "
                "Training env와 별도의 evaluation env를 전달해야 합니다."
            )

        eval_env = self.eval_env

        # ------------------------------------------------------
        # 모든 checkpoint에서 동일한 evaluation seed 사용
        # ------------------------------------------------------
        if eval_seeds is None:
            eval_seeds = [
                1000 + i
                for i in range(n_eval_episodes)
            ]

        eval_seeds = list(eval_seeds)

        if len(eval_seeds) != n_eval_episodes:
            raise ValueError(
                f"n_eval_episodes={n_eval_episodes}인데 "
                f"eval_seeds 길이는 {len(eval_seeds)}입니다."
            )

        # ------------------------------------------------------
        # Evaluation이 training의 RNG 흐름을 바꾸지 않도록 저장
        # ------------------------------------------------------
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()

        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()
        else:
            cuda_rng_state = None

        # 기존 network mode 저장
        ue_was_training = self.ue_actor.training
        bs_was_training = self.bs_actor.training
        critic_was_training = self.critic.training

        objective_values = []

        try:
            self.ue_actor.eval()
            self.bs_actor.eval()
            self.critic.eval()

            for eval_ep, eval_seed in enumerate(eval_seeds):
                # 각 checkpoint마다 항상 동일한 seed 집합 사용
                set_seed(int(eval_seed))

                local_obs, global_obs = eval_env.reset()

                ep_slot_rates = []

                for eval_step in range(steps_per_episode):
                    (
                        ue_actions,
                        _ue_logp_np,
                        _ue_ent_np,
                        _ue_masks_np,
                        bs_actions,
                        _bs_logp_np,
                        _bs_ent_np,
                        _bs_obs_np,
                        _bs_masks_np,
                        cand_lists,
                        _v_n,
                    ) = self.select_actions(
                        local_obs=local_obs,
                        global_obs=global_obs,
                        env=eval_env,

                        # stochastic action sampling
                        deterministic=deterministic,
                    )

                    (
                        next_local_obs,
                        next_global_obs,
                        info,
                        done,
                    ) = eval_env.step_joint(
                        ue_actions=ue_actions,
                        bs_actions=bs_actions,
                        cand_lists=cand_lists,
                    )

                    rates_this_slot = np.asarray(
                        [
                            float(
                                info["served_rates"][u.ue_id]
                            )
                            for u in eval_env.users
                        ],
                        dtype=np.float64,
                    )

                    ep_slot_rates.append(rates_this_slot)

                    local_obs = next_local_obs
                    global_obs = next_global_obs

                    if done:
                        break

                ep_slot_rates = np.asarray(
                    ep_slot_rates,
                    dtype=np.float64,
                )

                if (
                    ep_slot_rates.ndim != 2
                    or ep_slot_rates.shape[0] == 0
                ):
                    objective = np.nan

                else:
                    avg_user_rates = np.mean(
                        ep_slot_rates,
                        axis=0,
                    )

                    objective = float(
                        np.sum(
                            np.log(
                                np.maximum(
                                    avg_user_rates,
                                    objective_eps,
                                )
                            )
                        )
                    )

                objective_values.append(objective)

            objective_values = np.asarray(
                objective_values,
                dtype=np.float64,
            )

            finite_values = objective_values[
                np.isfinite(objective_values)
            ]

            if finite_values.size > 0:
                objective_mean = float(
                    np.mean(finite_values)
                )

                objective_std = float(
                    np.std(finite_values)
                )

            else:
                objective_mean = np.nan
                objective_std = np.nan

        finally:
            # --------------------------------------------------
            # Evaluation 전 training RNG 상태로 복원
            # --------------------------------------------------
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)

            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(
                    cuda_rng_state
                )

            # 기존 network mode 복원
            self.ue_actor.train(ue_was_training)
            self.bs_actor.train(bs_was_training)
            self.critic.train(critic_was_training)

        return {
            "objective_values": objective_values,
            "objective_mean": objective_mean,
            "objective_std": objective_std,
            "eval_seeds": np.asarray(
                eval_seeds,
                dtype=np.int32,
            ),
        }
    
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
        global_obs = torch.tensor(np.stack(self.rb["global_obs"], axis=0), dtype=torch.float32, device=self.device)
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
        ue_local_obs = torch.tensor(np.stack(self.rb["local_obs"], axis=0), dtype=torch.float32, device=self.device)
        ue_masks = torch.tensor(np.stack(self.rb["ue_masks"], axis=0), dtype=torch.bool, device=self.device)
        ue_actions = torch.tensor(np.stack(self.rb["ue_actions"], axis=0), dtype=torch.long, device=self.device)
        ue_old_logp = torch.tensor(np.stack(self.rb["ue_logp"], axis=0), dtype=torch.float32, device=self.device)

        ue_local_f = ue_local_obs.reshape(T * N, -1)
        ue_masks_f = ue_masks.reshape(T * N, -1)
        ue_actions_f = ue_actions.reshape(T * N)
        ue_old_logp_f = ue_old_logp.reshape(T * N)
        ue_adv_f = adv.repeat_interleave(N)

        # BS tensors
        bs_obs = torch.tensor(np.stack(self.rb["bs_obs"], axis=0), dtype=torch.float32, device=self.device)
        bs_masks = torch.tensor(np.stack(self.rb["bs_masks"], axis=0), dtype=torch.bool, device=self.device)
        bs_actions = torch.tensor(np.stack(self.rb["bs_actions"], axis=0), dtype=torch.long, device=self.device)
        bs_old_logp = torch.tensor(np.stack(self.rb["bs_logp"], axis=0), dtype=torch.float32, device=self.device)

        bs_obs_f = bs_obs.reshape(T * B, -1)
        bs_masks_f = bs_masks.reshape(T * B, -1)
        bs_actions_f = bs_actions.reshape(T * B)
        bs_old_logp_f = bs_old_logp.reshape(T * B)
        bs_adv_f = adv.repeat_interleave(B)

        losses = {
            "critic": 0.0, 
            "actor_ue": 0.0, "actor_bs": 0.0,
            "entropy_ue": 0.0, "entropy_bs": 0.0
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

            # happo correction term
            # with torch.no_grad():
            #     updated_ue_logits = self.ue_actor(ue_local_f).masked_fill(~ue_masks_f, float("-inf"))
            #     updated_ue_dist = Categorical(logits=updated_ue_logits)
            #     updated_ue_logp = updated_ue_dist.log_prob(ue_actions_f).reshape(T, N)
            #     old_ue_logp = ue_old_logp.reshape(T, N)
            #     ue_log_ratio_per_agent = (updated_ue_logp - old_ue_logp)
            #     ue_group_log_ratio = (ue_log_ratio_per_agent.mean(dim=1))
            #     ue_correction=torch.exp(ue_group_log_ratio)
            #     M_ue_t = (ue_correction * adv).detach()
            #     bs_M_f = M_ue_t.repeat_interleave(B)

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
                # surr1 = ratio * bs_M_f[mb]  # happo correction term
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * bs_adv_f[mb]
                # surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * bs_M_f[mb]  # happo correction term
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
    def train(self,
        n_episodes: int,
        steps_per_episode: int,
        update_interval: int = 128,
        save_npz_path: Optional[str] = None,
        eval_every: int = 1,
        eval_n_episodes: int = 3,
        eval_steps_per_episode: Optional[int] = None,
        eval_seeds: Optional[list] = None,
        eval_deterministic: bool = False,
        policy_improvement_dir: Optional[str] = None,
        checkpoint_every_updates_early: int = 8,
        checkpoint_every_updates_mid: int = 40,
        checkpoint_every_updates_late: int = 80,
        checkpoint_early_until_step: int = 10000,
        checkpoint_mid_until_step: int = 50000,
        save_episode_end_checkpoint: bool = True,
    ):
        if eval_steps_per_episode is None:
            eval_steps_per_episode = steps_per_episode

        print(f"\n{'='*100}")
        print(" HAPPO Training - Episodic")
        print(f"{'='*100}")
        print(f"Total train episodes: {n_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Total train steps: {n_episodes * steps_per_episode}")
        print(f"Update interval: {update_interval}")
        print(f"Hard constraint during training: {self.env.use_hard_constraint}")
        print(f"{'='*100}\n")

        # --------------------------------------------------
        # Minimal training logs
        # --------------------------------------------------
        update_step_history = []
        update_episode_history = []
        handover_ratio_history = []
        episode_handover_ratio_history = []
        episode_handover_gap_history = []

        critic_loss_history = []
        actor_ue_loss_history = []
        actor_bs_loss_history = []
        entropy_ue_history = []
        entropy_bs_history = []

        # reward / queue logs 정도만 training에서 확인
        global_reward_history = []
        Q_mean_history = []
        Z_mean_history = []
        G_mean_history = []
        G_max_history = []

        mu_E_mean_history = []
        nu_H_mean_history = []
        RemE_mean_history = []
        RemH_mean_history = []
        C_E_mean_history = []
        C_H_mean_history = []

        # optional: episode-level summaries
        episode_reward_history = []
        
        # Episode 종료 시 최종 정책 평가
        eval_episode_idx_history = []
        eval_objective_mean_history = []
        eval_objective_std_history = []

        # Policy-improvement checkpoints: initial policy + intermediate policies
        policy_eval_step_history = []
        policy_eval_episode_history = []
        policy_eval_update_history = []
        policy_eval_objective_mean_history = []
        policy_eval_objective_std_history = []
        policy_eval_objective_values_history = []
        policy_checkpoint_path_history = []

        episode_Q_last_history = []
        episode_Z_last_history = []
        episode_G_last_history = []

        episode_mu_E_last_history = []
        episode_nu_H_last_history = []
        episode_RemE_last_history = []
        episode_RemH_last_history = []
        episode_C_E_last_history = []
        episode_C_H_last_history = []

        global_step = 0
        update_count = 0
        saved_checkpoint_steps = set()

        if policy_improvement_dir is not None:
            os.makedirs(policy_improvement_dir, exist_ok=True)

        def _checkpoint_interval_updates(step: int) -> int:
            if step <= checkpoint_early_until_step:
                return max(1, int(checkpoint_every_updates_early))
            if step <= checkpoint_mid_until_step:
                return max(1, int(checkpoint_every_updates_mid))
            return max(1, int(checkpoint_every_updates_late))

        def _save_and_evaluate_policy_checkpoint(
            step: int,
            episode_idx: int,
            current_update_count: int,
            label: str,
        ):
            if step in saved_checkpoint_steps:
                return

            checkpoint_path = ""
            if policy_improvement_dir is not None:
                checkpoint_path = os.path.join(
                    policy_improvement_dir,
                    f"checkpoint_step_{step:07d}_{label}.pt",
                )
                self.save_model(checkpoint_path, save_optim=False)

            if self.eval_env is not None:
                eval_result = self.evaluate_current_policy_objective(
                    n_eval_episodes=eval_n_episodes,
                    steps_per_episode=eval_steps_per_episode,
                    eval_seeds=eval_seeds,
                    deterministic=eval_deterministic,
                )

                policy_eval_step_history.append(int(step))
                policy_eval_episode_history.append(int(episode_idx))
                policy_eval_update_history.append(int(current_update_count))
                policy_eval_objective_mean_history.append(
                    float(eval_result["objective_mean"])
                )
                policy_eval_objective_std_history.append(
                    float(eval_result["objective_std"])
                )
                policy_eval_objective_values_history.append(
                    np.asarray(
                        eval_result["objective_values"],
                        dtype=np.float64,
                    )
                )
                policy_checkpoint_path_history.append(checkpoint_path)

                print(
                    f"[POLICY CHECKPOINT] Step {step:7d} | "
                    f"Episode {episode_idx:4d} | "
                    f"Update {current_update_count:5d} | "
                    f"Objective {eval_result['objective_mean']:.6f} "
                    f"± {eval_result['objective_std']:.6f}"
                )

            saved_checkpoint_steps.add(int(step))

        # Step 0: policy before any PPO/HAPPO update
        _save_and_evaluate_policy_checkpoint(
            step=0,
            episode_idx=0,
            current_update_count=0,
            label="initial",
        )

        for ep in range(n_episodes):
            # --------------------------------------------------
            # Episode reset
            # 여기서 env.reset()은 현재 코드 기준으로 topology까지 reset함.
            # topology 유지하고 싶으면 env 쪽 reset_episode(reset_topology=False)를 따로 만드는 게 좋음.
            # --------------------------------------------------
            local_obs, global_obs = self.env.reset()
            self.reset_rollout()

            ep_reward_sum = 0.0
            # Objective calculation for this episode
            ep_slot_rates = []
            ep_energy_cost = []

            for ep_step in range(steps_per_episode):
                global_step += 1

                (ue_actions, ue_logp_np, ue_ent_np, ue_masks_np,
                bs_actions, bs_logp_np, bs_ent_np, bs_obs_np, bs_masks_np, cand_lists, v_n) = self.select_actions(local_obs, global_obs)

                next_local_obs, next_global_obs, info, done = self.env.step_joint(
                    ue_actions=ue_actions,
                    bs_actions=bs_actions,
                    cand_lists=cand_lists
                )

                # --------------------------------------------------
                # Store per-user served rates for episode objective
                # --------------------------------------------------
                rates_this_slot = np.asarray(
                    [
                        float(info["served_rates"][u.ue_id])
                        for u in self.env.users
                    ],
                    dtype=np.float64,
                )

                ep_slot_rates.append(rates_this_slot)

                # Energy cost at this slot
                power_consumed = info.get("power_consumed", {})

                if isinstance(power_consumed, dict):
                    slot_energy_cost = float(
                        np.sum([
                            float(power_consumed[bs.bs_id])
                            for bs in self.env.base_stations
                        ])
                    )
                else:
                    power_array = np.asarray(
                        power_consumed,
                        dtype=np.float64,
                    ).reshape(-1)

                    slot_energy_cost = (
                        float(np.sum(power_array))
                        if power_array.size > 0
                        else 0.0
                    )

                ep_energy_cost.append(slot_energy_cost)

                handover_u = info.get("handover_u", {})

                if isinstance(handover_u, dict):
                    slot_ho_ratio = float(
                        np.mean([
                            float(handover_u[u.ue_id])
                            for u in self.env.users
                        ])
                    )
                else:
                    handover_arr = np.asarray(
                        handover_u,
                        dtype=np.float32,
                    ).reshape(-1)

                    slot_ho_ratio = (
                        float(np.mean(handover_arr))
                        if handover_arr.size > 0
                        else 0.0
                    )

                handover_ratio_history.append(slot_ho_ratio)

                reward = float(info["global_reward"])
                ep_reward_sum += reward
                global_reward_history.append(reward)

                if self._is_constrained_env():
                    cstats = self._get_constrained_log_stats()

                    mu_E_mean_history.append(cstats["mu_E_mean"])
                    nu_H_mean_history.append(cstats["nu_H_mean"])
                    RemE_mean_history.append(cstats["RemE_mean"])
                    RemH_mean_history.append(cstats["RemH_mean"])
                    C_E_mean_history.append(cstats["C_E_mean"])
                    C_H_mean_history.append(cstats["C_H_mean"])
                else:
                    Q_vals = list(info["Q_u"].values())
                    Z_vals = list(info["Z_b"].values())
                    G_vals = list(info["G_u"].values())

                    Q_mean_history.append(np.mean(Q_vals))
                    Z_mean_history.append(np.mean(Z_vals))
                    G_mean_history.append(np.mean(G_vals))
                    G_max_history.append(np.max(G_vals))

                # episode 마지막에서만 done=True
                done_to_store = bool(done or (ep_step == steps_per_episode - 1))

                with torch.no_grad():
                    next_global_t = torch.as_tensor(next_global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    nv_n = self.critic(next_global_t).squeeze(0).detach().cpu().numpy().astype(np.float32)

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
                    done=done_to_store
                )
                
                local_obs, global_obs = next_local_obs, next_global_obs

                # --------------------------------------------------
                # PPO/HAPPO update inside episode
                # - Q/Z/G는 reset하지 않음
                # - rollout buffer만 update() 안에서 reset됨
                # --------------------------------------------------
                should_update = ((ep_step + 1) % update_interval == 0) or done_to_store
                
                if should_update:
                    losses = self.update()

                    if losses:
                        update_step_history.append(global_step)
                        update_episode_history.append(ep + 1)
                        critic_loss_history.append(float(losses["critic"]))
                        actor_ue_loss_history.append(float(losses["actor_ue"]))
                        actor_bs_loss_history.append(float(losses["actor_bs"]))
                        entropy_ue_history.append(float(losses["entropy_ue"]))
                        entropy_bs_history.append(float(losses["entropy_bs"]))

                        update_count += 1

                        print(
                            f"[UPDATE] Ep {ep+1:4d} | "
                            f"EpStep {ep_step+1:5d}/{steps_per_episode} | "
                            f"GlobalStep {global_step:7d} | "
                            f"UE_Actor:{losses['actor_ue']:.4f} | "
                            f"BS_Actor:{losses['actor_bs']:.4f} | "
                            f"Critic:{losses['critic']:.4f} | "
                            f"Ent(UE):{losses['entropy_ue']:.4f} | "
                            f"Ent(BS):{losses['entropy_bs']:.4f}"
                        )

                        interval_updates = _checkpoint_interval_updates(global_step)
                        if update_count % interval_updates == 0:
                            _save_and_evaluate_policy_checkpoint(
                                step=global_step,
                                episode_idx=ep + 1,
                                current_update_count=update_count,
                                label="intermediate",
                            )

                # --------------------------------------------------
                # Light progress logging
                # 통신 성능은 evaluate에서 보면 되니까 여기서는 queue/reward만 확인
                # --------------------------------------------------
                if global_step % 1000 == 0:
                    recent_reward = float(np.mean(global_reward_history[-100:]))
                    if self._is_constrained_env():
                        print(
                            f"Ep {ep+1:4d} | "
                            f"GlobalStep {global_step:7d} | "
                            f"Reward(1k): {recent_reward:.3f} | "
                            f"muE:{mu_E_mean_history[-1]:.3f} | "
                            f"nuH:{nu_H_mean_history[-1]:.3f} | "
                            f"RemE:{RemE_mean_history[-1]:.3f} | "
                            f"RemH:{RemH_mean_history[-1]:.3f} | "
                            f"CE:{C_E_mean_history[-1]:.4f} | "
                            f"CH:{C_H_mean_history[-1]:.4f}"
                        )
                    else:
                        print(
                            f"Ep {ep+1:4d} | "
                            f"GlobalStep {global_step:7d} | "
                            f"Reward(1k): {recent_reward:.3f} | "
                            f"Qmean:{Q_mean_history[-1]:.3f} | "
                            f"Zmean:{Z_mean_history[-1]:.3f} | "
                            f"Gmean:{G_mean_history[-1]:.3f} | "
                            f"Gmax:{G_max_history[-1]:.3f}"
                        )

            # --------------------------------------------------
            # Episode summary
            # 여기서 env.reset()은 아직 호출하지 않음.
            # 다음 episode 시작에서 reset됨.
            # --------------------------------------------------
            episode_reward_history.append(
                float(
                    ep_reward_sum
                    / max(1, steps_per_episode)
                )
            )

            episode_start = ep * steps_per_episode
            episode_end = len(handover_ratio_history)

            episode_ho_slots = np.asarray(
                handover_ratio_history[
                    episode_start:episode_end
                ],
                dtype=np.float64,
            )
            # 첫 slot의 HO가 항상 0이므로,
            # 논문의 T-1 분모에 맞추려면 첫 slot 제외
            if episode_ho_slots.size > 1:
                episode_ho_ratio = float(
                    np.sum(episode_ho_slots[1:])
                    / (episode_ho_slots.size - 1)
                )
            elif episode_ho_slots.size == 1:
                episode_ho_ratio = 0.0
            else:
                episode_ho_ratio = np.nan

            episode_ho_gap = (
                episode_ho_ratio
                - float(self.env.kappa)
            )

            episode_handover_ratio_history.append(
                episode_ho_ratio
            )

            episode_handover_gap_history.append(
                episode_ho_gap
            )    

            if self._is_constrained_env():
                episode_mu_E_last_history.append(float(mu_E_mean_history[-1]))
                episode_nu_H_last_history.append(float(nu_H_mean_history[-1]))
                episode_RemE_last_history.append(float(RemE_mean_history[-1]))
                episode_RemH_last_history.append(float(RemH_mean_history[-1]))
                episode_C_E_last_history.append(float(C_E_mean_history[-1]))
                episode_C_H_last_history.append(float(C_H_mean_history[-1]))

                print(
                    f"[EP END] Ep {ep+1:4d} | "
                    f"AvgReward:{episode_reward_history[-1]:.3f} | "
                    f"muE:{episode_mu_E_last_history[-1]:.3f} | "
                    f"nuH:{episode_nu_H_last_history[-1]:.3f} | "
                    f"RemE:{episode_RemE_last_history[-1]:.3f} | "
                    f"RemH:{episode_RemH_last_history[-1]:.3f} | "
                    f"CE:{episode_C_E_last_history[-1]:.4f} | "
                    f"CH:{episode_C_H_last_history[-1]:.4f}"
                )
            
            else:
                episode_Q_last_history.append(float(Q_mean_history[-1]))
                episode_Z_last_history.append(float(Z_mean_history[-1]))
                episode_G_last_history.append(float(G_mean_history[-1]))

                print(
                    f"[EP END] Ep {ep+1:4d} | "
                    f"AvgReward:{episode_reward_history[-1]:.3f} | "
                    f"HO:{episode_handover_ratio_history[-1]:.5f} | "
                    f"Kappa:{float(self.env.kappa):.5f} | "
                    f"HOGap:{episode_handover_gap_history[-1]:+.5f} | "
                    f"Last Qmean:{episode_Q_last_history[-1]:.3f} | "
                    f"Last Zmean:{episode_Z_last_history[-1]:.3f} | "
                    f"Last Gmean:{episode_G_last_history[-1]:.3f}"
                )       

            if save_episode_end_checkpoint:
                _save_and_evaluate_policy_checkpoint(
                    step=global_step,
                    episode_idx=ep + 1,
                    current_update_count=update_count,
                    label=f"episode_{ep + 1:04d}_end",
                )

            # ==================================================
            # Legacy episode-level evaluation history
            # ==================================================
            if (
                self.eval_env is not None
                and eval_every > 0
                and (ep + 1) % eval_every == 0
            ):
                eval_result = self.evaluate_current_policy_objective(
                    n_eval_episodes=eval_n_episodes,
                    steps_per_episode=eval_steps_per_episode,
                    eval_seeds=eval_seeds,
                    deterministic=eval_deterministic
                )

                eval_episode_idx_history.append(
                    ep + 1
                )

                eval_objective_mean_history.append(
                    eval_result["objective_mean"]
                )

                eval_objective_std_history.append(
                    eval_result["objective_std"]
                )

                print(
                    f"[POLICY EVAL] Ep {ep+1:4d} | "
                    f"Objective:"
                    f"{eval_result['objective_mean']:.6f} "
                    f"± {eval_result['objective_std']:.6f} | "
                    f"EvalEpisodes:{eval_n_episodes} | "
                    f"Seeds:{list(eval_result['eval_seeds'])}"
                )

        results = {
            # reward / queue
            "global_reward": global_reward_history,
            "handover_ratio": handover_ratio_history,
            "episode_handover_ratio": episode_handover_ratio_history,
            "episode_handover_gap":episode_handover_gap_history,
            "kappa": float(self.env.kappa),

            # final-policy evaluation objective
            "eval_episode_idx_history":
                eval_episode_idx_history,
            "eval_objective_mean_history":
                eval_objective_mean_history,
            "eval_objective_std_history":
                eval_objective_std_history,

            # step-wise policy-improvement evaluation
            "policy_eval_step_history": policy_eval_step_history,
            "policy_eval_episode_history": policy_eval_episode_history,
            "policy_eval_update_history": policy_eval_update_history,
            "policy_eval_objective_mean_history":
                policy_eval_objective_mean_history,
            "policy_eval_objective_std_history":
                policy_eval_objective_std_history,
            "policy_eval_objective_values_history":
                policy_eval_objective_values_history,
            "policy_checkpoint_path_history":
                policy_checkpoint_path_history,

            # loss curves
            "update_step_history": update_step_history,
            "update_episode_history": update_episode_history,
            "critic_loss_history": critic_loss_history,
            "actor_ue_loss_history": actor_ue_loss_history,
            "actor_bs_loss_history": actor_bs_loss_history,
            "entropy_ue_history": entropy_ue_history,
            "entropy_bs_history": entropy_bs_history,

            "episode_reward_history": episode_reward_history,
        }
        if self._is_constrained_env():
            results.update({
                "mu_E_mean_history": mu_E_mean_history,
                "nu_H_mean_history": nu_H_mean_history,
                "RemE_mean_history": RemE_mean_history,
                "RemH_mean_history": RemH_mean_history,
                "C_E_mean_history": C_E_mean_history,
                "C_H_mean_history": C_H_mean_history,

                "episode_mu_E_last_history": episode_mu_E_last_history,
                "episode_nu_H_last_history": episode_nu_H_last_history,
                "episode_RemE_last_history": episode_RemE_last_history,
                "episode_RemH_last_history": episode_RemH_last_history,
                "episode_C_E_last_history": episode_C_E_last_history,
                "episode_C_H_last_history": episode_C_H_last_history,
            })
        else:
            results.update({
            "Q_mean_history": Q_mean_history,
            "Z_mean_history": Z_mean_history,
            "G_mean_history": G_mean_history,
            "G_max_history": G_max_history,

            # episode summaries
            
            "episode_Q_last_history": episode_Q_last_history,
            "episode_Z_last_history": episode_Z_last_history,
            "episode_G_last_history": episode_G_last_history,
        })

        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="train")

        return results

    @torch.no_grad()
    def evaluate(self, 
                 n_episodes: int,
                 steps_per_episode: int, 
                 save_npz_path: Optional[str] = None):
        
        print(f"\n{'='*84}")
        print(" EVALUATION - Episodic Horizon")
        print(f"{'='*84}")
        print(f"Eval episodes: {n_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Total eval steps: {n_episodes * steps_per_episode}")
        print(f"Hard constraint during evaluation: {self.env.use_hard_constraint}\n")

        self.ue_actor.eval()
        self.bs_actor.eval()
        self.critic.eval()

        # --------------------------------------------------
        # Slot-wise histories across all eval episodes
        # --------------------------------------------------
        throughput_history = []
        # fairness_history = []
        fairness_block_history = []
        fairness_block_x_history = []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        slot_rates = []

        global_reward_hist = []
        ue_per_user_reward_hist = []      
        
        handover_count_history = []       # slot-wise total handover count
        handover_ratio_history = []       # slot-wise handover ratio = HO / #UE
        handover_count_per_user = np.zeros(
            self.env.n_agents,
            dtype=np.float64,
        )

        served_ratio_history = []         # optional QoE
        outage_ratio_history = []         # optional QoE

        Q_mean_history = []
        Z_mean_history = []
        G_mean_history = []

        # --------------------------------------------------
        # Episode-level summaries
        # --------------------------------------------------
        episode_throughput_mean = []
        episode_fairness_last = []
        episode_on_ratio_mean = []
        episode_handover_ratio_mean = []
        episode_served_ratio_mean = []
        episode_outage_ratio_mean = []
        episode_reward_mean = []

        global_step = 0

        for ep in range(n_episodes):
            # --------------------------------------------------
            # Evaluation episode reset
            # 현재 env.reset()은 Q/Z/G뿐 아니라 위치와 채널도 reset함.
            # topology/channel을 유지하고 싶으면 env.reset_episode(reset_topology=False)로 바꾸면 됨.
            # --------------------------------------------------
            local_obs, global_obs = self.env.reset()

            ep_throughput = []
            ep_slot_rates = []
            ep_handover_ratio = []
            ep_served_ratio = []
            ep_outage_ratio = []
            ep_reward = []

            ep_power = {bs.bs_id: [] for bs in self.env.base_stations}
            ep_on100_hist = {bs.bs_id: [] for bs in self.env.base_stations}

            for ep_step in range(steps_per_episode):
                global_step += 1
                (ue_actions, ue_logp_np, ue_ent_np, ue_masks_np,
                bs_actions, bs_logp_np, bs_ent_np, bs_obs_np, bs_masks_np, cand_lists,
                v_n) = self.select_actions(local_obs, global_obs)

                next_local_obs, next_global_obs, info, done = self.env.step_joint(
                    ue_actions=ue_actions,
                    bs_actions=bs_actions,
                    cand_lists=cand_lists
                )

                # --------------------------------------------------
                # Throughput / fairness
                # --------------------------------------------------
                total_thr = float(info["total_throughput"])
                throughput_history.append(total_thr)
                ep_throughput.append(total_thr)

                rates_this_slot = [
                    float(info["served_rates"][u.ue_id])
                    for u in self.env.users
                ]
                slot_rates.append(rates_this_slot)
                ep_slot_rates.append(rates_this_slot)

                # fairness는 episode 내부 기준으로 계산하는 게 더 자연스러움
                # fair_now = float(self.env.calculate_jain_fairness(ep_slot_rates))
                # fairness_history.append(fair_now)
   
                # --------------------------------------------------
                # Handover
                # --------------------------------------------------
                handover_u = info.get("handover_u", {})

                if isinstance(handover_u, dict):
                    ho_flags = np.asarray(
                        [
                            float(handover_u[u.ue_id])
                            for u in self.env.users
                        ],
                        dtype=np.float64,
                    )
                else:
                    ho_flags = np.asarray(
                        handover_u,
                        dtype=np.float64,
                    ).reshape(-1)

                if ho_flags.size != self.env.n_agents:
                    raise ValueError(
                        f"handover_u size={ho_flags.size}, "
                        f"expected n_agents={self.env.n_agents}"
                    )

                # 사용자별 누적 handover 횟수
                handover_count_per_user += ho_flags

                # 기존 slot-wise 값
                ho_count = float(np.sum(ho_flags))
                ho_ratio = ho_count / max(1, self.env.n_agents)

                handover_count_history.append(ho_count)
                handover_ratio_history.append(ho_ratio)
                ep_handover_ratio.append(ho_ratio)

                # --------------------------------------------------
                # Served / outage
                # --------------------------------------------------
                served_flags = np.array(
                    [1.0 if info["served_rates"][u.ue_id] > 0.0 else 0.0
                    for u in self.env.users],
                    dtype=np.float32,
                )
                served_ratio = float(np.mean(served_flags))
                outage_ratio = 1.0 - served_ratio

                served_ratio_history.append(served_ratio)
                outage_ratio_history.append(outage_ratio)
                ep_served_ratio.append(served_ratio)
                ep_outage_ratio.append(outage_ratio)

                # --------------------------------------------------
                # Queue histories
                # --------------------------------------------------
                Q_mean_history.append(float(np.mean(list(info["Q_u"].values()))))
                Z_mean_history.append(float(np.mean(list(info["Z_b"].values()))))
                G_mean_history.append(float(np.mean(list(info["G_u"].values()))))

                # --------------------------------------------------
                # Power / ON ratio
                # --------------------------------------------------
                for bs_id, power in info["power_consumed"].items():
                    power = float(power)
                    power_history[bs_id].append(power)
                    ep_power[bs_id].append(power)

                # --------------------------------------------------
                # Reward
                # --------------------------------------------------
                reward = float(info["global_reward"])
                global_reward_hist.append(reward)
                ep_reward.append(reward)

                ue_per_user_reward_hist.append([
                    float(info["ue_per_user_rewards"][u.ue_id])
                    for u in self.env.users
                ])

                local_obs, global_obs = next_local_obs, next_global_obs

                # --------------------------------------------------
                # Print every 100 steps
                # --------------------------------------------------
                if (ep_step + 1) % 100 == 0:
                    recent_thr = float(np.mean(ep_throughput[-100:]))
                    recent_fair = self.env.calculate_jain_fairness(
                        ep_slot_rates,
                        window=100
                    )

                    on_parts = []
                    for bs in self.env.base_stations:
                        hist = list(self.env.bs_on_hist[bs.bs_id])
                        on_ratio_100 = float(np.mean(hist[-100:])) if len(hist) > 0 else 0.0
                        ep_on100_hist[bs.bs_id].append(on_ratio_100)
                        on_parts.append(f"BS{bs.bs_id}:{on_ratio_100:.3f}")
                    on_str = " ".join(on_parts)

                    print(
                        f"[EVAL] Ep {ep+1:3d} | "
                        f"Step {ep_step+1:5d}/{steps_per_episode} | "
                        f"GlobalStep {global_step:7d} | "
                        f"Thr(100):{recent_thr:.3f} | "
                        f"Fair:{recent_fair:.3f} | "
                        f"ON(100): {on_str}"
                    )
            
            # --------------------------------------------------
            # Episode-level summary
            # --------------------------------------------------
            ep_thr_mean = float(np.mean(ep_throughput))
            # ep_fair_last = float(self.env.calculate_jain_fairness(ep_slot_rates))

            # 1000-step block fairness
            ep_fair_block_mean, ep_fair_blocks, ep_fair_block_x = self.env.compute_block_jain_fairness(
                ep_slot_rates,
                block_size=1000
            )
            fairness_block_history.extend(ep_fair_blocks.tolist())
            fairness_block_x_history.extend(
                (ep_fair_block_x + ep * steps_per_episode).tolist()
            )
            ep_fair_last = ep_fair_block_mean

            ep_ho_mean = float(np.mean(ep_handover_ratio))
            ep_served_mean = float(np.mean(ep_served_ratio))
            ep_outage_mean = float(np.mean(ep_outage_ratio))
            ep_reward_mean_val = float(np.mean(ep_reward))

            on_per_bs = []
            for bs in self.env.base_stations:
                powers = np.asarray(ep_power[bs.bs_id], dtype=np.float32)
                on_per_bs.append(float(np.mean(powers > 0.0)))

            ep_on_mean = float(np.mean(on_per_bs))

            episode_throughput_mean.append(ep_thr_mean)
            episode_fairness_last.append(ep_fair_last)
            episode_on_ratio_mean.append(ep_on_mean)
            episode_handover_ratio_mean.append(ep_ho_mean)
            episode_served_ratio_mean.append(ep_served_mean)
            episode_outage_ratio_mean.append(ep_outage_mean)
            episode_reward_mean.append(ep_reward_mean_val)

            on_str = " ".join([
                f"BS{bs.bs_id}:{on_per_bs[i]:.3f}"
                for i, bs in enumerate(self.env.base_stations)
            ])

            print(
                f"[EVAL-EP END] Ep {ep+1:3d} | "
                f"ThrMean:{ep_thr_mean:.3f} | "
                f"Fair:{ep_fair_last:.3f} | "
                f"ONMean:{ep_on_mean:.3f} ({on_str}) | "
                f"HOmean:{ep_ho_mean:.4f}/{self.env.kappa:.4f} | "
                f"Served:{ep_served_mean:.3f} | "
                f"Outage:{ep_outage_mean:.3f}"
            )  

        # --------------------------------------------------
        # Overall summary
        # --------------------------------------------------f
        print(f"\n{'='*84}")
        print(" EVALUATION SUMMARY")
        print(f"{'='*84}")
        print(f"Mean throughput: {float(np.mean(episode_throughput_mean)):.4f}")
        print(f"Mean fairness:   {float(np.mean(episode_fairness_last)):.4f}")
        print(f"Mean ON ratio:   {float(np.mean(episode_on_ratio_mean)):.4f}")
        print(f"Mean HO ratio:   {float(np.mean(episode_handover_ratio_mean)):.4f}")
        print(f"{'='*84}\n")

        # 첫 slot에서는 handover가 없으므로 T-1 기준
        total_eval_steps = n_episodes * steps_per_episode

        handover_ratio_per_user = (
            handover_count_per_user
            / max(total_eval_steps - n_episodes, 1)
        )

        results = {
            # slot-wise histories
            "throughput_history": throughput_history,
            "fairness_block_history": fairness_block_history,
            "fairness_block_x_history": fairness_block_x_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "global_reward": global_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,

            "handover_count_history": handover_count_history,
            "handover_ratio_history": handover_ratio_history,
            "handover_count_per_user": handover_count_per_user,
            "handover_ratio_per_user": handover_ratio_per_user,

            "served_ratio_history": served_ratio_history,
            "outage_ratio_history": outage_ratio_history,

            "Q_mean_history": Q_mean_history,
            "Z_mean_history": Z_mean_history,
            "G_mean_history": G_mean_history,

            # episode-level summaries
            "episode_throughput_mean": episode_throughput_mean,
            "episode_fairness_last": episode_fairness_last,
            "episode_on_ratio_mean": episode_on_ratio_mean,
            "episode_handover_ratio_mean": episode_handover_ratio_mean,
            "episode_served_ratio_mean": episode_served_ratio_mean,
            "episode_outage_ratio_mean": episode_outage_ratio_mean,
            "episode_reward_mean": episode_reward_mean,
        }

        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="eval")

        return results

    # =========================================================
    # NPZ save
    # =========================================================
    def save_results_npz(self, results: Dict, npz_path: str, tag: str = "run"):
        os.makedirs(os.path.dirname(npz_path) if os.path.dirname(npz_path) else ".", exist_ok=True)

        tag = str(tag)
        is_constrained = (
            "mu_E_mean_history" in results
            or hasattr(self.env, "mu_E_b")
        )

        # =====================================================
        # Common arrays
        # =====================================================
        global_reward = np.asarray(results.get("global_reward", []), dtype=np.float32)
        handover_ratio = np.asarray(results.get("handover_ratio", []), dtype=np.float32)
        episode_handover_ratio = np.asarray(results.get("episode_handover_ratio", []), dtype=np.float32)
        episode_handover_gap = np.asarray(results.get("episode_handover_gap", []), dtype=np.float32)
        kappa_value = np.asarray([float(results.get("kappa", getattr(self.env, "kappa", np.nan)))], dtype=np.float32)

        Q_mean = np.asarray(results.get("Q_mean_history", []), dtype=np.float32)
        Z_mean = np.asarray(results.get("Z_mean_history", []), dtype=np.float32)
        G_mean = np.asarray(results.get("G_mean_history", []), dtype=np.float32)
        G_max = np.asarray(results.get("G_max_history", []), dtype=np.float32)

        update_steps = np.asarray(results.get("update_step_history", []), dtype=np.int32)
        update_episodes = np.asarray(results.get("update_episode_history", []), dtype=np.int32)

        critic_loss = np.asarray(results.get("critic_loss_history", []), dtype=np.float32)
        actor_ue_loss = np.asarray(results.get("actor_ue_loss_history", []), dtype=np.float32)
        actor_bs_loss = np.asarray(results.get("actor_bs_loss_history", []), dtype=np.float32)
        entropy_ue = np.asarray(results.get("entropy_ue_history", []), dtype=np.float32)
        entropy_bs = np.asarray(results.get("entropy_bs_history", []), dtype=np.float32)

        # block average for reward plotting
        reward_x_500, global_reward_500 = (
            block_avg_1d(global_reward, 500)
            if global_reward.size > 0 else
            (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32))
        )

        # =====================================================
        # TRAIN SAVE: lightweight
        # =====================================================
        if tag == "train":
            episode_reward = np.asarray(results.get("episode_reward_history", []), dtype=np.float32)
            eval_episode_idx = np.asarray(
                results.get(
                    "eval_episode_idx_history",
                    [],
                ),
                dtype=np.int32,
            )

            eval_objective_mean = np.asarray(
                results.get(
                    "eval_objective_mean_history",
                    [],
                ),
                dtype=np.float32,
            )

            eval_objective_std = np.asarray(
                results.get(
                    "eval_objective_std_history",
                    [],
                ),
                dtype=np.float32,
            )

            policy_eval_steps = np.asarray(
                results.get("policy_eval_step_history", []),
                dtype=np.int64,
            )
            policy_eval_episodes = np.asarray(
                results.get("policy_eval_episode_history", []),
                dtype=np.int32,
            )
            policy_eval_updates = np.asarray(
                results.get("policy_eval_update_history", []),
                dtype=np.int32,
            )
            policy_eval_objective_mean = np.asarray(
                results.get("policy_eval_objective_mean_history", []),
                dtype=np.float32,
            )
            policy_eval_objective_std = np.asarray(
                results.get("policy_eval_objective_std_history", []),
                dtype=np.float32,
            )
            raw_policy_values = results.get(
                "policy_eval_objective_values_history",
                [],
            )
            if len(raw_policy_values) > 0:
                policy_eval_objective_values = np.stack(
                    raw_policy_values,
                    axis=0,
                ).astype(np.float32)
            else:
                policy_eval_objective_values = np.empty(
                    (0, 0),
                    dtype=np.float32,
                )
            policy_checkpoint_paths = np.asarray(
                results.get("policy_checkpoint_path_history", []),
                dtype=str,
            )

            if is_constrained:
                mu_E_mean = np.asarray(results.get("mu_E_mean_history", []), dtype=np.float32)
                nu_H_mean = np.asarray(results.get("nu_H_mean_history", []), dtype=np.float32)
                RemE_mean = np.asarray(results.get("RemE_mean_history", []), dtype=np.float32)
                RemH_mean = np.asarray(results.get("RemH_mean_history", []), dtype=np.float32)
                C_E_mean = np.asarray(results.get("C_E_mean_history", []), dtype=np.float32)
                C_H_mean = np.asarray(results.get("C_H_mean_history", []), dtype=np.float32)

                episode_mu_E_last = np.asarray(results.get("episode_mu_E_last_history", []), dtype=np.float32)
                episode_nu_H_last = np.asarray(results.get("episode_nu_H_last_history", []), dtype=np.float32)
                episode_RemE_last = np.asarray(results.get("episode_RemE_last_history", []), dtype=np.float32)
                episode_RemH_last = np.asarray(results.get("episode_RemH_last_history", []), dtype=np.float32)
                episode_C_E_last = np.asarray(results.get("episode_C_E_last_history", []), dtype=np.float32)
                episode_C_H_last = np.asarray(results.get("episode_C_H_last_history", []), dtype=np.float32)

                np.savez_compressed(
                    npz_path,
                    tag=tag,
                    n_users=int(self.env.n_agents),
                    n_bs=int(self.env.n_bs),

                    global_reward=global_reward,
                    reward_x_500=reward_x_500,
                    global_reward_500=global_reward_500,

                    eval_episode_idx=eval_episode_idx,
                    eval_objective_mean=eval_objective_mean,
                    eval_objective_std=eval_objective_std,

                    policy_eval_steps=policy_eval_steps,
                    policy_eval_episodes=policy_eval_episodes,
                    policy_eval_updates=policy_eval_updates,
                    policy_eval_objective_mean=policy_eval_objective_mean,
                    policy_eval_objective_std=policy_eval_objective_std,
                    policy_eval_objective_values=policy_eval_objective_values,
                    policy_checkpoint_paths=policy_checkpoint_paths,

                    handover_ratio=handover_ratio,
                    episode_handover_ratio=episode_handover_ratio,
                    episode_handover_gap=episode_handover_gap,
                    kappa=kappa_value,

                    mu_E_mean=mu_E_mean,
                    nu_H_mean=nu_H_mean,
                    RemE_mean=RemE_mean,
                    RemH_mean=RemH_mean,
                    C_E_mean=C_E_mean,
                    C_H_mean=C_H_mean,

                    episode_reward=episode_reward,
                    episode_mu_E_last=episode_mu_E_last,
                    episode_nu_H_last=episode_nu_H_last,
                    episode_RemE_last=episode_RemE_last,
                    episode_RemH_last=episode_RemH_last,
                    episode_C_E_last=episode_C_E_last,
                    episode_C_H_last=episode_C_H_last,

                    update_steps=update_steps,
                    update_episodes=update_episodes,
                    critic_loss=critic_loss,
                    actor_ue_loss=actor_ue_loss,
                    actor_bs_loss=actor_bs_loss,
                    entropy_ue=entropy_ue,
                    entropy_bs=entropy_bs,
                )
                print(f"✅ Saved constrained train npz: {npz_path}")
                return

            else:
                episode_Q_last = np.asarray(results.get("episode_Q_last_history", []), dtype=np.float32)
                episode_Z_last = np.asarray(results.get("episode_Z_last_history", []), dtype=np.float32)
                episode_G_last = np.asarray(results.get("episode_G_last_history", []), dtype=np.float32)

                np.savez_compressed(
                    npz_path,
                    tag=tag,
                    n_users=int(self.env.n_agents),
                    n_bs=int(self.env.n_bs),

                    global_reward=global_reward,
                    reward_x_500=reward_x_500,
                    global_reward_500=global_reward_500,

                    eval_episode_idx=eval_episode_idx,
                    eval_objective_mean=eval_objective_mean,
                    eval_objective_std=eval_objective_std,

                    policy_eval_steps=policy_eval_steps,
                    policy_eval_episodes=policy_eval_episodes,
                    policy_eval_updates=policy_eval_updates,
                    policy_eval_objective_mean=policy_eval_objective_mean,
                    policy_eval_objective_std=policy_eval_objective_std,
                    policy_eval_objective_values=policy_eval_objective_values,
                    policy_checkpoint_paths=policy_checkpoint_paths,

                    handover_ratio=handover_ratio,
                    episode_handover_ratio=episode_handover_ratio,
                    episode_handover_gap=episode_handover_gap,
                    kappa=kappa_value,

                    Q_mean=Q_mean,
                    Z_mean=Z_mean,
                    G_mean=G_mean,
                    G_max=G_max,

                    episode_reward=episode_reward,
                    episode_Q_last=episode_Q_last,
                    episode_Z_last=episode_Z_last,
                    episode_G_last=episode_G_last,

                    update_steps=update_steps,
                    update_episodes=update_episodes,
                    critic_loss=critic_loss,
                    actor_ue_loss=actor_ue_loss,
                    actor_bs_loss=actor_bs_loss,
                    entropy_ue=entropy_ue,
                    entropy_bs=entropy_bs,
                )
                print(f"✅ Saved train npz: {npz_path}")
                return
        
        # =====================================================
        # EVAL SAVE: performance metrics
        # =====================================================
        thr = np.asarray(results.get("throughput_history", []), dtype=np.float32)
        fair = np.asarray(results.get("fairness_block_history", []), dtype=np.float32)
        slot_rates = np.asarray(results.get("slot_rates", []), dtype=np.float32)

        ue_per_user = np.asarray(results.get("ue_per_user_reward", []), dtype=np.float32)

        handover_count = np.asarray(results.get("handover_count_history", []), dtype=np.float32)
        handover_ratio = np.asarray(results.get("handover_ratio_history", []), dtype=np.float32)
        handover_count_per_user = np.asarray(results.get("handover_count_per_user",[]),dtype=np.float32)
        handover_ratio_per_user = np.asarray(results.get("handover_ratio_per_user",[]),dtype=np.float32)

        served_ratio = np.asarray(results.get("served_ratio_history", []), dtype=np.float32)
        outage_ratio = np.asarray(results.get("outage_ratio_history", []), dtype=np.float32)

        episode_throughput_mean = np.asarray(results.get("episode_throughput_mean", []), dtype=np.float32)
        episode_fairness_last = np.asarray(results.get("episode_fairness_last", []), dtype=np.float32)
        episode_on_ratio_mean = np.asarray(results.get("episode_on_ratio_mean", []), dtype=np.float32)
        episode_handover_ratio_mean = np.asarray(results.get("episode_handover_ratio_mean", []), dtype=np.float32)
        episode_served_ratio_mean = np.asarray(results.get("episode_served_ratio_mean", []), dtype=np.float32)
        episode_outage_ratio_mean = np.asarray(results.get("episode_outage_ratio_mean", []), dtype=np.float32)
        episode_reward_mean = np.asarray(results.get("episode_reward_mean", []), dtype=np.float32)

        if ue_per_user.ndim == 2 and ue_per_user.shape[0] > 0:
            mean_user_reward_step = ue_per_user.mean(axis=1).astype(np.float32)
            mean_user_reward_ma100 = moving_avg(mean_user_reward_step, 100)
        else:
            mean_user_reward_step = np.asarray([], dtype=np.float32)
            mean_user_reward_ma100 = np.asarray([], dtype=np.float32)

        # block average for reward plotting
        user_reward_x_500, user_mean_reward_500 = (
            block_avg_1d(mean_user_reward_step, 500)
            if mean_user_reward_step.size > 0 else
            (np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32))
        )

        power_hist = results.get("power_history", {})
        bs_ids_sorted = sorted(list(power_hist.keys())) if isinstance(power_hist, dict) else []

        power_mat = []
        for bs_id in bs_ids_sorted:
            power_mat.append(np.asarray(power_hist[bs_id], dtype=np.float32))
        power_mat = np.stack(power_mat, axis=0) if len(power_mat) > 0 else np.zeros((0, len(thr)), dtype=np.float32)

        # --------------------------------------------------
        # Energy constraint metrics
        # --------------------------------------------------
        # power_mat shape: [n_bs, T]
        # P_max is used when BS is ON, 0 when OFF
        if power_mat.size > 0:
            bs_on_mat = (power_mat > 0.0).astype(np.float32)       # [n_bs, T]
            bs_on_ratio_per_bs = bs_on_mat.mean(axis=1)            # [n_bs]
            bs_on_ratio_mean = np.asarray([bs_on_ratio_per_bs.mean()], dtype=np.float32)

            energy_budget_ratio = np.asarray([self.env.power_budget_ratio], dtype=np.float32)
            energy_violation_per_bs = np.maximum(
                0.0,
                bs_on_ratio_per_bs - self.env.power_budget_ratio
            ).astype(np.float32)
            energy_violation_mean = np.asarray([energy_violation_per_bs.mean()], dtype=np.float32)
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
            handover_ratio_mean = np.asarray([float(np.mean(handover_ratio))], dtype=np.float32)
            handover_budget_ratio = np.asarray([float(self.env.kappa)], dtype=np.float32)
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

        # --------------------------------------------------
        # Eq. (11) Objective / EA-PF Utility
        # J = sum_u log(avg_t R_u(t)) - lambda_E * avg_t sum_b e_b y_b(t)
        # --------------------------------------------------
        eps = 1e-12
        obj_window = min(10000, len(thr))

        if slot_rates.ndim == 2 and slot_rates.shape[0] > 0:
            recent_slot_rates = slot_rates[-obj_window:]
            avg_user_rates = np.mean(recent_slot_rates, axis=0)
            pf_utility_value = float(np.sum(np.log(avg_user_rates + eps)))
        else:
            avg_user_rates = np.asarray([], dtype=np.float32)
            pf_utility_value = np.nan

        if power_mat.size > 0:
            recent_power_mat = power_mat[:, -obj_window:]
            energy_per_slot = np.sum(recent_power_mat, axis=0)
            avg_energy_cost_value = float(np.mean(energy_per_slot))
        else:
            energy_per_slot = np.asarray([], dtype=np.float32)
            avg_energy_cost_value = np.nan

        lambda_E_value = float(getattr(self.env, "lambda_E", 0.0))

        if np.isnan(pf_utility_value) or np.isnan(avg_energy_cost_value):
            ea_pf_utility_value = np.nan
        else:
            ea_pf_utility_value = pf_utility_value - lambda_E_value * avg_energy_cost_value

        avg_user_rates = np.asarray(avg_user_rates, dtype=np.float32)
        energy_per_slot = np.asarray(energy_per_slot, dtype=np.float32)
        pf_utility = np.asarray([pf_utility_value], dtype=np.float32)
        avg_energy_cost = np.asarray([avg_energy_cost_value], dtype=np.float32)
        ea_pf_utility = np.asarray([ea_pf_utility_value], dtype=np.float32)

        np.savez_compressed(
            npz_path,
            tag=str(tag),
            n_users=int(self.env.n_agents),
            n_bs=int(self.env.n_bs),

            # slot-wise performance
            throughput=thr,
            fairness=fair,
            slot_rates=slot_rates,
            power_mat=power_mat,
            bs_ids=np.asarray(bs_ids_sorted, dtype=np.int32),

            # reward
            global_reward=global_reward,
            ue_per_user_reward=ue_per_user,
            mean_user_reward_step=mean_user_reward_step,
            mean_user_reward_ma100=mean_user_reward_ma100,
            reward_x_500=reward_x_500,
            global_reward_500=global_reward_500,
            user_reward_x_500=user_reward_x_500,
            user_mean_reward_500=user_mean_reward_500,

            # handover / QoE
            handover_count=handover_count,
            handover_ratio=handover_ratio,
            handover_ratio_mean=handover_ratio_mean,
            handover_count_per_user=handover_count_per_user,
            handover_ratio_per_user=handover_ratio_per_user,
            served_ratio=served_ratio,
            outage_ratio=outage_ratio,

            # ON / energy constraint
            bs_on_ratio_per_bs=bs_on_ratio_per_bs,
            bs_on_ratio_mean=bs_on_ratio_mean,
            energy_budget_ratio=energy_budget_ratio,
            energy_violation_per_bs=energy_violation_per_bs,
            energy_violation_mean=energy_violation_mean,
            energy_violation_ratio=energy_violation_ratio,

            # handover constraint
            handover_budget_ratio=handover_budget_ratio,
            handover_violation_mean=handover_violation_mean,
            handover_violation_flag=handover_violation_flag,

            # queue
            Q_mean=Q_mean,
            Z_mean=Z_mean,
            G_mean=G_mean,
            G_max=G_max,

            # episode-level evaluation summaries
            episode_throughput_mean=episode_throughput_mean,
            episode_fairness_last=episode_fairness_last,
            episode_on_ratio_mean=episode_on_ratio_mean,
            episode_handover_ratio_mean=episode_handover_ratio_mean,
            episode_served_ratio_mean=episode_served_ratio_mean,
            episode_outage_ratio_mean=episode_outage_ratio_mean,
            episode_reward_mean=episode_reward_mean,

            # objective
            lambda_E=np.asarray([lambda_E_value], dtype=np.float32),
            avg_user_rates=avg_user_rates,
            energy_per_slot=energy_per_slot,
            pf_utility=pf_utility,
            avg_energy_cost=avg_energy_cost,
            ea_pf_utility=ea_pf_utility,
            performance_metric=ea_pf_utility,
        )

        print(f"✅ Saved eval npz: {npz_path}")

    # helpers
    def _is_constrained_env(self) -> bool:
        return hasattr(self.env, "mu_E_b") and hasattr(self.env, "nu_H_u")
    
    def _get_constrained_log_stats(self):
        mu_vals = list(getattr(self.env, "mu_E_b", {}).values())
        nu_vals = list(getattr(self.env, "nu_H_u", {}).values())

        rem_e_vals = [
            self.env._remaining_energy_ratio(bs.bs_id)
            for bs in self.env.base_stations
        ]

        rem_h_vals = [
            self.env._remaining_handover_ratio(u.ue_id)
            for u in self.env.users
        ]

        # Last episode-level violation.
        # 아직 dual update 전이면 0.0으로 둠.
        if len(getattr(self.env, "C_E_b_history", [])) > 0:
            C_E_mean = float(np.mean(self.env.C_E_b_history[-1]))
        else:
            C_E_mean = 0.0

        if len(getattr(self.env, "C_H_u_history", [])) > 0:
            C_H_mean = float(np.mean(self.env.C_H_u_history[-1]))
        else:
            C_H_mean = 0.0

        return {
            "mu_E_mean": float(np.mean(mu_vals)) if len(mu_vals) > 0 else 0.0,
            "nu_H_mean": float(np.mean(nu_vals)) if len(nu_vals) > 0 else 0.0,
            "RemE_mean": float(np.mean(rem_e_vals)) if len(rem_e_vals) > 0 else 0.0,
            "RemH_mean": float(np.mean(rem_h_vals)) if len(rem_h_vals) > 0 else 0.0,
            "C_E_mean": C_E_mean,
            "C_H_mean": C_H_mean,
        }