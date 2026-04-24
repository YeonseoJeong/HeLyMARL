import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
from typing import Dict, Optional

from networks_happo import (
    UEActorNetwork,
    BSActorNetwork,
    CentralizedCritic,
    ValueNorm
)
from utils_happo import moving_avg, block_avg_1d


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
    def select_actions(self, local_obs: Dict[int, np.ndarray], global_obs: np.ndarray):
        users = self.env.users
        global_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        v_n = self.critic(global_t).squeeze(0)

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
    def train(self, n_steps: int, update_interval: int = 128, save_npz_path: Optional[str] = None):
        print(f"\n{'='*100}")
        print(" HAPPO Training")
        print(f"{'='*100}")
        print(f"Total train steps: {n_steps}")
        print(f"Update interval: {update_interval}")
        print(f"Hard constraint during training: {self.env.use_hard_constraint}")
        print(f"{'='*100}\n")

        throughput_history = []
        fairness_history = []
        power_history = {bs.bs_id: [] for bs in self.env.base_stations}
        slot_rates = []
        queue_history = {"Q_u": defaultdict(list), "Z_b": defaultdict(list)}

        global_reward_hist = []
        ue_per_user_reward_hist = []

        local_obs, global_obs = self.env.reset()

        for step in range(n_steps):
            (ue_actions, ue_logp_np, ue_ent_np, ue_masks_np,
             bs_actions, bs_logp_np, bs_ent_np, bs_obs_np, bs_masks_np, cand_lists,
             v_n) = self.select_actions(local_obs, global_obs)

            next_local_obs, next_global_obs, info, done = self.env.step_joint(
                ue_actions=ue_actions,
                bs_actions=bs_actions,
                cand_lists=cand_lists
            )

            with torch.no_grad():
                next_global_t = torch.as_tensor(next_global_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
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

            throughput_history.append(info["total_throughput"])
            rates_this_slot = [info["served_rates"][u.ue_id] for u in self.env.users]
            slot_rates.append(rates_this_slot)
            fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

            for bs_id, power in info["power_consumed"].items():
                power_history[bs_id].append(power)

            for ue_id, q_val in info["Q_u"].items():
                queue_history["Q_u"][ue_id].append(q_val)
            for bs_id, zb_val in info["Z_b"].items():
                queue_history["Z_b"][bs_id].append(zb_val)

            global_reward_hist.append(reward)
            ue_per_user_reward_hist.append([float(info["ue_per_user_rewards"][u.ue_id]) for u in self.env.users])
            
            local_obs, global_obs = next_local_obs, next_global_obs

            if (step + 1) % update_interval == 0:
                losses = self.update()
                if losses:
                    print(
                        f"[UPDATE] Step {step+1} | "
                        f"UE_Actor:{losses['actor_ue']:.4f} | BS_Actor:{losses['actor_bs']:.4f} | "
                        f"Critic:{losses['critic']:.4f} |  "
                        f"Ent(UE):{losses['entropy_ue']:.4f} | Ent(BS):{losses['entropy_bs']:.4f}"
                    )

            if (step + 1) % 100 == 0:
                recent_thr = float(np.mean(throughput_history[-100:]))
                recent_fair = float(fairness_history[-1])
                global_rew_100 = float(np.mean(global_reward_hist[-100:]))

                on_parts = []
                for bs in self.env.base_stations:
                    hist = list(self.env.bs_on_hist[bs.bs_id])
                    on_ratio_100 = float(np.mean(hist[-100:])) if len(hist) > 0 else 0.0
                    on_parts.append(f"BS{bs.bs_id}:{on_ratio_100:.3f}")
                on_str = " ".join(on_parts)

                print(
                    f"Step {step+1:5d} | Thr:{recent_thr:.3f} | Fair:{recent_fair:.3f} | "
                    f"ON(100): {on_str} | "
                    f"GlobalRew(100):{global_rew_100:.3f}"
                )

        results = {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "queue_history": queue_history,

            "global_reward": global_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,
        }

        if save_npz_path is not None:
            self.save_results_npz(results, save_npz_path, tag="train")

        return results

    @torch.no_grad()
    def evaluate(self, n_steps: int, save_npz_path: Optional[str] = None):
        print(f"\n{'='*84}")
        print(" EVALUATION (No Learning)")
        print(f"{'='*84}")
        print(f"Total eval steps: {n_steps}")
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

        local_obs, global_obs = self.env.reset()

        for step in range(n_steps):
            (ue_actions, ue_logp_np, ue_ent_np, ue_masks_np,
             bs_actions, bs_logp_np, bs_ent_np, bs_obs_np, bs_masks_np, cand_lists,
             v_n) = self.select_actions(local_obs, global_obs)

            next_local_obs, next_global_obs, info, done = self.env.step_joint(
                ue_actions=ue_actions,
                bs_actions=bs_actions,
                cand_lists=cand_lists
            )

            throughput_history.append(info["total_throughput"])
            rates_this_slot = [info["served_rates"][u.ue_id] for u in self.env.users]
            slot_rates.append(rates_this_slot)
            fairness_history.append(self.env.calculate_jain_fairness(slot_rates))

            for bs_id, power in info["power_consumed"].items():
                power_history[bs_id].append(power)

            global_reward_hist.append(float(info["global_reward"]))
            ue_per_user_reward_hist.append([float(info["ue_per_user_rewards"][u.ue_id]) for u in self.env.users])

            local_obs, global_obs = next_local_obs, next_global_obs

            if (step + 1) % 100 == 0:
                recent_thr = float(np.mean(throughput_history[-100:]))
                recent_fair = float(fairness_history[-1])

                on_parts = []
                for bs in self.env.base_stations:
                    hist = list(self.env.bs_on_hist[bs.bs_id])
                    on_ratio_100 = float(np.mean(hist[-100:])) if len(hist) > 0 else 0.0
                    eval_on100_hist[bs.bs_id].append(on_ratio_100)
                    on_parts.append(f"BS{bs.bs_id}:{on_ratio_100:.3f}")
                on_str = " ".join(on_parts)

                print(
                    f"[EVAL] Step {step+1:5d} | Thr:{recent_thr:.3f} | Fair:{recent_fair:.3f} | "
                    f"ON(100): {on_str}"
                )

            if (step + 1) % 10000 == 0:
                thr_10k_mean = float(np.mean(throughput_history[-10000:]))
                fair_10k_mean = float(np.mean(fairness_history[-10000:]))

                on10k_parts = []
                n_blocks_10k = max(1, 10000 // 100)
                for bs in self.env.base_stations:
                    recent_on100 = eval_on100_hist[bs.bs_id][-n_blocks_10k:]
                    on10k_mean = float(np.mean(recent_on100)) if len(recent_on100) > 0 else 0.0
                    on10k_parts.append(f"BS{bs.bs_id}:{on10k_mean:.3f}")
                on10k_str = " ".join(on10k_parts)

                print(
                    f"[EVAL-10K] Step {step+1:5d} | "
                    f"ThroughputMean(10k):{thr_10k_mean:.3f} | "
                    f"Mean(step-wise Fair(100) over 10k):{fair_10k_mean:.3f} | "
                    f"ON100-Mean(10k): {on10k_str}"
                )

        results = {
            "throughput_history": throughput_history,
            "fairness_history": fairness_history,
            "power_history": power_history,
            "slot_rates": slot_rates,
            "global_reward": global_reward_hist,
            "ue_per_user_reward": ue_per_user_reward_hist,
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

        if ue_per_user.ndim == 2 and ue_per_user.shape[0] > 0:
            mean_user_reward_step = ue_per_user.mean(axis=1).astype(np.float32)
            mean_user_reward_ma100 = moving_avg(mean_user_reward_step, 100)
        else:
            mean_user_reward_step = np.asarray([], dtype=np.float32)
            mean_user_reward_ma100 = np.asarray([], dtype=np.float32)

        # block average for reward plotting
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
        power_mat = np.stack(power_mat, axis=0) if len(power_mat) > 0 else np.zeros((0, len(thr)), dtype=np.float32)

        np.savez_compressed(
            npz_path,
            tag=str(tag),
            n_users=int(self.env.n_agents),
            n_bs=int(self.env.n_bs),

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
        )
        print(f"✅ Saved results npz: {npz_path}")