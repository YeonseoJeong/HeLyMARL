"""
Usage examples:
  python qmixtest.py --n_env_steps 5000 --rollout_horizon 200 --device cuda
  python qmixtest.py --mode eval --episodes 5 --eval_epsilon 0.05
"""
from __future__ import annotations

import argparse
import random
import sys, os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /home/.../LyMARL
sys.path.insert(0, REPO_ROOT)
from dataclasses import asdict
from typing import List, Tuple
import numpy as np 
import torch
from basestation import SmallCellBaseStation
from user_equipment import UserEquipment
from core import generate_triangle_coverage
from LyMARL.env import MAPPOEnvironment
from LyMARL.trainer import MAPPOTrainer
from benchmark.HeteroQMIXAgent import HeteroQMIXAgent, HeteroQMIXcfg
import yaml 
import json
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _sample_positions_uniform(n: int, low: float=10.0, high: float = 90.0) -> List[Tuple[float, float]]:
    pts = np.random.uniform(low=low, high=high, size=(n, 2))
    return [(float(x), float(y)) for x, y in pts]

def build_env(n_ue: int, n_bs: int, bs_top_k: int, power_budget_ratio: float,
              V: float, enable_mobility: bool, enable_channel_variation: bool,
              hard_window_len: int, on_window: int, bs_over_penalty: float):
    # BS positions: triangle template + fill remainder uniformly
    tri_pos = generate_triangle_coverage()
    bs_pos = list(tri_pos[:min(len(tri_pos), n_bs)])
    if len(bs_pos) < n_bs:
        bs_pos += _sample_positions_uniform(n_bs - len(bs_pos))

    base_stations = [
        SmallCellBaseStation(bs_id = i+1, position=bs_pos[i], beam_limit=np.inf, coverage_radius=np.inf)
        for i in range(n_bs)
    ]

    # UE positions (env.reset() will randomize again)
    ue_pos = _sample_positions_uniform(n_ue)
    users = [
        UserEquipment(ue_id = i+1, position=ue_pos[i]) for i in range(n_ue)
    ]
    env = MAPPOEnvironment(
        base_stations=base_stations,
        users=users,
        V=V,
        power_budget_ratio=power_budget_ratio,
        enable_mobility=enable_mobility,
        enable_channel_variation=enable_channel_variation,
        on_window=on_window,
        bs_top_k=bs_top_k,
        hard_window_len=hard_window_len,
        bs_over_penalty=bs_over_penalty,
    )
    return env

def run_train(args):
    env = build_env(
        n_ue=args.n_ue,
        n_bs=args.n_bs,
        bs_top_k=args.bs_top_k,
        power_budget_ratio=args.power_budget_ratio,
        V=args.V,
        enable_mobility=args.enable_mobility,
        enable_channel_variation=args.enable_channel_variation,
        hard_window_len=args.hard_window_len,
        on_window=args.on_window,
        bs_over_penalty=args.bs_over_penalty
    )

    cfg = HeteroQMIXcfg(
        hidden_dim=args.hidden_dim,
        lr = args.lr,
        gamma=args.gamma,
        tau=args.tau,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        capacity_episodes=args.capacity_episodes,
        update_interval_steps=args.update_interval_steps,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
    )
    
    agent = HeteroQMIXAgent(env=env, cfg=cfg, log_dir="./results/train_logs", device=args.device)
    print("\n[QMIX TEST] Config:")
    print({
        "env":{
            "n_ue": args.n_ue,
            "n_bs": args.n_bs,
            "bs_top_k": args.bs_top_k,
            "power_budget_ratio": args.power_budget_ratio,
            "V": args.V,
            "enable_mobility": args.enable_mobility,
            "enable_channel_variation": args.enable_channel_variation,
            "hard_window_len": args.hard_window_len,
            "on_window": args.on_window,
            "bs_over_penalty": args.bs_over_penalty
        },
        "agent": asdict(cfg),
        }
    )
    print()
    logs = agent.train(n_env_steps=args.n_env_steps, rollout_horizon=args.rollout_horizon)
    rollouts = [x for x in logs if x.get("type") == "rollout"]
    updates = [x for x in logs if x.get("type") == "update"]
    if rollouts:
        last = rollouts[-1]
        print(
            f"[DONE] env_steps={agent.total_env_steps} | last_ep_len={last['ep_len']:.0f} "
            f"| last_ep_r_ue_sum={last['ep_r_ue_sum']:.3f} | last_ep_r_bs_mean={last['ep_r_bs_mean']:.3f} "
            f"| epsilon={last['epsilon']:.3f}"
            f"| thr_mean={last.get('thr_mean', float('nan')):.3f} "
            f"| fair_ep={last.get('fair_ep', float('nan')):.3f}"
            f"| on_ratio={last.get('on_ratio_mean', float('nan')):.3f} "
        )
    if updates:
        last_u = updates[-1]
        print(
            f"[DONE] last_update loss={last_u['loss']:.4f} (ue={last_u['loss_ue']:.4f}, bs={last_u['loss_bs']:.4f}) "
            f"| epsilon={last_u['epsilon']:.3f}"
        )

@torch.no_grad()
def run_eval(args):
    env = build_env(
        n_ue=args.n_ue,
        n_bs=args.n_bs,
        bs_top_k=args.bs_top_k,
        power_budget_ratio=args.power_budget_ratio,
        V=args.V,
        enable_mobility=args.enable_mobility,
        enable_channel_variation=args.enable_channel_variation,
        hard_window_len=args.hard_window_len,
        on_window=args.on_window,
        bs_over_penalty=args.bs_over_penalty
    )
    cfg = HeteroQMIXcfg(
        hidden_dim=args.hidden_dim,
        lr = args.lr,
        gamma=args.gamma,
        tau=args.tau,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        capacity_episodes=args.capacity_episodes,
        update_interval_steps=args.update_interval_steps,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=1.0,
    )
    agent = HeteroQMIXAgent(env=env, cfg=cfg, log_dir="./results/eval_logs", device=args.device)
    agent.eps = args.eval_epsilon

    print(f"\n[EVAL] episodes={args.episodes} | horizon={args.rollout_horizon} | epsilon={args.eval_epsilon}\n")
    for ep_i in range(args.episodes):
        out = agent.rollout_episode(n_steps=args.rollout_horizon)
        print(
            f"  ep={ep_i:03d} | len={out['ep_len']:.0f} | r_ue_sum={out['ep_r_ue_sum']:.3f} "
            f"| r_bs_mean={out['ep_r_bs_mean']:.3f} | epsilon={out['epsilon']:.3f}"
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # env
    parser.add_argument("--n_ue", type=int, default=20)
    parser.add_argument("--n_bs", type=int, default=3)
    parser.add_argument("--bs_top_k", type=int, default=5)
    parser.add_argument("--power_budget_ratio", type=float, default=0.6)
    parser.add_argument("--V", type=float, default=20.0)
    parser.add_argument("--enable_mobility", action="store_true", default=True)
    parser.add_argument("--enable_channel_variation", action="store_true", default=True)
    parser.add_argument("--hard_window_len", type=int, default=1000)
    parser.add_argument("--on_window", type=int, default=100)
    parser.add_argument("--bs_over_penalty", type=float, default=50.0)
    # rollout/train
    parser.add_argument("--rollout_horizon", type=int, default=200)
    parser.add_argument("--n_env_steps", type=int, default=50000)
    # agent
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--capacity_episodes", type=int, default=10000)
    parser.add_argument("--update_interval_steps", type=int, default=128)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.9995)

    # eval
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval_epsilon", type=float, default=0.05)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)

if __name__ == "__main__":
    main()