"""
End-to-end PF/Jensen-HAPPO training and timescale verification.

This script performs the advisor-requested experiment in one run:

1) Train the selected Lagrangian baseline(s) from scratch.
   - 30 training episodes by default
   - physical energy costs: use_dimensionless=False
   - no hard budget mask during training
   - dual multipliers fixed within each episode
   - checkpoints saved after episodes 10, 25, and 30
   - dual histories saved after every completed episode

2) Inter-episode verification of Theorem 1.
   - mean post-update energy/handover multipliers
   - raw maximum signed violations
   - running average signed violations

3) Intra-episode verification at checkpoints 10, 25, and 30.
   - D_E and D_H
   - energy and handover budget utilization
   - HeLyMARL K=10 horizontal reference

The script is configured for a one-training-seed pilot run. After checking the
results, change TRAIN_SEEDS to [0, 1, 2] for the paper result.
"""

from __future__ import annotations

import csv
import gc
import inspect
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
# 0. Project root
# =============================================================================
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent if FILE_DIR.name == "experiments" else FILE_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 1. Project imports
# =============================================================================
from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.env_happo import HAPPOEnvironment
from HeLyMARL.trainer_happo import HAPPOTrainer

from baselines.env_constrainedhappo import (
    JensenHAPPOEnvironment,
    PFHAPPOEnvironment,
)


# =============================================================================
# 2. CONFIG: edit paths here only when necessary
# =============================================================================

# Use both if both were trained. Remove one entry when only one is available.
VARIANTS = ["pf"]

KAPPAS = [0.015]
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

CHECKPOINT_EPISODES = [10, 25, 30]
STEPS_PER_EPISODE = 10_000
TOTAL_TRAIN_EPISODES = 30
UPDATE_INTERVAL = 128

# End-to-end execution switches.
RUN_TRAINING = False
RUN_ANALYSIS = True
TRAIN_FROM_SCRATCH = False
SAVE_OPTIMIZER_IN_CHECKPOINTS = False
# Explicit run-directory templates for the newly completed Lagrangian training.
# Expected contents of each run directory:
#   dual_history.npz
#   checkpoints/checkpoint_step_*_episode_10_end.pt
#   checkpoints/checkpoint_step_*_episode_25_end.pt
#   checkpoints/checkpoint_step_*_episode_30_end.pt
#
# If an explicit path does not exist, the resolver performs a conservative
# recursive search under results/. Ambiguous matches raise an error.
LAGRANGIAN_RUN_DIR_TEMPLATES = {
    "jensen": (
        "results/jensen_convergence_50.0/"
        "kappa_{kappa3}/train_seed_{train_seed}"
    ),
    "pf": (
        "results/pf_convergence/"
        "kappa_{kappa3}/train_seed_{train_seed}"
    ),
}

# Existing K=10 HeLyMARL models:
#   results/results_kappa/HAPPO_kappa_0.015_seed_0/model.pt
#   results/results_kappa/HAPPO_kappa_0.03_seed_0/model.pt
HELYMARL_MODEL_ROOT = Path("results/results_kappa")

OUTPUT_ROOT = Path("results/advisor_timescale_verification_50.0")

# Evaluation settings must match the paper.
POWER_BUDGET_RATIO = 0.6
LAMBDA_E = 0.0
V = 5.0

AREA_SIZE = 100
NUM_USERS = 20
BS_TOP_K = 5
ON_WINDOW = 100
BS_OVER_PENALTY = 100.0

ETA_MU = 50.0
ETA_NU = 0.5
MU_MAX = 100.0
NU_MAX = 100.0

# Existing PF/Jensen training used physical energy costs.
USE_DIMENSIONLESS = False

# Match the existing stochastic evaluation protocol.
DETERMINISTIC = False

# False resumes safely from already saved trajectory files.
OVERWRITE_EVALUATION = True


# =============================================================================
# 3. Common helpers
# =============================================================================
def kappa_tags(kappa: float) -> List[str]:
    """Return folder-name variants used in the existing project."""
    candidates = [
        f"{kappa:.3f}",
        f"{kappa:.2f}",
        str(float(kappa)),
    ]

    unique: List[str] = []
    for value in candidates:
        if value not in unique:
            unique.append(value)

    return unique


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def std_across_train_seeds(
    values: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Sample std for >=2 seeds; zero for a one-seed pilot run."""
    values = np.asarray(values, dtype=np.float64)

    if values.shape[axis] <= 1:
        return np.zeros_like(
            np.mean(values, axis=axis),
            dtype=np.float64,
        )

    return np.std(values, axis=axis, ddof=1)


def as_float_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def lookup_entity_value(
    container: object,
    entity_id: int,
    zero_based_index: int,
    name: str,
) -> float:
    """
    Read an entity-specific scalar from either a dict or an array/list.
    Fail loudly instead of silently guessing.
    """
    if isinstance(container, Mapping):
        if entity_id in container:
            return float(container[entity_id])

        if str(entity_id) in container:
            return float(container[str(entity_id)])

        raise KeyError(
            f"{name} has no entry for entity_id={entity_id}. "
            f"Available keys={list(container.keys())}"
        )

    array = np.asarray(container, dtype=np.float64).reshape(-1)

    if zero_based_index >= array.size:
        raise IndexError(
            f"{name} has length {array.size}, "
            f"but index {zero_based_index} was requested."
        )

    return float(array[zero_based_index])


def save_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    ensure_dir(path.parent)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )


# =============================================================================
# 4. Network, environments, and trainer
# =============================================================================
def make_network(seed: int):
    set_seed(seed)

    positions = generate_triangle_coverage(
        AREA_SIZE,
        35,
    )

    base_stations = [
        SmallCellBaseStation(
            index + 1,
            position,
            10,
            35,
        )
        for index, position in enumerate(positions)
    ]

    users = [
        UserEquipment(
            index + 1,
            (
                np.random.uniform(10, 90),
                np.random.uniform(10, 90),
            ),
        )
        for index in range(NUM_USERS)
    ]

    return base_stations, users


def make_lagrangian_env(
    seed: int,
    variant: str,
    kappa: float,
    *,
    training: bool = False,
) -> JensenHAPPOEnvironment | PFHAPPOEnvironment:
    base_stations, users = make_network(seed)

    common_kwargs = dict(
        base_stations=base_stations,
        users=users,
        V=V,
        power_budget_ratio=POWER_BUDGET_RATIO,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=ON_WINDOW,
        bs_top_k=BS_TOP_K,
        hard_window_len=STEPS_PER_EPISODE,
        bs_over_penalty=BS_OVER_PENALTY,
        use_hard_constraint=(not training),
        lambda_E=LAMBDA_E,
        kappa=kappa,
        eta_mu=ETA_MU,
        eta_nu=ETA_NU,
        total_train_episodes=TOTAL_TRAIN_EPISODES,
        mu_max=MU_MAX,
        nu_max=NU_MAX,
        use_dimensionless=USE_DIMENSIONLESS,
        episode_length=STEPS_PER_EPISODE,
    )

    if variant == "jensen":
        return JensenHAPPOEnvironment(**common_kwargs)

    if variant == "pf":
        return PFHAPPOEnvironment(**common_kwargs)

    raise ValueError(f"Unknown variant: {variant}")


def make_helymarl_env(
    seed: int,
    kappa: float,
) -> HAPPOEnvironment:
    base_stations, users = make_network(seed)

    return HAPPOEnvironment(
        base_stations=base_stations,
        users=users,
        V=V,
        power_budget_ratio=POWER_BUDGET_RATIO,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=ON_WINDOW,
        bs_top_k=BS_TOP_K,
        hard_window_len=STEPS_PER_EPISODE,
        bs_over_penalty=BS_OVER_PENALTY,
        use_hard_constraint=True,
        lambda_E=LAMBDA_E,
        kappa=kappa,
    )


def make_trainer(env) -> HAPPOTrainer:
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


# =============================================================================
# 5. Lagrangian training from scratch
# =============================================================================
def validate_training_environment(env) -> None:
    """Fail early when the imported constrained environment is still outdated."""
    if env.use_hard_constraint:
        raise ValueError(
            "Training must use use_hard_constraint=False. "
            "The hard mask is reserved for evaluation."
        )

    if env.use_dimensionless:
        raise ValueError(
            "This experiment is configured for physical energy costs, "
            "so use_dimensionless must be False."
        )

    if not hasattr(env, "total_train_episodes"):
        raise AttributeError(
            "ConstrainedHAPPOEnvironment must accept total_train_episodes."
        )

    if int(env.total_train_episodes) != TOTAL_TRAIN_EPISODES:
        raise ValueError(
            "Environment total_train_episodes mismatch: "
            f"{env.total_train_episodes} != {TOTAL_TRAIN_EPISODES}"
        )

    if not hasattr(env, "H_bar_u"):
        raise AttributeError(
            "ConstrainedHAPPOEnvironment must define "
            "H_bar_u=floor(kappa*(T-1))/T."
        )

    # Detect the concrete implementation mistakes discussed during review.
    try:
        update_source = inspect.getsource(
            env._update_dual_variables_episode
        )
        compact = "".join(update_source.split())

        if "np.max(0.0," in compact:
            raise RuntimeError(
                "env_constrainedhappo.py still uses np.max(0.0, value). "
                "Replace it with max(0.0, value) or np.maximum(0.0, value)."
            )

        if "np.sqrt(self.episode_idx+1)" in compact:
            raise RuntimeError(
                "env_constrainedhappo.py still uses eta/sqrt(k+1). "
                "For the current Theorem 1 statement, use "
                "eta/sqrt(total_train_episodes)."
            )

        if "ho_ratio-self.kappa" in compact:
            raise RuntimeError(
                "env_constrainedhappo.py still computes C_H=ho_ratio-kappa. "
                "Use C_H=ho_ratio-H_bar_u instead."
            )
    except (OSError, TypeError):
        # Source inspection can fail in some packaged environments. The
        # runtime shape/history checks below still protect the experiment.
        pass


def print_physical_energy_scale(env) -> None:
    p_max = []
    p_bar = []

    for index, bs in enumerate(env.base_stations):
        p_max.append(
            lookup_entity_value(
                env.P_max,
                bs.bs_id,
                index,
                "P_max",
            )
        )
        p_bar.append(
            lookup_entity_value(
                env.P_bar,
                bs.bs_id,
                index,
                "P_bar",
            )
        )

    p_max_array = np.asarray(p_max, dtype=np.float64)
    p_bar_array = np.asarray(p_bar, dtype=np.float64)

    print(
        "[PHYSICAL ENERGY] "
        f"P_max={p_max_array.tolist()} W | "
        f"P_bar={p_bar_array.tolist()} W"
    )
    print(
        "[DUAL STEP] "
        f"beta_E={ETA_MU / np.sqrt(TOTAL_TRAIN_EPISODES):.8f} | "
        f"beta_H={ETA_NU / np.sqrt(TOTAL_TRAIN_EPISODES):.8f}"
    )


def save_dual_history(
    env,
    run_dir: Path,
    variant: str,
    kappa: float,
    train_seed: int,
) -> Path:
    path = run_dir / "dual_history.npz"

    mu_E = np.asarray(env.mu_E_b_history, dtype=np.float32)
    mu_H = np.asarray(env.nu_H_u_history, dtype=np.float32)
    C_E = np.asarray(env.C_E_b_history, dtype=np.float32)
    C_H = np.asarray(env.C_H_u_history, dtype=np.float32)

    lengths = {
        "mu_E": len(mu_E),
        "mu_H": len(mu_H),
        "C_E": len(C_E),
        "C_H": len(C_H),
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"Inconsistent dual-history lengths: {lengths}"
        )

    np.savez(
        path,
        mu_E_b_history=mu_E,
        nu_H_u_history=mu_H,
        C_E_b_history=C_E,
        C_H_u_history=C_H,
        completed_episodes=np.asarray(len(C_E), dtype=np.int32),
        variant=np.asarray(variant),
        kappa=np.asarray(kappa, dtype=np.float64),
        train_seed=np.asarray(train_seed, dtype=np.int32),
        total_train_episodes=np.asarray(
            TOTAL_TRAIN_EPISODES,
            dtype=np.int32,
        ),
        eta_mu=np.asarray(ETA_MU, dtype=np.float64),
        eta_nu=np.asarray(ETA_NU, dtype=np.float64),
        use_dimensionless=np.asarray(USE_DIMENSIONLESS),
    )

    return path


def save_training_log(
    run_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    save_csv(
        run_dir / "training_episode_log.csv",
        rows,
        [
            "variant",
            "kappa",
            "train_seed",
            "episode",
            "global_step",
            "episode_reward_mean",
            "on_ratio_mean",
            "handover_ratio_mean",
            "C_E_max",
            "C_H_max",
            "mu_E_mean_post_update",
            "mu_H_mean_post_update",
            "critic_loss_mean",
            "actor_ue_loss_mean",
            "actor_bs_loss_mean",
        ],
    )


def run_has_required_training_files(
    variant: str,
    kappa: float,
    train_seed: int,
) -> bool:
    run_dir = explicit_lagrangian_run_dir(
        variant,
        kappa,
        train_seed,
    )

    if not (run_dir / "dual_history.npz").exists():
        return False

    for episode in CHECKPOINT_EPISODES:
        matches = [
            path
            for path in (run_dir / "checkpoints").glob("*.pt")
            if checkpoint_episode_from_name(path) == episode
        ]
        if len(matches) != 1:
            return False

    return True


def train_one_lagrangian_run(
    variant: str,
    kappa: float,
    train_seed: int,
) -> None:
    run_dir = explicit_lagrangian_run_dir(
        variant,
        kappa,
        train_seed,
    )

    if TRAIN_FROM_SCRATCH and run_dir.exists():
        print(f"[TRAIN RESET] Removing old run: {run_dir}")
        shutil.rmtree(run_dir)

    if (
        not TRAIN_FROM_SCRATCH
        and run_has_required_training_files(
            variant,
            kappa,
            train_seed,
        )
    ):
        print(
            "[TRAIN SKIP] Complete run already exists: "
            f"{run_dir}"
        )
        return

    checkpoint_dir = run_dir / "checkpoints"
    ensure_dir(checkpoint_dir)

    print("\n" + "=" * 110)
    print(
        f"TRAIN {variant.upper()}-HAPPO | "
        f"kappa={kappa:.3f} | seed={train_seed} | "
        f"episodes={TOTAL_TRAIN_EPISODES}"
    )
    print("=" * 110)

    set_seed(train_seed)
    env = make_lagrangian_env(
        train_seed,
        variant,
        kappa,
        training=True,
    )
    validate_training_environment(env)
    print_physical_energy_scale(env)

    set_seed(train_seed)
    trainer = make_trainer(env)
    trainer.ue_actor.train()
    trainer.bs_actor.train()
    trainer.critic.train()

    global_step = 0
    episode_rows: List[Dict[str, object]] = []

    for episode_index in range(TOTAL_TRAIN_EPISODES):
        completed_episode = episode_index + 1
        local_obs, global_obs = env.reset()
        trainer.reset_rollout()

        episode_reward_sum = 0.0
        on_count = np.zeros(len(env.base_stations), dtype=np.float64)
        ho_count = np.zeros(len(env.users), dtype=np.float64)
        update_losses: Dict[str, List[float]] = {
            "critic": [],
            "actor_ue": [],
            "actor_bs": [],
        }

        for episode_step in range(STEPS_PER_EPISODE):
            global_step += 1

            (
                ue_actions,
                ue_logp_np,
                _ue_ent_np,
                ue_masks_np,
                bs_actions,
                bs_logp_np,
                _bs_ent_np,
                bs_obs_np,
                bs_masks_np,
                cand_lists,
                v_n,
            ) = trainer.select_actions(local_obs, global_obs)

            (
                next_local_obs,
                next_global_obs,
                info,
                done,
            ) = env.step_joint(
                ue_actions=ue_actions,
                bs_actions=bs_actions,
                cand_lists=cand_lists,
            )

            reward = float(info["global_reward"])
            episode_reward_sum += reward

            for bs_index, bs in enumerate(env.base_stations):
                on_count[bs_index] += float(
                    info["power_consumed"][bs.bs_id] > 0.0
                )

            for user_index, user in enumerate(env.users):
                ho_count[user_index] += float(
                    info["handover_u"][user.ue_id]
                )

            done_to_store = bool(
                done
                or episode_step == STEPS_PER_EPISODE - 1
            )

            with torch.no_grad():
                next_global_t = torch.as_tensor(
                    next_global_obs,
                    dtype=torch.float32,
                    device=trainer.device,
                ).unsqueeze(0)
                nv_n = float(
                    trainer.critic(next_global_t)
                    .squeeze(0)
                    .item()
                )

            trainer.store_step(
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
                nv_n=nv_n,
                done=done_to_store,
            )

            local_obs = next_local_obs
            global_obs = next_global_obs

            should_update = (
                (episode_step + 1) % UPDATE_INTERVAL == 0
                or done_to_store
            )

            if should_update:
                losses = trainer.update()
                for key in update_losses:
                    if key in losses:
                        update_losses[key].append(
                            float(losses[key])
                        )

            if (episode_step + 1) % 1000 == 0:
                print(
                    f"[TRAIN] Ep {completed_episode:2d}/"
                    f"{TOTAL_TRAIN_EPISODES} | "
                    f"Step {episode_step + 1:5d}/"
                    f"{STEPS_PER_EPISODE} | "
                    f"Global {global_step:7d} | "
                    f"AvgReward "
                    f"{episode_reward_sum / (episode_step + 1):.4f}"
                )

            if done and episode_step + 1 < STEPS_PER_EPISODE:
                raise RuntimeError(
                    "Training episode ended early: "
                    f"{episode_step + 1}/{STEPS_PER_EPISODE}"
                )

        expected_history_length = completed_episode
        actual_lengths = {
            "mu_E": len(env.mu_E_b_history),
            "mu_H": len(env.nu_H_u_history),
            "C_E": len(env.C_E_b_history),
            "C_H": len(env.C_H_u_history),
        }

        if any(
            length != expected_history_length
            for length in actual_lengths.values()
        ):
            raise RuntimeError(
                "Dual update was not executed exactly once at the "
                f"episode boundary. Expected {expected_history_length}, "
                f"got {actual_lengths}."
            )

        C_E_last = np.asarray(
            env.C_E_b_history[-1],
            dtype=np.float64,
        )
        C_H_last = np.asarray(
            env.C_H_u_history[-1],
            dtype=np.float64,
        )
        mu_E_last = np.asarray(
            env.mu_E_b_history[-1],
            dtype=np.float64,
        )
        mu_H_last = np.asarray(
            env.nu_H_u_history[-1],
            dtype=np.float64,
        )

        row = {
            "variant": variant,
            "kappa": kappa,
            "train_seed": train_seed,
            "episode": completed_episode,
            "global_step": global_step,
            "episode_reward_mean": (
                episode_reward_sum / STEPS_PER_EPISODE
            ),
            "on_ratio_mean": float(
                np.mean(on_count / STEPS_PER_EPISODE)
            ),
            "handover_ratio_mean": float(
                np.mean(ho_count / STEPS_PER_EPISODE)
            ),
            "C_E_max": float(np.max(C_E_last)),
            "C_H_max": float(np.max(C_H_last)),
            "mu_E_mean_post_update": float(
                np.mean(mu_E_last)
            ),
            "mu_H_mean_post_update": float(
                np.mean(mu_H_last)
            ),
            "critic_loss_mean": float(
                np.mean(update_losses["critic"])
            ) if update_losses["critic"] else float("nan"),
            "actor_ue_loss_mean": float(
                np.mean(update_losses["actor_ue"])
            ) if update_losses["actor_ue"] else float("nan"),
            "actor_bs_loss_mean": float(
                np.mean(update_losses["actor_bs"])
            ) if update_losses["actor_bs"] else float("nan"),
        }
        episode_rows.append(row)

        dual_path = save_dual_history(
            env,
            run_dir,
            variant,
            kappa,
            train_seed,
        )
        save_training_log(run_dir, episode_rows)

        print(
            f"[EP END] Ep {completed_episode:2d} | "
            f"ON={row['on_ratio_mean']:.4f} | "
            f"HO={row['handover_ratio_mean']:.5f} | "
            f"max C_E={row['C_E_max']:.6f} | "
            f"max C_H={row['C_H_max']:.6f} | "
            f"mean mu_E={row['mu_E_mean_post_update']:.6f} | "
            f"mean mu_H={row['mu_H_mean_post_update']:.6f}"
        )

        if completed_episode in CHECKPOINT_EPISODES:
            checkpoint_path = (
                checkpoint_dir
                / (
                    f"checkpoint_step_{global_step}_"
                    f"episode_{completed_episode}_end.pt"
                )
            )
            trainer.save_model(
                str(checkpoint_path),
                save_optim=SAVE_OPTIMIZER_IN_CHECKPOINTS,
            )
            print(f"[CHECKPOINT] {checkpoint_path}")

        print(f"[DUAL HISTORY] {dual_path}")

    final_model_path = run_dir / "model_final.pt"
    trainer.save_model(
        str(final_model_path),
        save_optim=SAVE_OPTIMIZER_IN_CHECKPOINTS,
    )

    del trainer
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_all_lagrangian_runs() -> None:
    for variant in VARIANTS:
        if variant not in {"pf", "jensen"}:
            raise ValueError(f"Unsupported variant: {variant}")

        for kappa in KAPPAS:
            for train_seed in TRAIN_SEEDS:
                train_one_lagrangian_run(
                    variant,
                    kappa,
                    train_seed,
                )


# =============================================================================
# 6. Existing-file resolution
# =============================================================================
def explicit_lagrangian_run_dir(
    variant: str,
    kappa: float,
    train_seed: int,
) -> Path:
    template = LAGRANGIAN_RUN_DIR_TEMPLATES[variant]

    return PROJECT_ROOT / template.format(
        kappa3=f"{kappa:.3f}",
        kappa2=f"{kappa:.2f}",
        train_seed=train_seed,
        variant=variant,
    )


def path_matches_run(
    path: Path,
    variant: str,
    kappa: float,
    train_seed: int,
) -> bool:
    text = str(path).lower()

    if variant.lower() not in text:
        return False

    if not any(
        f"kappa_{tag}" in text
        for tag in kappa_tags(kappa)
    ):
        return False

    seed_tokens = [
        f"seed_{train_seed}",
        f"train_seed_{train_seed}",
    ]

    return any(token in text for token in seed_tokens)


def resolve_dual_history_path(
    variant: str,
    kappa: float,
    train_seed: int,
) -> Path:
    explicit = (
        explicit_lagrangian_run_dir(
            variant,
            kappa,
            train_seed,
        )
        / "dual_history.npz"
    )

    if explicit.exists():
        return explicit

    results_root = PROJECT_ROOT / "results"

    candidates = [
        path
        for path in results_root.rglob("dual_history.npz")
        if path_matches_run(
            path,
            variant,
            kappa,
            train_seed,
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            "dual_history.npz not found.\n"
            f"variant={variant}, kappa={kappa}, train_seed={train_seed}\n"
            f"Expected first at: {explicit}"
        )

    raise RuntimeError(
        "Multiple dual-history files matched. "
        "Set LAGRANGIAN_RUN_DIR_TEMPLATES explicitly:\n"
        + "\n".join(str(path) for path in candidates)
    )


CHECKPOINT_REGEX = re.compile(
    r".*episode_(\d+)(?:_end)?\.pt$",
    re.IGNORECASE,
)


def checkpoint_episode_from_name(path: Path) -> Optional[int]:
    match = CHECKPOINT_REGEX.match(path.name)

    if match is None:
        return None

    return int(match.group(1))


def resolve_lagrangian_checkpoint(
    variant: str,
    kappa: float,
    train_seed: int,
    episode: int,
) -> Path:
    run_dir = explicit_lagrangian_run_dir(
        variant,
        kappa,
        train_seed,
    )

    explicit_candidates = [
        path
        for path in run_dir.rglob("*.pt")
        if checkpoint_episode_from_name(path) == episode
    ]

    if len(explicit_candidates) == 1:
        return explicit_candidates[0]

    if len(explicit_candidates) > 1:
        end_candidates = [
            path
            for path in explicit_candidates
            if "_end.pt" in path.name
        ]

        if len(end_candidates) == 1:
            return end_candidates[0]

        raise RuntimeError(
            f"Multiple episode-{episode} checkpoints in {run_dir}:\n"
            + "\n".join(str(path) for path in explicit_candidates)
        )

    results_root = PROJECT_ROOT / "results"

    fallback = [
        path
        for path in results_root.rglob("*.pt")
        if checkpoint_episode_from_name(path) == episode
        and path_matches_run(
            path,
            variant,
            kappa,
            train_seed,
        )
    ]

    if len(fallback) == 1:
        return fallback[0]

    if len(fallback) == 0:
        raise FileNotFoundError(
            f"Episode-{episode} checkpoint not found.\n"
            f"variant={variant}, kappa={kappa}, train_seed={train_seed}\n"
            f"Searched first under: {run_dir}"
        )

    raise RuntimeError(
        f"Multiple episode-{episode} checkpoints matched. "
        "Set LAGRANGIAN_RUN_DIR_TEMPLATES explicitly:\n"
        + "\n".join(str(path) for path in fallback)
    )


def resolve_helymarl_model(
    kappa: float,
    train_seed: int,
) -> Path:
    root = PROJECT_ROOT / HELYMARL_MODEL_ROOT

    candidates: List[Path] = []

    for tag in kappa_tags(kappa):
        path = (
            root
            / f"HAPPO_kappa_{tag}_seed_{train_seed}"
            / "model.pt"
        )

        if path.exists():
            candidates.append(path)

    unique = list(dict.fromkeys(candidates))

    if len(unique) == 1:
        return unique[0]

    if len(unique) == 0:
        raise FileNotFoundError(
            "HeLyMARL K=10 model not found. Tried:\n"
            + "\n".join(
                str(
                    root
                    / f"HAPPO_kappa_{tag}_seed_{train_seed}"
                    / "model.pt"
                )
                for tag in kappa_tags(kappa)
            )
        )

    raise RuntimeError(
        "Multiple HeLyMARL models matched:\n"
        + "\n".join(str(path) for path in unique)
    )


# =============================================================================
# 7. Inter-episode analysis: no evaluation required
# =============================================================================
def load_dual_history(path: Path) -> Dict[str, np.ndarray]:
    required = [
        "mu_E_b_history",
        "nu_H_u_history",
        "C_E_b_history",
        "C_H_u_history",
    ]

    with np.load(path, allow_pickle=True) as data:
        missing = [
            key
            for key in required
            if key not in data.files
        ]

        if missing:
            raise KeyError(
                f"Missing keys in {path}: {missing}\n"
                f"Available keys={data.files}"
            )

        result = {
            key: np.asarray(
                data[key],
                dtype=np.float64,
            )
            for key in required
        }

    lengths = {
        key: value.shape[0]
        for key, value in result.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Inconsistent episode lengths in {path}: {lengths}"
        )

    return result


def running_signed_violation(
    constraint_history: np.ndarray,
) -> np.ndarray:
    """
    constraint_history: [K, number_of_constraints]

    Returns:
        max_i [ (1/k) sum_{j=1}^k C_i(j) ]^+
        for k = 1,...,K.
    """
    episode_count = constraint_history.shape[0]

    cumulative_mean = (
        np.cumsum(
            constraint_history,
            axis=0,
        )
        / np.arange(
            1,
            episode_count + 1,
            dtype=np.float64,
        )[:, None]
    )

    return np.max(
        np.maximum(
            cumulative_mean,
            0.0,
        ),
        axis=1,
    )


def analyze_inter_episode_one_setting(
    variant: str,
    kappa: float,
) -> None:
    per_train_seed: List[Dict[str, np.ndarray]] = []
    source_paths: List[Path] = []

    for train_seed in TRAIN_SEEDS:
        path = resolve_dual_history_path(
            variant,
            kappa,
            train_seed,
        )
        source_paths.append(path)

        data = load_dual_history(path)

        per_train_seed.append(
            {
                "mu_E_mean": np.mean(
                    data["mu_E_b_history"],
                    axis=1,
                ),
                "mu_H_mean": np.mean(
                    data["nu_H_u_history"],
                    axis=1,
                ),
                "C_E_max_raw": np.max(
                    data["C_E_b_history"],
                    axis=1,
                ),
                "C_H_max_raw": np.max(
                    data["C_H_u_history"],
                    axis=1,
                ),
                "C_E_running_signed": running_signed_violation(
                    data["C_E_b_history"]
                ),
                "C_H_running_signed": running_signed_violation(
                    data["C_H_u_history"]
                ),
            }
        )

    common_k = min(
        len(run["mu_E_mean"])
        for run in per_train_seed
    )

    output_dir = (
        PROJECT_ROOT
        / OUTPUT_ROOT
        / "inter_episode"
        / variant
        / f"kappa_{kappa:.3f}"
    )
    ensure_dir(output_dir)

    metric_names = [
        "mu_E_mean",
        "mu_H_mean",
        "C_E_max_raw",
        "C_H_max_raw",
        "C_E_running_signed",
        "C_H_running_signed",
    ]

    stacked = {
        metric: np.stack(
            [
                run[metric][:common_k]
                for run in per_train_seed
            ],
            axis=0,
        )
        for metric in metric_names
    }

    rows: List[Dict[str, object]] = []

    for train_index, train_seed in enumerate(TRAIN_SEEDS):
        for episode_index in range(common_k):
            row: Dict[str, object] = {
                "variant": variant,
                "kappa": kappa,
                "train_seed": train_seed,
                "episode": episode_index + 1,
                "dual_history_path": str(
                    source_paths[train_index]
                ),
            }

            for metric in metric_names:
                row[metric] = float(
                    stacked[metric][
                        train_index,
                        episode_index,
                    ]
                )

            rows.append(row)

    save_csv(
        output_dir / "inter_episode_per_train_seed.csv",
        rows,
        [
            "variant",
            "kappa",
            "train_seed",
            "episode",
            *metric_names,
            "dual_history_path",
        ],
    )

    summary_rows: List[Dict[str, object]] = []

    for episode_index in range(common_k):
        row = {
            "variant": variant,
            "kappa": kappa,
            "episode": episode_index + 1,
        }

        for metric in metric_names:
            values = stacked[metric][:, episode_index]
            row[f"{metric}_mean"] = float(
                np.mean(values)
            )
            row[f"{metric}_std"] = float(
                std_across_train_seeds(values, axis=0)
            )

        summary_rows.append(row)

    summary_fields = [
        "variant",
        "kappa",
        "episode",
    ]

    for metric in metric_names:
        summary_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
            ]
        )

    save_csv(
        output_dir / "inter_episode_summary.csv",
        summary_rows,
        summary_fields,
    )

    plot_inter_episode(
        variant=variant,
        kappa=kappa,
        stacked=stacked,
        output_dir=output_dir,
    )


def plot_curve_with_std(
    x: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    save_path: Path,
    reference: Optional[np.ndarray] = None,
    reference_label: Optional[str] = None,
    second_values: Optional[np.ndarray] = None,
    second_label: Optional[str] = None,
) -> None:
    mean = np.mean(values, axis=0)
    std = std_across_train_seeds(values, axis=0)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    ax.plot(
        x,
        mean,
        linewidth=2.0,
        label="Mean across train seeds",
    )
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.18,
        label=r"$\pm$ std",
    )

    if second_values is not None:
        second_mean = np.mean(
            second_values,
            axis=0,
        )
        ax.plot(
            x,
            second_mean,
            linestyle=":",
            linewidth=1.5,
            label=second_label,
        )

    if reference is not None:
        ax.plot(
            x,
            reference,
            linestyle="--",
            linewidth=1.5,
            label=reference_label,
        )

    if x[-1] >= 10:
        ax.axvline(
            10,
            linestyle="-.",
            linewidth=1.1,
            label=r"$K=10$",
        )

    ax.set_xlabel("Training episode $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_inverse_sqrt_reference(
    curve: np.ndarray,
) -> np.ndarray:
    """
    Scale 1/sqrt(k) to the first positive point in the empirical mean curve.
    It is a visual rate reference, not a fitted theorem constant.
    """
    mean_curve = np.mean(curve, axis=0)
    episodes = np.arange(
        1,
        len(mean_curve) + 1,
        dtype=np.float64,
    )

    positive_indices = np.flatnonzero(
        mean_curve > 0.0
    )

    if positive_indices.size == 0:
        return np.zeros_like(mean_curve)

    index = int(positive_indices[0])
    scale = mean_curve[index] * np.sqrt(
        episodes[index]
    )

    return scale / np.sqrt(episodes)


def plot_inter_episode(
    variant: str,
    kappa: float,
    stacked: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    episode_count = stacked["mu_E_mean"].shape[1]
    x = np.arange(1, episode_count + 1)

    plot_curve_with_std(
        x=x,
        values=stacked["mu_E_mean"],
        ylabel=r"Post-update mean multiplier $\overline{\mu}_E(k+1)$",
        title=(
            f"{variant.upper()}-HAPPO energy multiplier, "
            rf"$\kappa={kappa}$"
        ),
        save_path=output_dir / "mu_E_convergence.png",
    )

    plot_curve_with_std(
        x=x,
        values=stacked["mu_H_mean"],
        ylabel=r"Post-update mean multiplier $\overline{\mu}_H(k+1)$",
        title=(
            f"{variant.upper()}-HAPPO handover multiplier, "
            rf"$\kappa={kappa}$"
        ),
        save_path=output_dir / "mu_H_convergence.png",
    )

    plot_curve_with_std(
        x=x,
        values=stacked["C_E_running_signed"],
        second_values=stacked["C_E_max_raw"],
        second_label=r"Raw $\max_b C_E^b(k)$",
        reference=make_inverse_sqrt_reference(
            stacked["C_E_running_signed"]
        ),
        reference_label=r"Scaled $1/\sqrt{k}$ guide",
        ylabel="Energy constraint violation",
        title=(
            f"{variant.upper()}-HAPPO inter-episode energy violation, "
            rf"$\kappa={kappa}$"
        ),
        save_path=output_dir / "C_E_inter_episode.png",
    )

    plot_curve_with_std(
        x=x,
        values=stacked["C_H_running_signed"],
        second_values=stacked["C_H_max_raw"],
        second_label=r"Raw $\max_u C_H^u(k)$",
        reference=make_inverse_sqrt_reference(
            stacked["C_H_running_signed"]
        ),
        reference_label=r"Scaled $1/\sqrt{k}$ guide",
        ylabel="Handover constraint violation",
        title=(
            f"{variant.upper()}-HAPPO inter-episode handover violation, "
            rf"$\kappa={kappa}$"
        ),
        save_path=output_dir / "C_H_inter_episode.png",
    )


# =============================================================================
# 8. Evaluation-only trajectory generation
# =============================================================================
def select_actions_compat(
    trainer: HAPPOTrainer,
    env,
    local_obs,
    global_obs,
):
    """
    Handle both trainer versions:
      select_actions(local_obs, global_obs)
      select_actions(local_obs, global_obs, env=..., deterministic=...)
    """
    signature = inspect.signature(
        trainer.select_actions
    )

    kwargs = {}

    if "env" in signature.parameters:
        kwargs["env"] = env

    if "deterministic" in signature.parameters:
        kwargs["deterministic"] = DETERMINISTIC

    return trainer.select_actions(
        local_obs,
        global_obs,
        **kwargs,
    )


def extract_action_outputs(action_output):
    if not isinstance(action_output, tuple):
        raise TypeError(
            "trainer.select_actions must return a tuple."
        )

    if len(action_output) < 10:
        raise ValueError(
            "Unexpected select_actions output length: "
            f"{len(action_output)}"
        )

    ue_actions = action_output[0]
    bs_actions = action_output[4]
    candidate_lists = action_output[9]

    return ue_actions, bs_actions, candidate_lists


def extract_dict_or_array(
    values: object,
    entity_ids: Sequence[int],
    name: str,
) -> np.ndarray:
    if isinstance(values, Mapping):
        result = []

        for entity_id in entity_ids:
            if entity_id in values:
                result.append(
                    float(values[entity_id])
                )
            elif str(entity_id) in values:
                result.append(
                    float(values[str(entity_id)])
                )
            else:
                raise KeyError(
                    f"{name} has no value for entity {entity_id}."
                )

        return np.asarray(
            result,
            dtype=np.float64,
        )

    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size != len(entity_ids):
        raise ValueError(
            f"{name} length mismatch: "
            f"got {array.size}, expected {len(entity_ids)}."
        )

    return array


def get_energy_constants(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      e_bar[b] = per-slot activation cost
      E_bar[b] = per-slot budget allocation

    The constrained training code uses P_max and P_bar when
    USE_DIMENSIONLESS=False, so this evaluator requires the same quantities.
    """
    if not hasattr(env, "P_max"):
        raise AttributeError(
            "Environment has no P_max. "
            "Cannot calculate the requested physical-energy D_E exactly."
        )

    if not hasattr(env, "P_bar"):
        raise AttributeError(
            "Environment has no P_bar. "
            "Cannot calculate the requested physical-energy D_E exactly."
        )

    e_bar = []
    E_bar = []

    for index, bs in enumerate(env.base_stations):
        e_bar.append(
            lookup_entity_value(
                env.P_max,
                bs.bs_id,
                index,
                "P_max",
            )
        )

        E_bar.append(
            lookup_entity_value(
                env.P_bar,
                bs.bs_id,
                index,
                "P_bar",
            )
        )

    e_bar_array = np.asarray(
        e_bar,
        dtype=np.float64,
    )
    E_bar_array = np.asarray(
        E_bar,
        dtype=np.float64,
    )

    expected_E_bar = (
        POWER_BUDGET_RATIO
        * e_bar_array
    )

    if not np.allclose(
        E_bar_array,
        expected_E_bar,
        rtol=1e-6,
        atol=1e-12,
    ):
        raise ValueError(
            "P_bar is inconsistent with "
            "POWER_BUDGET_RATIO * P_max.\n"
            f"P_bar={E_bar_array}\n"
            f"expected={expected_E_bar}"
        )

    return e_bar_array, E_bar_array


@torch.no_grad()
def evaluate_trajectory(
    model_path: Path,
    env,
    trainer: HAPPOTrainer,
    eval_seed: int,
) -> Dict[str, np.ndarray]:
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    trainer.load_model(str(model_path))

    trainer.ue_actor.eval()
    trainer.bs_actor.eval()
    trainer.critic.eval()

    # Match the existing evaluation protocol.
    set_seed(eval_seed)
    local_obs, global_obs = env.reset()

    bs_ids = [
        bs.bs_id
        for bs in env.base_stations
    ]
    ue_ids = [
        user.ue_id
        for user in env.users
    ]

    y_history: List[np.ndarray] = []
    h_history: List[np.ndarray] = []

    for step in range(STEPS_PER_EPISODE):
        action_output = select_actions_compat(
            trainer,
            env,
            local_obs,
            global_obs,
        )

        (
            ue_actions,
            bs_actions,
            candidate_lists,
        ) = extract_action_outputs(
            action_output
        )

        (
            next_local_obs,
            next_global_obs,
            info,
            done,
        ) = env.step_joint(
            ue_actions=ue_actions,
            bs_actions=bs_actions,
            cand_lists=candidate_lists,
        )

        power = extract_dict_or_array(
            info["power_consumed"],
            bs_ids,
            "power_consumed",
        )

        handover = extract_dict_or_array(
            info["handover_u"],
            ue_ids,
            "handover_u",
        )

        y_history.append(
            (power > 0.0).astype(np.float64)
        )
        h_history.append(
            handover.astype(np.float64)
        )

        local_obs = next_local_obs
        global_obs = next_global_obs

        if done and step + 1 < STEPS_PER_EPISODE:
            raise RuntimeError(
                "Evaluation episode ended early: "
                f"{step + 1}/{STEPS_PER_EPISODE}"
            )

    y = np.stack(
        y_history,
        axis=0,
    )
    h = np.stack(
        h_history,
        axis=0,
    )

    if y.shape != (
        STEPS_PER_EPISODE,
        len(bs_ids),
    ):
        raise ValueError(
            f"Unexpected y shape: {y.shape}"
        )

    if h.shape != (
        STEPS_PER_EPISODE,
        len(ue_ids),
    ):
        raise ValueError(
            f"Unexpected h shape: {h.shape}"
        )

    e_bar, E_bar = get_energy_constants(env)

    return {
        "y": y,
        "h": h,
        "e_bar": e_bar,
        "E_bar": E_bar,
    }


def trajectory_file(
    method: str,
    kappa: float,
    checkpoint: str,
    train_seed: int,
    eval_seed: int,
) -> Path:
    return (
        PROJECT_ROOT
        / OUTPUT_ROOT
        / "raw_trajectories"
        / method
        / f"kappa_{kappa:.3f}"
        / checkpoint
        / f"train_seed_{train_seed}"
        / f"eval_seed_{eval_seed}.npz"
    )


def load_or_evaluate_lagrangian(
    variant: str,
    kappa: float,
    episode: int,
    train_seed: int,
    eval_seed: int,
) -> Dict[str, np.ndarray]:
    path = trajectory_file(
        method=variant,
        kappa=kappa,
        checkpoint=f"episode_{episode}",
        train_seed=train_seed,
        eval_seed=eval_seed,
    )

    if path.exists() and not OVERWRITE_EVALUATION:
        with np.load(path, allow_pickle=True) as data:
            return {
                key: np.asarray(
                    data[key],
                    dtype=np.float64,
                )
                for key in [
                    "y",
                    "h",
                    "e_bar",
                    "E_bar",
                ]
            }

    checkpoint_path = resolve_lagrangian_checkpoint(
        variant,
        kappa,
        train_seed,
        episode,
    )

    print(
        f"[EVAL] {variant.upper()} | "
        f"kappa={kappa} | episode={episode} | "
        f"train={train_seed} | eval={eval_seed}\n"
        f"       model={checkpoint_path}"
    )

    set_seed(eval_seed)
    env = make_lagrangian_env(
        eval_seed,
        variant,
        kappa,
    )

    set_seed(eval_seed)
    trainer = make_trainer(env)

    result = evaluate_trajectory(
        model_path=checkpoint_path,
        env=env,
        trainer=trainer,
        eval_seed=eval_seed,
    )

    ensure_dir(path.parent)
    np.savez(
        path,
        **result,
        model_path=np.asarray(
            str(checkpoint_path)
        ),
    )

    del trainer
    del env
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def load_or_evaluate_helymarl(
    kappa: float,
    train_seed: int,
    eval_seed: int,
) -> Dict[str, np.ndarray]:
    path = trajectory_file(
        method="helymarl",
        kappa=kappa,
        checkpoint="K10_reference",
        train_seed=train_seed,
        eval_seed=eval_seed,
    )

    if path.exists() and not OVERWRITE_EVALUATION:
        with np.load(path, allow_pickle=True) as data:
            return {
                key: np.asarray(
                    data[key],
                    dtype=np.float64,
                )
                for key in [
                    "y",
                    "h",
                    "e_bar",
                    "E_bar",
                ]
            }

    model_path = resolve_helymarl_model(
        kappa,
        train_seed,
    )

    print(
        f"[EVAL] HeLyMARL K=10 | "
        f"kappa={kappa} | "
        f"train={train_seed} | eval={eval_seed}\n"
        f"       model={model_path}"
    )

    set_seed(eval_seed)
    env = make_helymarl_env(
        eval_seed,
        kappa,
    )

    set_seed(eval_seed)
    trainer = make_trainer(env)

    result = evaluate_trajectory(
        model_path=model_path,
        env=env,
        trainer=trainer,
        eval_seed=eval_seed,
    )

    ensure_dir(path.parent)
    np.savez(
        path,
        **result,
        model_path=np.asarray(
            str(model_path)
        ),
    )

    del trainer
    del env
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =============================================================================
# 9. Exact D_E, D_H, and utilization calculation
# =============================================================================
def compute_intra_episode_metrics(
    trajectories: Sequence[Dict[str, np.ndarray]],
    kappa: float,
) -> Dict[str, float]:
    """
    The expectation over evaluation seeds is taken BEFORE max_{b,tau}
    and max_{u,tau}, exactly matching the advisor's formulas.
    """
    if len(trajectories) != len(EVAL_SEEDS):
        raise ValueError(
            f"Expected {len(EVAL_SEEDS)} evaluation trajectories, "
            f"got {len(trajectories)}."
        )

    y = np.stack(
        [
            trajectory["y"]
            for trajectory in trajectories
        ],
        axis=0,
    )  # [S,T,B]

    h = np.stack(
        [
            trajectory["h"]
            for trajectory in trajectories
        ],
        axis=0,
    )  # [S,T,U]

    e_bar = trajectories[0]["e_bar"]
    E_bar = trajectories[0]["E_bar"]

    for trajectory in trajectories[1:]:
        if not np.allclose(
            trajectory["e_bar"],
            e_bar,
        ):
            raise ValueError(
                "e_bar differs across evaluation seeds."
            )

        if not np.allclose(
            trajectory["E_bar"],
            E_bar,
        ):
            raise ValueError(
                "E_bar differs across evaluation seeds."
            )

    S, T, B = y.shape
    _, T_h, U = h.shape

    if T != STEPS_PER_EPISODE or T_h != T:
        raise ValueError(
            f"Unexpected trajectory horizon: y={T}, h={T_h}"
        )

    # -----------------------------------------------------------------
    # Energy:
    # D_E = max_b max_tau E[(sum_{t=0}^tau e_bar_b y_b(t)
    #                        - (tau+1) E_bar_b)^+]
    # -----------------------------------------------------------------
    slot_energy = (
        y
        * e_bar[None, None, :]
    )

    cumulative_energy = np.cumsum(
        slot_energy,
        axis=1,
    )

    energy_allocation = (
        np.arange(
            1,
            T + 1,
            dtype=np.float64,
        )[:, None]
        * E_bar[None, :]
    )

    energy_positive_deficit = np.maximum(
        cumulative_energy
        - energy_allocation[None, :, :],
        0.0,
    )

    expected_energy_deficit = np.mean(
        energy_positive_deficit,
        axis=0,
    )  # [T,B]

    D_E = float(
        np.max(
            expected_energy_deficit
        )
    )

    tau_E_index, bs_E_index = np.unravel_index(
        int(
            np.argmax(
                expected_energy_deficit
            )
        ),
        expected_energy_deficit.shape,
    )

    E_max = (
        float(T)
        * E_bar
    )

    energy_used = np.sum(
        slot_energy,
        axis=1,
    )  # [S,B]

    energy_utilization_by_bs = np.mean(
        energy_used
        / E_max[None, :],
        axis=0,
    )

    terminal_on_ratio = float(
        np.mean(y)
    )

    # -----------------------------------------------------------------
    # Handover:
    # D_H = max_u max_{1<=tau<=T-1}
    #       E[(sum_{t=1}^tau h_u(t) - tau H_bar_u)^+]
    #
    # Paper definition:
    #   H_max = floor(kappa * (T-1))
    #   H_bar = H_max / T
    # -----------------------------------------------------------------
    H_max_scalar = int(
        math.floor(
            kappa
            * (T - 1)
        )
    )

    if H_max_scalar <= 0:
        raise ValueError(
            f"Non-positive handover budget: H_max={H_max_scalar}"
        )

    H_bar_scalar = (
        float(H_max_scalar)
        / float(T)
    )

    h_from_t1 = h[:, 1:, :]

    cumulative_handover = np.cumsum(
        h_from_t1,
        axis=1,
    )

    handover_allocation = (
        np.arange(
            1,
            T,
            dtype=np.float64,
        )[:, None]
        * H_bar_scalar
    )

    handover_positive_deficit = np.maximum(
        cumulative_handover
        - handover_allocation[None, :, :],
        0.0,
    )

    expected_handover_deficit = np.mean(
        handover_positive_deficit,
        axis=0,
    )  # [T-1,U]

    D_H = float(
        np.max(
            expected_handover_deficit
        )
    )

    tau_H_index, user_H_index = np.unravel_index(
        int(
            np.argmax(
                expected_handover_deficit
            )
        ),
        expected_handover_deficit.shape,
    )

    handover_used = np.sum(
        h_from_t1,
        axis=1,
    )  # [S,U]

    handover_utilization_by_user = np.mean(
        handover_used
        / float(H_max_scalar),
        axis=0,
    )

    terminal_handover_ratio = float(
        np.mean(
            np.sum(
                h_from_t1,
                axis=1,
            )
            / float(T),
        )
    )

    return {
        "D_E": D_E,
        "D_E_normalized_by_budget": float(
            D_E
            / float(
                np.mean(E_max)
            )
        ),
        "D_E_bs_index_zero_based": int(
            bs_E_index
        ),
        "D_E_tau": int(
            tau_E_index
        ),
        "energy_utilization_mean": float(
            np.mean(
                energy_utilization_by_bs
            )
        ),
        "energy_utilization_min_bs": float(
            np.min(
                energy_utilization_by_bs
            )
        ),
        "energy_utilization_max_bs": float(
            np.max(
                energy_utilization_by_bs
            )
        ),
        "terminal_on_ratio": terminal_on_ratio,
        "D_H": D_H,
        "D_H_normalized_by_budget": float(
            D_H
            / float(H_max_scalar)
        ),
        "D_H_user_index_zero_based": int(
            user_H_index
        ),
        # index 0 corresponds to tau=1
        "D_H_tau": int(
            tau_H_index + 1
        ),
        "handover_utilization_mean": float(
            np.mean(
                handover_utilization_by_user
            )
        ),
        "handover_utilization_min_user": float(
            np.min(
                handover_utilization_by_user
            )
        ),
        "handover_utilization_max_user": float(
            np.max(
                handover_utilization_by_user
            )
        ),
        "terminal_handover_ratio": terminal_handover_ratio,
        "H_max": H_max_scalar,
        "H_bar": H_bar_scalar,
    }


INTRA_METRIC_FIELDS = [
    "D_E",
    "D_E_normalized_by_budget",
    "D_E_bs_index_zero_based",
    "D_E_tau",
    "energy_utilization_mean",
    "energy_utilization_min_bs",
    "energy_utilization_max_bs",
    "terminal_on_ratio",
    "D_H",
    "D_H_normalized_by_budget",
    "D_H_user_index_zero_based",
    "D_H_tau",
    "handover_utilization_mean",
    "handover_utilization_min_user",
    "handover_utilization_max_user",
    "terminal_handover_ratio",
    "H_max",
    "H_bar",
]


# =============================================================================
# 10. Intra-episode experiment and summary
# =============================================================================
def evaluate_helymarl_reference(
    kappa: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for train_seed in TRAIN_SEEDS:
        trajectories = [
            load_or_evaluate_helymarl(
                kappa=kappa,
                train_seed=train_seed,
                eval_seed=eval_seed,
            )
            for eval_seed in EVAL_SEEDS
        ]

        metrics = compute_intra_episode_metrics(
            trajectories,
            kappa,
        )

        rows.append(
            {
                "method": "HeLyMARL",
                "kappa": kappa,
                "checkpoint_episode": 10,
                "train_seed": train_seed,
                **metrics,
            }
        )

    return rows


def evaluate_lagrangian_checkpoints(
    variant: str,
    kappa: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for episode in CHECKPOINT_EPISODES:
        for train_seed in TRAIN_SEEDS:
            trajectories = [
                load_or_evaluate_lagrangian(
                    variant=variant,
                    kappa=kappa,
                    episode=episode,
                    train_seed=train_seed,
                    eval_seed=eval_seed,
                )
                for eval_seed in EVAL_SEEDS
            ]

            metrics = compute_intra_episode_metrics(
                trajectories,
                kappa,
            )

            rows.append(
                {
                    "method": (
                        "Jensen-HAPPO"
                        if variant == "jensen"
                        else "PF-HAPPO"
                    ),
                    "variant": variant,
                    "kappa": kappa,
                    "checkpoint_episode": episode,
                    "train_seed": train_seed,
                    **metrics,
                }
            )

    return rows


def aggregate_across_train_seeds(
    rows: Sequence[Mapping[str, object]],
    group_fields: Sequence[str],
) -> List[Dict[str, object]]:
    groups: Dict[
        Tuple[object, ...],
        List[Mapping[str, object]],
    ] = {}

    for row in rows:
        key = tuple(
            row[field]
            for field in group_fields
        )
        groups.setdefault(
            key,
            [],
        ).append(row)

    numeric_metrics = [
        "D_E",
        "D_E_normalized_by_budget",
        "energy_utilization_mean",
        "terminal_on_ratio",
        "D_H",
        "D_H_normalized_by_budget",
        "handover_utilization_mean",
        "terminal_handover_ratio",
    ]

    summary: List[Dict[str, object]] = []

    for key, group_rows in groups.items():
        if len(group_rows) != len(TRAIN_SEEDS):
            raise ValueError(
                f"Group {key} has {len(group_rows)} train seeds, "
                f"expected {len(TRAIN_SEEDS)}."
            )

        result = {
            field: value
            for field, value in zip(
                group_fields,
                key,
            )
        }

        result["n_train_seeds"] = len(
            group_rows
        )
        result["n_eval_seeds_per_model"] = len(
            EVAL_SEEDS
        )

        for metric in numeric_metrics:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in group_rows
                ],
                dtype=np.float64,
            )

            result[f"{metric}_mean"] = float(
                np.mean(values)
            )
            result[f"{metric}_std"] = float(
                std_across_train_seeds(values, axis=0)
            )

        summary.append(result)

    summary.sort(
        key=lambda row: tuple(
            row[field]
            for field in group_fields
        )
    )

    return summary


def run_intra_episode_experiment(
    variant: str,
    kappa: float,
) -> None:
    output_dir = (
        PROJECT_ROOT
        / OUTPUT_ROOT
        / "intra_episode"
        / variant
        / f"kappa_{kappa:.3f}"
    )
    ensure_dir(output_dir)

    lagrangian_rows = evaluate_lagrangian_checkpoints(
        variant,
        kappa,
    )

    helymarl_rows = evaluate_helymarl_reference(
        kappa,
    )

    all_rows = [
        *lagrangian_rows,
        *helymarl_rows,
    ]

    per_seed_fields = [
        "method",
        "variant",
        "kappa",
        "checkpoint_episode",
        "train_seed",
        *INTRA_METRIC_FIELDS,
    ]

    save_csv(
        output_dir / "intra_episode_per_train_seed.csv",
        all_rows,
        per_seed_fields,
    )

    lagrangian_summary = aggregate_across_train_seeds(
        lagrangian_rows,
        group_fields=[
            "method",
            "kappa",
            "checkpoint_episode",
        ],
    )

    helymarl_summary = aggregate_across_train_seeds(
        helymarl_rows,
        group_fields=[
            "method",
            "kappa",
            "checkpoint_episode",
        ],
    )

    summary = [
        *lagrangian_summary,
        *helymarl_summary,
    ]

    summary_fields = sorted(
        {
            key
            for row in summary
            for key in row.keys()
        }
    )

    preferred_prefix = [
        "method",
        "kappa",
        "checkpoint_episode",
        "n_train_seeds",
        "n_eval_seeds_per_model",
    ]

    summary_fields = preferred_prefix + [
        field
        for field in summary_fields
        if field not in preferred_prefix
    ]

    save_csv(
        output_dir / "intra_episode_summary.csv",
        summary,
        summary_fields,
    )

    plot_intra_episode_metrics(
        variant=variant,
        kappa=kappa,
        lagrangian_summary=lagrangian_summary,
        helymarl_summary=helymarl_summary,
        output_dir=output_dir,
    )


def get_summary_row(
    rows: Sequence[Mapping[str, object]],
    episode: int,
) -> Mapping[str, object]:
    matches = [
        row
        for row in rows
        if int(
            row["checkpoint_episode"]
        ) == episode
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one summary row for episode {episode}, "
            f"found {len(matches)}."
        )

    return matches[0]


def plot_checkpoint_metric(
    variant: str,
    kappa: float,
    metric: str,
    ylabel: str,
    lagrangian_summary: Sequence[Mapping[str, object]],
    helymarl_summary: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    x = np.asarray(
        CHECKPOINT_EPISODES,
        dtype=np.int32,
    )

    y = np.asarray(
        [
            float(
                get_summary_row(
                    lagrangian_summary,
                    episode,
                )[f"{metric}_mean"]
            )
            for episode in CHECKPOINT_EPISODES
        ],
        dtype=np.float64,
    )

    y_std = np.asarray(
        [
            float(
                get_summary_row(
                    lagrangian_summary,
                    episode,
                )[f"{metric}_std"]
            )
            for episode in CHECKPOINT_EPISODES
        ],
        dtype=np.float64,
    )

    reference = helymarl_summary[0]
    ref_mean = float(
        reference[f"{metric}_mean"]
    )
    ref_std = float(
        reference[f"{metric}_std"]
    )

    fig, ax = plt.subplots(
        figsize=(6.8, 4.5)
    )

    ax.errorbar(
        x,
        y,
        yerr=y_std,
        marker="o",
        linewidth=2.0,
        capsize=4,
        label=(
            "Jensen-HAPPO"
            if variant == "jensen"
            else "PF-HAPPO"
        ),
    )

    ax.axhline(
        ref_mean,
        linestyle="--",
        linewidth=1.8,
        label="HeLyMARL (K=10)",
    )

    ax.fill_between(
        [
            x.min(),
            x.max(),
        ],
        [
            ref_mean - ref_std,
            ref_mean - ref_std,
        ],
        [
            ref_mean + ref_std,
            ref_mean + ref_std,
        ],
        alpha=0.14,
    )

    ax.set_xticks(x)
    ax.set_xlabel("Lagrangian training checkpoint $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(
        rf"$\kappa={kappa}$"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_intra_episode_metrics(
    variant: str,
    kappa: float,
    lagrangian_summary: Sequence[Mapping[str, object]],
    helymarl_summary: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    specifications = [
        (
            "D_E",
            r"$\mathcal{D}_E$",
            "D_E.png",
        ),
        (
            "D_H",
            r"$\mathcal{D}_H$",
            "D_H.png",
        ),
        (
            "energy_utilization_mean",
            "Energy budget utilization",
            "energy_utilization.png",
        ),
        (
            "handover_utilization_mean",
            "Handover budget utilization",
            "handover_utilization.png",
        ),
    ]

    for metric, ylabel, filename in specifications:
        plot_checkpoint_metric(
            variant=variant,
            kappa=kappa,
            metric=metric,
            ylabel=ylabel,
            lagrangian_summary=lagrangian_summary,
            helymarl_summary=helymarl_summary,
            output_path=output_dir / filename,
        )


# =============================================================================
# 11. Validation before the expensive evaluation
# =============================================================================
def validate_inputs() -> None:
    print("\n[Input validation]")

    for variant in VARIANTS:
        if variant not in LAGRANGIAN_RUN_DIR_TEMPLATES:
            raise KeyError(
                f"No run-directory template for variant={variant}"
            )

        for kappa in KAPPAS:
            for train_seed in TRAIN_SEEDS:
                dual_path = resolve_dual_history_path(
                    variant,
                    kappa,
                    train_seed,
                )
                print(f"  dual: {dual_path}")

                for episode in CHECKPOINT_EPISODES:
                    checkpoint_path = resolve_lagrangian_checkpoint(
                        variant,
                        kappa,
                        train_seed,
                        episode,
                    )
                    print(
                        f"  checkpoint ep{episode}: "
                        f"{checkpoint_path}"
                    )

    for kappa in KAPPAS:
        for train_seed in TRAIN_SEEDS:
            model_path = resolve_helymarl_model(
                kappa,
                train_seed,
            )
            print(f"  HeLyMARL: {model_path}")


# =============================================================================
# 12. Main
# =============================================================================
def main() -> None:
    ensure_dir(PROJECT_ROOT / OUTPUT_ROOT)

    if RUN_TRAINING:
        print("\n[1/3] PF/Jensen-HAPPO training")
        train_all_lagrangian_runs()

    if not RUN_ANALYSIS:
        print("\nTraining completed. Analysis is disabled.")
        return

    # Validation is intentionally performed after training.
    print("\n[2/3] Input validation and inter-episode analysis")
    validate_inputs()

    for variant in VARIANTS:
        for kappa in KAPPAS:
            analyze_inter_episode_one_setting(
                variant,
                kappa,
            )

    print("\n[3/3] Intra-episode checkpoint evaluation")
    for variant in VARIANTS:
        for kappa in KAPPAS:
            run_intra_episode_experiment(
                variant,
                kappa,
            )

    print("\nCompleted.")
    print(
        "Training runs saved under:\n"
        "  results/pf_convergence or results/jensen_convergence_50.0\n"
        "Verification results saved under:\n"
        f"  {PROJECT_ROOT / OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
