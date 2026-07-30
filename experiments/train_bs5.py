#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B=5 환경에서 MAPPO, HAPPO, LyMARL을
kappa in {0.015, 0.020}에 대해 한 번에 학습 + 평가하는 통합 실행 코드.

공통 설정
---------
- BS = 5
- UE = 20
- BS 위치:
    BS1 = (50, 50)
    BS2 = (25, 25)
    BS3 = (75, 25)
    BS4 = (25, 75)
    BS5 = (75, 75)
- Train seeds = [0, 1, 2]
- Eval seeds = [2000, ..., 2004]
- 10 episodes x 10,000 steps
- Training: soft constraint
- Evaluation: hard constraint
- tx_power = 20 dBm
- rho = 0.6
- V = 5
- Top-K = 5

필수 파일
---------
LyMARL_extended/
├── env/
├── HeLyMARL/
├── LyMARL_handover.py
└── experiments/
    └── train_eval_b5_all.py

주의
----
LyMARL_handover.py의 import가 아래처럼 되어 있어야 한다.

from env.basestation import BaseStation, SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage
"""

import os
import sys

# ============================================================
# 0. Project root / thread settings
# ============================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import gc
from typing import Dict, List, Tuple

import numpy as np
import torch

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.env_happo import HAPPOEnvironment as UnifiedEnvironment
from HeLyMARL.trainer_mappo import MAPPOTrainer as UnifiedMAPPOTrainer
from HeLyMARL.trainer_happo import HAPPOTrainer as UnifiedHAPPOTrainer

from LyMARL_handover import (
    MAPPOEnvironment as LyMARLEnvironment,
    MAPPOTrainer as LyMARLTrainer,
)


# ============================================================
# 1. Experiment settings
# ============================================================
ALGORITHMS = ["HAPPO"]

TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

KAPPA_LIST = [0.015, 0.020]

NUM_USERS = 20
NUM_BS = 5

AREA_SIZE = 100.0
COVERAGE_RADIUS = 35.0
TX_POWER_DBM = 20.0

V = 5.0
POWER_BUDGET_RATIO = 0.6
LAMBDA_E = 0.0

ON_WINDOW = 1000
BS_TOP_K = 5

TRAIN_EPISODES = 10
STEPS_PER_EPISODE = 10000
TRAIN_STEPS = TRAIN_EPISODES * STEPS_PER_EPISODE
EVAL_STEPS = 10000
UPDATE_INTERVAL = 128

JFI_BLOCK_SIZE = 1000
OBJECTIVE_EPS = 1e-12

# 기존 실험과 동일하게 stochastic evaluation
DETERMINISTIC_EVAL = False

RUN_TRAIN = True
RUN_EVAL = True

# 기존 model/eval 파일이 있으면 건너뜀
OVERWRITE_EXISTING = False

SAVE_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "results_b5_all",
)

os.makedirs(SAVE_ROOT, exist_ok=True)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# 2. Utility
# ============================================================
def kappa_to_str(kappa: float) -> str:
    return f"{float(kappa):.3f}".rstrip("0").rstrip(".")


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_csv(rows: List[Dict], path: str):
    if not rows:
        return

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[CSV SAVED] {path}")


# ============================================================
# 3. Five-BS topology
# ============================================================
def generate_five_bs_coverage(
    area_size: float = 100.0,
) -> List[Tuple[float, float]]:
    """
    그림의 B=5 배치와 동일.

        BS4 (25,75)          BS5 (75,75)

                    BS1 (50,50)

        BS2 (25,25)          BS3 (75,25)
    """
    center_x = area_size / 2.0
    center_y = area_size / 2.0
    offset = area_size / 4.0

    return [
        (center_x, center_y),                    # BS1
        (center_x - offset, center_y - offset), # BS2
        (center_x + offset, center_y - offset), # BS3
        (center_x - offset, center_y + offset), # BS4
        (center_x + offset, center_y + offset), # BS5
    ]


def make_topology(
    seed: int,
) -> Tuple[List[SmallCellBaseStation], List[UserEquipment]]:
    set_seed(seed)

    positions = generate_five_bs_coverage(
        area_size=AREA_SIZE,
    )

    if len(positions) != NUM_BS:
        raise RuntimeError(
            f"Expected {NUM_BS} BS positions, got {len(positions)}"
        )

    base_stations = [
        SmallCellBaseStation(
            i + 1,
            position,
            TX_POWER_DBM,
            COVERAGE_RADIUS,
        )
        for i, position in enumerate(positions)
    ]

    users = [
        UserEquipment(
            i + 1,
            (
                np.random.uniform(10, 90),
                np.random.uniform(10, 90),
            ),
        )
        for i in range(NUM_USERS)
    ]

    return base_stations, users


# ============================================================
# 4. Environment builders
# ============================================================
def build_unified_env(
    seed: int,
    kappa: float,
    use_hard_constraint: bool,
):
    base_stations, users = make_topology(seed)

    env = UnifiedEnvironment(
        base_stations=base_stations,
        users=users,
        V=V,
        power_budget_ratio=POWER_BUDGET_RATIO,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=ON_WINDOW,
        bs_top_k=BS_TOP_K,
        hard_window_len=STEPS_PER_EPISODE,
        bs_over_penalty=0.0,
        use_hard_constraint=use_hard_constraint,
        lambda_E=LAMBDA_E,
        kappa=kappa,
    )

    if env.n_bs != NUM_BS or env.n_agents != NUM_USERS:
        raise RuntimeError(
            f"Unified env size mismatch: B={env.n_bs}, U={env.n_agents}"
        )

    return env


def build_lymarl_env(
    seed: int,
    kappa: float,
    use_hard_constraint: bool,
):
    base_stations, users = make_topology(seed)

    env = LyMARLEnvironment(
        base_stations=base_stations,
        users=users,
        V=V,
        power_budget_ratio=POWER_BUDGET_RATIO,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=ON_WINDOW,
        bs_top_k=BS_TOP_K,
        hard_window_len=STEPS_PER_EPISODE,

        # Role-specific LyMARL shaping 제거 설정
        bs_over_penalty=0.0,
        eta_q=1.0,
        alpha_rate=0.0,
        beta_z=0.0,
        alpha3=0.0,

        use_hard_constraint=use_hard_constraint,
        use_imperfect_csi=False,
        csi_error_var=0.0,
        kappa=kappa,
    )

    if env.n_bs != NUM_BS or env.n_agents != NUM_USERS:
        raise RuntimeError(
            f"LyMARL env size mismatch: B={env.n_bs}, U={env.n_agents}"
        )

    return env


def build_env(
    algorithm: str,
    seed: int,
    kappa: float,
    use_hard_constraint: bool,
):
    if algorithm in ("MAPPO", "HAPPO"):
        return build_unified_env(
            seed=seed,
            kappa=kappa,
            use_hard_constraint=use_hard_constraint,
        )

    if algorithm == "LyMARL":
        return build_lymarl_env(
            seed=seed,
            kappa=kappa,
            use_hard_constraint=use_hard_constraint,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================
# 5. Trainer builders
# ============================================================
def build_unified_trainer(
    env,
    algorithm: str,
):
    if algorithm == "MAPPO":
        trainer_class = UnifiedMAPPOTrainer
    elif algorithm == "HAPPO":
        trainer_class = UnifiedHAPPOTrainer
    else:
        raise ValueError(
            f"Unified trainer does not support: {algorithm}"
        )

    return trainer_class(
        env=env,
        eval_env=None,
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


def build_lymarl_trainer(env):
    return LyMARLTrainer(
        env=env,
        lr_actor_ue=3e-4,
        lr_actor_bs=3e-4,
        lr_critic_ue=1e-3,
        lr_critic_bs=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef_ue=0.05,
        entropy_coef_bs=0.05,
        value_coef_ue=0.5,
        value_coef_bs=0.5,
        max_grad_norm=0.5,
        n_epochs=4,
        minibatch_size=256,
    )


def build_trainer(
    env,
    algorithm: str,
):
    if algorithm in ("MAPPO", "HAPPO"):
        return build_unified_trainer(
            env=env,
            algorithm=algorithm,
        )

    if algorithm == "LyMARL":
        return build_lymarl_trainer(env)

    raise ValueError(f"Unknown algorithm: {algorithm}")


# ============================================================
# 6. Paths
# ============================================================
def make_run_dir(
    algorithm: str,
    kappa: float,
    train_seed: int,
) -> str:
    return os.path.join(
        SAVE_ROOT,
        (
            f"{algorithm}_B5_"
            f"kappa_{kappa_to_str(kappa)}_"
            f"seed_{train_seed}"
        ),
    )


def make_model_path(
    algorithm: str,
    kappa: float,
    train_seed: int,
) -> str:
    return os.path.join(
        make_run_dir(algorithm, kappa, train_seed),
        "model.pt",
    )


def make_train_npz_path(
    algorithm: str,
    kappa: float,
    train_seed: int,
) -> str:
    return os.path.join(
        make_run_dir(algorithm, kappa, train_seed),
        "train.npz",
    )


def make_eval_npz_path(
    algorithm: str,
    kappa: float,
    train_seed: int,
    eval_seed: int,
) -> str:
    return os.path.join(
        make_run_dir(algorithm, kappa, train_seed),
        f"eval_seed_{eval_seed}.npz",
    )


# ============================================================
# 7. Train
# ============================================================
def train_one_model(
    algorithm: str,
    kappa: float,
    train_seed: int,
):
    run_dir = make_run_dir(
        algorithm,
        kappa,
        train_seed,
    )
    os.makedirs(run_dir, exist_ok=True)

    model_path = make_model_path(
        algorithm,
        kappa,
        train_seed,
    )

    train_npz_path = make_train_npz_path(
        algorithm,
        kappa,
        train_seed,
    )

    if (
        not OVERWRITE_EXISTING
        and os.path.exists(model_path)
    ):
        print(f"[SKIP TRAIN] {model_path}")
        return

    print("\n" + "=" * 110)
    print(
        f"TRAIN | {algorithm} | "
        f"B={NUM_BS} | U={NUM_USERS} | "
        f"kappa={kappa:.3f} | "
        f"train_seed={train_seed}"
    )
    print("=" * 110)

    env = build_env(
        algorithm=algorithm,
        seed=train_seed,
        kappa=kappa,
        use_hard_constraint=False,
    )

    # topology 생성 과정에서 RNG가 사용되므로
    # network initialization 직전에 seed를 다시 고정
    set_seed(train_seed)

    trainer = build_trainer(
        env=env,
        algorithm=algorithm,
    )

    if algorithm in ("MAPPO", "HAPPO"):
        trainer.train(
            n_episodes=TRAIN_EPISODES,
            steps_per_episode=STEPS_PER_EPISODE,
            update_interval=UPDATE_INTERVAL,
            save_npz_path=train_npz_path,
            eval_every=0,
            policy_improvement_dir=None,
            save_episode_end_checkpoint=False,
        )
    else:
        trainer.train(
            n_steps=TRAIN_STEPS,
            update_interval=UPDATE_INTERVAL,
            save_npz_path=train_npz_path,
            steps_per_episode=STEPS_PER_EPISODE,
        )

    trainer.save_model(model_path)

    del trainer
    del env
    cleanup_cuda()


# ============================================================
# 8. Model load
# ============================================================
def load_model_for_eval(
    trainer,
    algorithm: str,
    model_path: str,
):
    if algorithm == "LyMARL":
        trainer.load_model(
            model_path,
            load_optim=False,
        )
    else:
        trainer.load_model(model_path)


# ============================================================
# 9. Common evaluation action selection
# ============================================================
@torch.no_grad()
def select_actions_for_eval(
    trainer,
    env,
    algorithm: str,
    local_obs,
    global_obs,
):
    if algorithm in ("MAPPO", "HAPPO"):
        (
            ue_actions,
            _ue_logp,
            _ue_entropy,
            _ue_masks,
            bs_actions,
            _bs_logp,
            _bs_entropy,
            _bs_obs,
            _bs_masks,
            cand_lists,
            _value,
        ) = trainer.select_actions(
            local_obs=local_obs,
            global_obs=global_obs,
            env=env,
            deterministic=DETERMINISTIC_EVAL,
        )

        return ue_actions, bs_actions, cand_lists

    # LyMARL_handover.py의 select_actions는 stochastic sampling 사용
    if DETERMINISTIC_EVAL:
        raise ValueError(
            "현재 통합 코드에서 LyMARL deterministic eval은 지원하지 않습니다. "
            "DETERMINISTIC_EVAL=False로 두세요."
        )

    (
        ue_actions,
        _ue_logp,
        _ue_entropy,
        _ue_masks,
        bs_actions,
        _bs_logp,
        _bs_entropy,
        _bs_obs,
        _bs_masks,
        cand_lists,
        _v_ue,
        _v_bs,
    ) = trainer.select_actions(
        local_obs,
        global_obs,
    )

    return ue_actions, bs_actions, cand_lists


# ============================================================
# 10. Common metric helpers
# ============================================================
def compute_block_jfi(
    slot_rates,
    block_size: int = 1000,
    eps: float = 1e-12,
) -> float:
    rates = np.asarray(
        slot_rates,
        dtype=np.float64,
    )

    if rates.ndim != 2 or rates.shape[0] == 0:
        return np.nan

    values = []

    for start in range(
        0,
        rates.shape[0],
        block_size,
    ):
        block = rates[
            start:start + block_size
        ]

        if block.shape[0] == 0:
            continue

        avg_user_rates = block.mean(axis=0)

        numerator = float(
            avg_user_rates.sum() ** 2
        )

        denominator = float(
            len(avg_user_rates)
            * np.square(avg_user_rates).sum()
        )

        if denominator > eps:
            values.append(
                numerator / denominator
            )

    return (
        float(np.mean(values))
        if values
        else np.nan
    )


def compute_pf_objective(
    slot_rates,
    eps: float = 1e-12,
) -> float:
    rates = np.asarray(
        slot_rates,
        dtype=np.float64,
    )

    if rates.ndim != 2 or rates.shape[0] == 0:
        return np.nan

    avg_user_rates = rates.mean(axis=0)

    return float(
        np.sum(
            np.log(avg_user_rates + eps)
        )
    )


def compute_request_metrics(
    request_history,
) -> Tuple[float, float]:
    request_arr = np.asarray(
        request_history,
        dtype=np.int64,
    )

    if request_arr.ndim != 2 or request_arr.shape[0] == 0:
        return np.nan, np.nan

    if request_arr.shape[0] == 1:
        return 1.0, 0.0

    switch_flags = (
        request_arr[1:]
        != request_arr[:-1]
    )

    request_switch = float(
        np.mean(switch_flags)
    )

    runs_per_user = (
        switch_flags.sum(axis=0)
        + 1
    )

    request_dwell = float(
        np.mean(
            request_arr.shape[0]
            / np.maximum(runs_per_user, 1)
        )
    )

    return request_dwell, request_switch


def compute_conditional_ho(
    serving_event_history,
) -> float:
    """
    serving_event_history:
        [T, U]
        0 = 그 슬롯에서 미서비스
        1..B = 그 슬롯에서 실제 serving BS
    """
    serving_arr = np.asarray(
        serving_event_history,
        dtype=np.int64,
    )

    if serving_arr.ndim != 2:
        return np.nan

    total_transitions = 0
    total_switches = 0

    for user_idx in range(
        serving_arr.shape[1]
    ):
        sequence = serving_arr[:, user_idx]
        sequence = sequence[
            sequence > 0
        ]

        if sequence.size <= 1:
            continue

        switches = (
            sequence[1:]
            != sequence[:-1]
        )

        total_transitions += (
            sequence.size - 1
        )

        total_switches += int(
            switches.sum()
        )

    if total_transitions == 0:
        return 0.0

    return float(
        total_switches
        / total_transitions
    )


# ============================================================
# 11. Info parsing
# ============================================================
def get_handover_map(
    algorithm: str,
    info: Dict,
) -> Dict[int, float]:
    if algorithm in ("MAPPO", "HAPPO"):
        source = info.get("handover_u")
    else:
        source = info.get("handover_flags")

    if source is None:
        raise KeyError(
            f"Handover key not found for {algorithm}. "
            f"Available info keys: {list(info.keys())}"
        )

    return source


def get_request_bs_map(
    env,
    algorithm: str,
    info: Dict,
    ue_actions: Dict[int, int],
) -> Dict[int, int]:
    if (
        algorithm == "LyMARL"
        and "current_request_bs" in info
    ):
        return {
            int(ue_id): int(bs_id)
            for ue_id, bs_id
            in info["current_request_bs"].items()
        }

    # Unified와 fallback:
    # UE action index -> 실제 BS ID
    return {
        user.ue_id: int(
            env.base_stations[
                int(ue_actions[user.ue_id])
            ].bs_id
        )
        for user in env.users
    }


def get_serving_event_map(
    env,
    algorithm: str,
    info: Dict,
) -> Dict[int, int]:
    """
    그 슬롯에서 실제 service를 받은 UE만 BS ID 기록.
    미서비스 사용자는 0.
    """
    served = {
        user.ue_id: 0
        for user in env.users
    }

    served_rates = info.get(
        "served_rates",
        {},
    )

    if (
        algorithm in ("MAPPO", "HAPPO")
        and "served_bs_of_user" in info
    ):
        source = info["served_bs_of_user"]

        for user in env.users:
            bs_id = source.get(
                user.ue_id,
                None,
            )

            if (
                bs_id is not None
                and float(
                    served_rates.get(
                        user.ue_id,
                        0.0,
                    )
                ) > 0.0
            ):
                served[user.ue_id] = int(bs_id)

        return served

    # LyMARL: bs_selections에서 해당 슬롯 service event 복구
    selections = info.get(
        "bs_selections",
        {},
    )

    for bs_id, ue_id in selections.items():
        if ue_id is None:
            continue

        ue_id = int(ue_id)

        if float(
            served_rates.get(
                ue_id,
                0.0,
            )
        ) > 0.0:
            served[ue_id] = int(bs_id)

    return served


# ============================================================
# 12. One common evaluation rollout
# ============================================================
@torch.no_grad()
def evaluate_one_model(
    algorithm: str,
    kappa: float,
    train_seed: int,
    eval_seed: int,
) -> Dict:
    model_path = make_model_path(
        algorithm,
        kappa,
        train_seed,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    eval_npz_path = make_eval_npz_path(
        algorithm,
        kappa,
        train_seed,
        eval_seed,
    )

    if (
        not OVERWRITE_EXISTING
        and os.path.exists(eval_npz_path)
    ):
        print(f"[SKIP EVAL] {eval_npz_path}")
        return load_metric_row_from_npz(
            eval_npz_path
        )

    print("\n" + "=" * 110)
    print(
        f"EVAL | {algorithm} | "
        f"B={NUM_BS} | U={NUM_USERS} | "
        f"kappa={kappa:.3f} | "
        f"train_seed={train_seed} | "
        f"eval_seed={eval_seed}"
    )
    print("=" * 110)

    env = build_env(
        algorithm=algorithm,
        seed=eval_seed,
        kappa=kappa,
        use_hard_constraint=True,
    )

    trainer = build_trainer(
        env=env,
        algorithm=algorithm,
    )

    load_model_for_eval(
        trainer=trainer,
        algorithm=algorithm,
        model_path=model_path,
    )

    set_seed(eval_seed)

    trainer.ue_actor.eval()
    trainer.bs_actor.eval()

    if hasattr(trainer, "critic"):
        trainer.critic.eval()
    if hasattr(trainer, "critic_ue"):
        trainer.critic_ue.eval()
    if hasattr(trainer, "critic_bs"):
        trainer.critic_bs.eval()

    local_obs, global_obs = env.reset()

    throughput_history = []
    slot_rates = []
    power_history = []
    handover_history = []
    request_history = []
    serving_event_history = []

    log_window = 1000

    for step in range(EVAL_STEPS):
        (
            ue_actions,
            bs_actions,
            cand_lists,
        ) = select_actions_for_eval(
            trainer=trainer,
            env=env,
            algorithm=algorithm,
            local_obs=local_obs,
            global_obs=global_obs,
        )

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

        throughput_history.append(
            float(info["total_throughput"])
        )

        slot_rates.append([
            float(
                info["served_rates"][
                    user.ue_id
                ]
            )
            for user in env.users
        ])

        power_history.append([
            float(
                info["power_consumed"][
                    bs.bs_id
                ]
            )
            for bs in env.base_stations
        ])

        handover_map = get_handover_map(
            algorithm=algorithm,
            info=info,
        )

        handover_history.append([
            float(
                handover_map[
                    user.ue_id
                ]
            )
            for user in env.users
        ])

        request_map = get_request_bs_map(
            env=env,
            algorithm=algorithm,
            info=info,
            ue_actions=ue_actions,
        )

        request_history.append([
            int(
                request_map[
                    user.ue_id
                ]
            )
            for user in env.users
        ])

        serving_map = get_serving_event_map(
            env=env,
            algorithm=algorithm,
            info=info,
        )

        serving_event_history.append([
            int(
                serving_map[
                    user.ue_id
                ]
            )
            for user in env.users
        ])

        local_obs = next_local_obs
        global_obs = next_global_obs

        if (step + 1) % log_window == 0:
            recent_thr = float(
                np.mean(
                    throughput_history[-log_window:]
                )
            )

            recent_power = np.asarray(
                power_history[-log_window:],
                dtype=np.float64,
            )

            recent_ho = np.asarray(
                handover_history[-log_window:],
                dtype=np.float64,
            )

            print(
                f"[{algorithm}] "
                f"step={step + 1:5d} | "
                f"Thr={recent_thr:.4f} | "
                f"ON={np.mean(recent_power > 0.0):.4f} | "
                f"HO={np.mean(recent_ho):.4f}"
            )

        if done:
            break

    throughput_arr = np.asarray(
        throughput_history,
        dtype=np.float64,
    )

    slot_rates_arr = np.asarray(
        slot_rates,
        dtype=np.float64,
    )

    # [B, T]
    power_mat = np.asarray(
        power_history,
        dtype=np.float64,
    ).T

    # [T, U]
    handover_mat = np.asarray(
        handover_history,
        dtype=np.float64,
    )

    request_arr = np.asarray(
        request_history,
        dtype=np.int64,
    )

    serving_event_arr = np.asarray(
        serving_event_history,
        dtype=np.int64,
    )

    request_dwell, request_switch = (
        compute_request_metrics(
            request_arr
        )
    )

    conditional_ho = compute_conditional_ho(
        serving_event_arr
    )

    metrics = {
        "throughput": float(
            np.mean(throughput_arr)
        ),
        "jfi": compute_block_jfi(
            slot_rates_arr,
            block_size=JFI_BLOCK_SIZE,
        ),
        "objective": compute_pf_objective(
            slot_rates_arr,
            eps=OBJECTIVE_EPS,
        ),
        "on_ratio": float(
            np.mean(power_mat > 0.0)
        ),
        "ho_ratio": float(
            np.mean(handover_mat)
        ),
        "request_dwell": float(
            request_dwell
        ),
        "request_switch": float(
            request_switch
        ),
        "conditional_ho": float(
            conditional_ho
        ),
        "same_bs": float(
            1.0 - conditional_ho
        ),
    }

    os.makedirs(
        os.path.dirname(eval_npz_path),
        exist_ok=True,
    )

    np.savez_compressed(
        eval_npz_path,
        tag=np.asarray(
            (
                f"{algorithm}_B5_"
                f"kappa_{kappa_to_str(kappa)}"
            )
        ),
        algorithm=np.asarray(algorithm),
        n_users=np.asarray(NUM_USERS),
        n_bs=np.asarray(NUM_BS),
        kappa=np.asarray(kappa),
        train_seed=np.asarray(train_seed),
        eval_seed=np.asarray(eval_seed),

        throughput=throughput_arr,
        slot_rates=slot_rates_arr,
        power_mat=power_mat,
        handover_mat=handover_mat,
        request_bs_history=request_arr,
        serving_bs_history=serving_event_arr,

        final_throughput=np.asarray(
            metrics["throughput"]
        ),
        final_jfi=np.asarray(
            metrics["jfi"]
        ),
        final_objective=np.asarray(
            metrics["objective"]
        ),
        final_on_ratio=np.asarray(
            metrics["on_ratio"]
        ),
        final_ho_ratio=np.asarray(
            metrics["ho_ratio"]
        ),
        final_request_dwell=np.asarray(
            metrics["request_dwell"]
        ),
        final_request_switch=np.asarray(
            metrics["request_switch"]
        ),
        final_conditional_ho=np.asarray(
            metrics["conditional_ho"]
        ),
        final_same_bs=np.asarray(
            metrics["same_bs"]
        ),
    )

    row = {
        "algorithm": algorithm,
        "kappa": float(kappa),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "n_bs": NUM_BS,
        "n_users": NUM_USERS,
        **metrics,
    }

    print(
        f"[RESULT] {algorithm} | "
        f"kappa={kappa:.3f} | "
        f"train={train_seed} | "
        f"eval={eval_seed} | "
        f"Throughput={metrics['throughput']:.4f} | "
        f"JFI={metrics['jfi']:.4f} | "
        f"ON={metrics['on_ratio']:.4f} | "
        f"HO={metrics['ho_ratio']:.4f} | "
        f"ReqSwitch={metrics['request_switch']:.4f} | "
        f"CondHO={metrics['conditional_ho']:.4f}"
    )

    print(f"[NPZ SAVED] {eval_npz_path}")

    del trainer
    del env
    cleanup_cuda()

    return row


def load_metric_row_from_npz(
    path: str,
) -> Dict:
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            "algorithm": str(
                data["algorithm"].item()
            ),
            "kappa": float(
                data["kappa"]
            ),
            "train_seed": int(
                data["train_seed"]
            ),
            "eval_seed": int(
                data["eval_seed"]
            ),
            "n_bs": int(
                data["n_bs"]
            ),
            "n_users": int(
                data["n_users"]
            ),
            "throughput": float(
                data["final_throughput"]
            ),
            "jfi": float(
                data["final_jfi"]
            ),
            "objective": float(
                data["final_objective"]
            ),
            "on_ratio": float(
                data["final_on_ratio"]
            ),
            "ho_ratio": float(
                data["final_ho_ratio"]
            ),
            "request_dwell": float(
                data["final_request_dwell"]
            ),
            "request_switch": float(
                data["final_request_switch"]
            ),
            "conditional_ho": float(
                data["final_conditional_ho"]
            ),
            "same_bs": float(
                data["final_same_bs"]
            ),
        }


# ============================================================
# 13. Summary
# ============================================================
METRIC_NAMES = [
    "throughput",
    "jfi",
    "objective",
    "on_ratio",
    "ho_ratio",
    "request_dwell",
    "request_switch",
    "conditional_ho",
    "same_bs",
]


def aggregate_results(
    raw_rows: List[Dict],
):
    per_train_rows = []

    # 한 train seed 내부에서 eval seed 5개 평균
    for kappa in KAPPA_LIST:
        for algorithm in ALGORITHMS:
            for train_seed in TRAIN_SEEDS:
                rows = [
                    row
                    for row in raw_rows
                    if (
                        row["algorithm"] == algorithm
                        and np.isclose(
                            row["kappa"],
                            kappa,
                        )
                        and row["train_seed"] == train_seed
                    )
                ]

                if not rows:
                    continue

                summary = {
                    "algorithm": algorithm,
                    "kappa": float(kappa),
                    "train_seed": int(train_seed),
                    "n_bs": NUM_BS,
                    "n_users": NUM_USERS,
                    "n_eval_seeds": len(rows),
                }

                for metric in METRIC_NAMES:
                    summary[metric] = float(
                        np.mean([
                            row[metric]
                            for row in rows
                        ])
                    )

                per_train_rows.append(summary)

    final_rows = []

    # train seed 평균 및 표준편차
    for kappa in KAPPA_LIST:
        for algorithm in ALGORITHMS:
            rows = [
                row
                for row in per_train_rows
                if (
                    row["algorithm"] == algorithm
                    and np.isclose(
                        row["kappa"],
                        kappa,
                    )
                )
            ]

            if not rows:
                continue

            summary = {
                "algorithm": algorithm,
                "kappa": float(kappa),
                "n_bs": NUM_BS,
                "n_users": NUM_USERS,
                "n_train_seeds": len(rows),
            }

            for metric in METRIC_NAMES:
                values = np.asarray(
                    [
                        row[metric]
                        for row in rows
                    ],
                    dtype=np.float64,
                )

                summary[
                    f"{metric}_mean"
                ] = float(
                    np.mean(values)
                )

                summary[
                    f"{metric}_std"
                ] = float(
                    np.std(values, ddof=0)
                )

            final_rows.append(summary)

    return per_train_rows, final_rows


def print_final_summary(
    final_rows: List[Dict],
):
    print("\n" + "=" * 180)
    print(
        "B=5, U=20 TRAINING + EVALUATION SUMMARY"
    )
    print(
        "Each train seed is first averaged over evaluation seeds; "
        "mean/std are across train seeds."
    )
    print("=" * 180)

    for kappa in KAPPA_LIST:
        print(f"\n[KAPPA = {kappa:.3f}]")

        for algorithm in ALGORITHMS:
            matches = [
                row
                for row in final_rows
                if (
                    row["algorithm"] == algorithm
                    and np.isclose(
                        row["kappa"],
                        kappa,
                    )
                )
            ]

            if not matches:
                continue

            row = matches[0]

            print(
                f"{algorithm:7s} | "
                f"Throughput="
                f"{row['throughput_mean']:.4f}"
                f" +/- {row['throughput_std']:.4f} | "
                f"JFI="
                f"{row['jfi_mean']:.4f}"
                f" +/- {row['jfi_std']:.4f} | "
                f"ON="
                f"{row['on_ratio_mean']:.4f}"
                f" +/- {row['on_ratio_std']:.4f} | "
                f"HO="
                f"{row['ho_ratio_mean']:.4f}"
                f" +/- {row['ho_ratio_std']:.4f} | "
                f"ReqSwitch="
                f"{row['request_switch_mean']:.4f}"
                f" +/- {row['request_switch_std']:.4f} | "
                f"CondHO="
                f"{row['conditional_ho_mean']:.4f}"
                f" +/- {row['conditional_ho_std']:.4f} | "
                f"SameBS="
                f"{row['same_bs_mean']:.4f}"
                f" +/- {row['same_bs_std']:.4f}"
            )

    print("\n" + "=" * 180)


# ============================================================
# 14. Main
# ============================================================
def print_configuration():
    print("\n" + "#" * 120)
    print("B=5 THREE-ALGORITHM EXPERIMENT")
    print("#" * 120)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Algorithms         : {ALGORITHMS}")
    print(f"Kappa list         : {KAPPA_LIST}")
    print(f"Train seeds        : {TRAIN_SEEDS}")
    print(f"Eval seeds         : {EVAL_SEEDS}")
    print(f"Users / BS         : {NUM_USERS} / {NUM_BS}")
    print(f"Train episodes     : {TRAIN_EPISODES}")
    print(f"Steps per episode  : {STEPS_PER_EPISODE}")
    print(f"Evaluation steps   : {EVAL_STEPS}")
    print(f"TX power           : {TX_POWER_DBM} dBm")
    print(f"Power budget ratio : {POWER_BUDGET_RATIO}")
    print(f"Save root          : {SAVE_ROOT}")
    print(f"Overwrite          : {OVERWRITE_EXISTING}")
    print("BS positions       :")
    for bs_id, position in enumerate(
        generate_five_bs_coverage(AREA_SIZE),
        start=1,
    ):
        print(f"  BS{bs_id}: {position}")
    print("#" * 120 + "\n")


def main():
    print_configuration()

    raw_rows = []

    for kappa in KAPPA_LIST:
        print("\n" + "#" * 120)
        print(f"START KAPPA = {kappa:.3f}")
        print("#" * 120)

        for algorithm in ALGORITHMS:
            print("\n" + "-" * 120)
            print(
                f"START ALGORITHM = {algorithm} | "
                f"KAPPA = {kappa:.3f}"
            )
            print("-" * 120)

            for train_seed in TRAIN_SEEDS:
                if RUN_TRAIN:
                    train_one_model(
                        algorithm=algorithm,
                        kappa=kappa,
                        train_seed=train_seed,
                    )

                if RUN_EVAL:
                    for eval_seed in EVAL_SEEDS:
                        row = evaluate_one_model(
                            algorithm=algorithm,
                            kappa=kappa,
                            train_seed=train_seed,
                            eval_seed=eval_seed,
                        )

                        raw_rows.append(row)

                        # 중간 실패가 생겨도 완료된 결과는 CSV에 남김
                        save_csv(
                            raw_rows,
                            os.path.join(
                                SAVE_ROOT,
                                "raw_results_partial.csv",
                            ),
                        )

    save_csv(
        raw_rows,
        os.path.join(
            SAVE_ROOT,
            "raw_results.csv",
        ),
    )

    (
        per_train_rows,
        final_rows,
    ) = aggregate_results(
        raw_rows
    )

    save_csv(
        per_train_rows,
        os.path.join(
            SAVE_ROOT,
            "per_train_summary.csv",
        ),
    )

    save_csv(
        final_rows,
        os.path.join(
            SAVE_ROOT,
            "final_summary.csv",
        ),
    )

    print_final_summary(
        final_rows
    )


if __name__ == "__main__":
    main()
