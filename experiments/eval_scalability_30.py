import os
import sys
import csv
import gc
import glob
import inspect

# ============================================================
# 0. Project path
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

import numpy as np
import torch
from torch.distributions import Categorical

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

# Unified MAPPO / HAPPO
from HeLyMARL.env_happo import (
    HAPPOEnvironment as UnifiedEnvironment,
)
from HeLyMARL.trainer_mappo import (
    MAPPOTrainer as UnifiedMAPPOTrainer,
)
from HeLyMARL.trainer_happo import (
    HAPPOTrainer as UnifiedHAPPOTrainer,
)
from HeLyMARL.utils_happo import set_seed

# Role-specific LyMARL
from LyMARL_handover import (
    MAPPOEnvironment as LyMARLEnvironment,
    MAPPOTrainer as LyMARLTrainer,
)


# ============================================================
# 1. Experiment settings
# ============================================================
ALGORITHMS = [
    "MAPPO",
    "HAPPO",
    "LyMARL",
]

TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

TRAINED_NUM_USERS = 20
EVAL_NUM_USERS = 30

KAPPA = 0.015

STEPS_PER_EPISODE = 10000
JFI_BLOCK_SIZE = 1000
OBJECTIVE_EPS = 1e-12

# 기존 evaluation과 동일하게 stochastic sampling
DETERMINISTIC = False

SAVE_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "scalability_u30_kappa_0.015",
)

os.makedirs(
    SAVE_ROOT,
    exist_ok=True,
)

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# 2. Model path
# ============================================================
def get_model_path(
    algorithm: str,
    train_seed: int,
) -> str:
    """
    알고리즘별 모델 경로를 찾는다.

    먼저 예상 경로를 확인하고, 찾지 못하면
    results 아래에서 재귀적으로 검색한다.
    """
    if algorithm == "LyMARL":
        candidates = [
            os.path.join(
                PROJECT_ROOT,
                "results",
                "results_mappo_happo",
                "LyMARL_HO_kappa_0.015",
                f"seed_{train_seed}",
                "model.pt",
            ),
            os.path.join(
                PROJECT_ROOT,
                "results",
                "results_kappa",
                "LyMARL_HO_kappa_0.015",
                f"seed_{train_seed}",
                "model.pt",
            ),
        ]

        fallback_patterns = [
            os.path.join(
                PROJECT_ROOT,
                "results",
                "**",
                "LyMARL*0.015*",
                f"seed_{train_seed}",
                "model.pt",
            ),
            os.path.join(
                PROJECT_ROOT,
                "results",
                "**",
                f"LyMARL*0.015*seed_{train_seed}*",
                "model.pt",
            ),
        ]

    elif algorithm in ["MAPPO", "HAPPO"]:
        candidates = [
            os.path.join(
                PROJECT_ROOT,
                "results",
                "results_kappa",
                f"{algorithm}_kappa_0.015_seed_{train_seed}",
                "model.pt",
            ),
            os.path.join(
                PROJECT_ROOT,
                "results",
                "results_mappo_happo",
                f"{algorithm}_kappa_0.015_seed_{train_seed}",
                "model.pt",
            ),
            os.path.join(
                PROJECT_ROOT,
                "results",
                "results_mappo_happo",
                f"{algorithm}_kappa_0.015",
                f"seed_{train_seed}",
                "model.pt",
            ),
        ]

        fallback_patterns = [
            os.path.join(
                PROJECT_ROOT,
                "results",
                "**",
                f"{algorithm}*0.015*seed_{train_seed}*",
                "model.pt",
            ),
            os.path.join(
                PROJECT_ROOT,
                "results",
                "**",
                f"{algorithm}*0.015*",
                f"seed_{train_seed}",
                "model.pt",
            ),
        ]

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}"
        )

    # 예상 경로 우선
    for path in candidates:
        if os.path.isfile(path):
            print(
                f"[MODEL FOUND] "
                f"{algorithm}, train_seed={train_seed}\n"
                f"  {path}"
            )
            return path

    # 예상 경로에 없으면 재귀 검색
    found_paths = []

    for pattern in fallback_patterns:
        found_paths.extend(
            glob.glob(
                pattern,
                recursive=True,
            )
        )

    found_paths = sorted(
        set(
            path
            for path in found_paths
            if os.path.isfile(path)
        )
    )

    if len(found_paths) == 1:
        print(
            f"[MODEL FOUND BY SEARCH] "
            f"{algorithm}, train_seed={train_seed}\n"
            f"  {found_paths[0]}"
        )
        return found_paths[0]

    if len(found_paths) > 1:
        raise RuntimeError(
            f"{algorithm}, train_seed={train_seed}에 대해 "
            f"여러 모델이 검색되었습니다.\n"
            + "\n".join(found_paths)
            + "\nget_model_path()에서 정확한 경로를 지정하세요."
        )

    raise FileNotFoundError(
        f"{algorithm}, train_seed={train_seed} 모델을 "
        f"찾지 못했습니다.\n"
        + "\n".join(candidates)
    )


# ============================================================
# 3. Topology
# ============================================================
def make_topology(
    seed: int,
    num_users: int,
    tx_power_dbm: float = 20.0,
):
    set_seed(seed)

    area_size = 100

    sbs_positions = generate_triangle_coverage(
        area_size,
        35,
    )

    sbs_list = [
        SmallCellBaseStation(
            i + 1,
            position,
            tx_power_dbm,
            35,
        )
        for i, position in enumerate(
            sbs_positions
        )
    ]

    users = [
        UserEquipment(
            i + 1,
            (
                np.random.uniform(10, 90),
                np.random.uniform(10, 90),
            ),
        )
        for i in range(num_users)
    ]

    return sbs_list, users


# ============================================================
# 4. Environments
# ============================================================
def build_lymarl_env(
    seed: int,
    num_users: int,
    use_hard_constraint: bool,
):
    sbs_list, users = make_topology(
        seed=seed,
        num_users=num_users,
        tx_power_dbm=20.0,
    )

    return LyMARLEnvironment(
        base_stations=sbs_list,
        users=users,
        V=5.0,
        power_budget_ratio=0.6,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=1000,
        bs_top_k=5,
        hard_window_len=STEPS_PER_EPISODE,
        bs_over_penalty=0.0,
        eta_q=1.0,
        alpha_rate=0.0,
        beta_z=0.0,
        alpha3=0.0,
        use_hard_constraint=use_hard_constraint,
        use_imperfect_csi=False,
        csi_error_var=0.0,
        kappa=KAPPA,
    )


def build_unified_env(
    seed: int,
    num_users: int,
    use_hard_constraint: bool,
):
    sbs_list, users = make_topology(
        seed=seed,
        num_users=num_users,
        tx_power_dbm=20.0,
    )

    return UnifiedEnvironment(
        base_stations=sbs_list,
        users=users,
        V=5.0,
        power_budget_ratio=0.6,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=1000,
        bs_top_k=5,
        hard_window_len=STEPS_PER_EPISODE,
        bs_over_penalty=0.0,
        use_hard_constraint=use_hard_constraint,
        lambda_E=0.0,
        kappa=KAPPA,
    )


# ============================================================
# 5. Trainers
# ============================================================
def instantiate_supported(
    trainer_class,
    **kwargs,
):
    """
    Trainer 버전에 따라 지원하지 않는 keyword는 제거한다.
    """
    signature = inspect.signature(
        trainer_class.__init__
    )

    has_var_keyword = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if has_var_keyword:
        return trainer_class(**kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    return trainer_class(
        **supported_kwargs
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
        n_epochs=4,
        minibatch_size=256,
    )


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
            f"Unsupported unified algorithm: "
            f"{algorithm}"
        )

    return instantiate_supported(
        trainer_class,
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
        max_grad_norm=0.5,
        n_epochs=4,
        minibatch_size=256,
    )


def build_env_and_trainer(
    algorithm: str,
    eval_seed: int,
):
    if algorithm == "LyMARL":
        env = build_lymarl_env(
            seed=eval_seed,
            num_users=EVAL_NUM_USERS,
            use_hard_constraint=True,
        )

        trainer = build_lymarl_trainer(
            env
        )

    elif algorithm in ["MAPPO", "HAPPO"]:
        env = build_unified_env(
            seed=eval_seed,
            num_users=EVAL_NUM_USERS,
            use_hard_constraint=True,
        )

        trainer = build_unified_trainer(
            env=env,
            algorithm=algorithm,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}"
        )

    return env, trainer


# ============================================================
# 6. Load actor only
# ============================================================
def load_actor_only(
    trainer,
    model_path: str,
):
    payload = torch.load(
        model_path,
        map_location=trainer.device,
    )

    if "ue_actor" in payload:
        ue_actor_state = payload["ue_actor"]
    elif "actor_ue" in payload:
        ue_actor_state = payload["actor_ue"]
    else:
        raise KeyError(
            f"UE actor key가 없습니다.\n"
            f"Available keys: {list(payload.keys())}"
        )

    if "bs_actor" in payload:
        bs_actor_state = payload["bs_actor"]
    elif "actor_bs" in payload:
        bs_actor_state = payload["actor_bs"]
    else:
        raise KeyError(
            f"BS actor key가 없습니다.\n"
            f"Available keys: {list(payload.keys())}"
        )

    trainer.ue_actor.load_state_dict(
        ue_actor_state,
        strict=True,
    )

    trainer.bs_actor.load_state_dict(
        bs_actor_state,
        strict=True,
    )

    trainer.ue_actor.eval()
    trainer.bs_actor.eval()

    meta = payload.get(
        "meta",
        {},
    )

    print(
        f"[ACTOR-ONLY LOAD]\n"
        f"  model={model_path}\n"
        f"  checkpoint users="
        f"{meta.get('n_users', 'unknown')}\n"
        f"  evaluation users="
        f"{len(trainer.env.users)}"
    )


# ============================================================
# 7. Actor-only action selection
# ============================================================
@torch.no_grad()
def select_actions_actor_only(
    trainer,
    env,
    local_obs,
    deterministic: bool,
):
    # --------------------------------------------------------
    # UE actions
    # --------------------------------------------------------
    ue_obs_batch = np.stack(
        [
            local_obs[user.ue_id]
            for user in env.users
        ],
        axis=0,
    ).astype(np.float32)

    ue_mask_batch = np.stack(
        [
            env._get_action_mask(
                user.ue_id
            )
            for user in env.users
        ],
        axis=0,
    ).astype(bool)

    ue_obs_t = torch.as_tensor(
        ue_obs_batch,
        dtype=torch.float32,
        device=trainer.device,
    )

    ue_mask_t = torch.as_tensor(
        ue_mask_batch,
        dtype=torch.bool,
        device=trainer.device,
    )

    ue_logits = trainer.ue_actor(
        ue_obs_t
    )

    ue_logits = ue_logits.masked_fill(
        ~ue_mask_t,
        float("-inf"),
    )

    if deterministic:
        ue_actions_t = torch.argmax(
            ue_logits,
            dim=-1,
        )
    else:
        ue_actions_t = Categorical(
            logits=ue_logits
        ).sample()

    ue_actions = {
        user.ue_id:
            int(ue_actions_t[i].item())
        for i, user in enumerate(
            env.users
        )
    }

    # --------------------------------------------------------
    # BS actions
    # --------------------------------------------------------
    (
        bs_obs_batch,
        bs_mask_batch,
        cand_lists,
    ) = env.build_bs_decision_inputs(
        ue_actions
    )

    bs_obs_t = torch.as_tensor(
        bs_obs_batch,
        dtype=torch.float32,
        device=trainer.device,
    )

    bs_mask_t = torch.as_tensor(
        bs_mask_batch,
        dtype=torch.bool,
        device=trainer.device,
    )

    bs_logits = trainer.bs_actor(
        bs_obs_t
    )

    bs_logits = bs_logits.masked_fill(
        ~bs_mask_t,
        float("-inf"),
    )

    if deterministic:
        bs_actions_t = torch.argmax(
            bs_logits,
            dim=-1,
        )
    else:
        bs_actions_t = Categorical(
            logits=bs_logits
        ).sample()

    bs_actions = {
        bs.bs_id:
            int(bs_actions_t[i].item())
        for i, bs in enumerate(
            env.base_stations
        )
    }

    return (
        ue_actions,
        bs_actions,
        cand_lists,
    )


# ============================================================
# 8. Metric helpers
# ============================================================
def compute_block_jfi(
    slot_rates,
    block_size: int = 1000,
    eps: float = 1e-12,
):
    rates = np.asarray(
        slot_rates,
        dtype=np.float64,
    )

    if (
        rates.ndim != 2
        or rates.shape[0] == 0
    ):
        return np.nan

    block_jfis = []

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

        avg_user_rates = np.mean(
            block,
            axis=0,
        )

        numerator = float(
            np.sum(avg_user_rates) ** 2
        )

        denominator = float(
            len(avg_user_rates)
            * np.sum(avg_user_rates ** 2)
        )

        if denominator > eps:
            block_jfis.append(
                numerator / denominator
            )

    return (
        float(np.mean(block_jfis))
        if block_jfis
        else np.nan
    )


def compute_pf_objective(
    slot_rates,
    eps: float = 1e-12,
):
    rates = np.asarray(
        slot_rates,
        dtype=np.float64,
    )

    if (
        rates.ndim != 2
        or rates.shape[0] == 0
    ):
        return np.nan

    avg_user_rates = np.mean(
        rates,
        axis=0,
    )

    return float(
        np.sum(
            np.log(
                avg_user_rates + eps
            )
        )
    )


def compute_request_metrics(
    request_history,
):
    request_arr = np.asarray(
        request_history,
        dtype=np.int64,
    )

    if (
        request_arr.ndim != 2
        or request_arr.shape[0] == 0
    ):
        return np.nan, np.nan

    if request_arr.shape[0] == 1:
        return 1.0, 0.0

    switch_flags = (
        request_arr[1:]
        != request_arr[:-1]
    )

    req_switch = float(
        np.mean(switch_flags)
    )

    runs_per_user = (
        np.sum(
            switch_flags,
            axis=0,
        )
        + 1
    )

    req_dwell = float(
        np.mean(
            request_arr.shape[0]
            / np.maximum(
                runs_per_user,
                1,
            )
        )
    )

    return req_dwell, req_switch


def compute_conditional_ho(
    serving_event_history,
):
    """
    serving_event_history:
        [T, U]

    0은 해당 슬롯에서 service를 받지 않은 사용자.
    양수는 해당 슬롯의 실제 serving BS ID.
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
        service_sequence = (
            serving_arr[:, user_idx]
        )

        service_sequence = (
            service_sequence[
                service_sequence > 0
            ]
        )

        if service_sequence.size <= 1:
            continue

        switches = (
            service_sequence[1:]
            != service_sequence[:-1]
        )

        total_transitions += (
            service_sequence.size - 1
        )

        total_switches += int(
            np.sum(switches)
        )

    if total_transitions == 0:
        return 0.0

    return float(
        total_switches
        / total_transitions
    )


# ============================================================
# 9. Info parsing helpers
# ============================================================
def extract_handover_map(
    info,
):
    if "handover_flags" in info:
        return info["handover_flags"]

    if "handover_u" in info:
        return info["handover_u"]

    raise KeyError(
        "Handover 정보를 찾을 수 없습니다.\n"
        f"Available keys: {list(info.keys())}"
    )


def extract_request_map(
    env,
    info,
    ue_actions,
):
    if "current_request_bs" in info:
        return info["current_request_bs"]

    return {
        user.ue_id:
            int(
                env.base_stations[
                    int(
                        ue_actions[user.ue_id]
                    )
                ].bs_id
            )
        for user in env.users
    }


def extract_serving_event_map(
    env,
    info,
):
    """
    해당 슬롯에서 실제로 service 받은 사용자만 serving BS를 기록한다.
    서비스받지 않은 사용자는 0.
    """
    served_map = {
        user.ue_id: 0
        for user in env.users
    }

    served_rates = info.get(
        "served_rates",
        {},
    )

    if "bs_selections" in info:
        for bs_id, ue_id in (
            info["bs_selections"].items()
        ):
            if ue_id is None:
                continue

            rate = float(
                served_rates.get(
                    ue_id,
                    0.0,
                )
            )

            if rate > 0.0:
                served_map[int(ue_id)] = int(
                    bs_id
                )

        return served_map

    if "served_bs_of_user" in info:
        source = info[
            "served_bs_of_user"
        ]

        for user in env.users:
            bs_id = source.get(
                user.ue_id,
                None,
            )

            if bs_id is not None:
                rate = float(
                    served_rates.get(
                        user.ue_id,
                        0.0,
                    )
                )

                if rate > 0.0:
                    served_map[
                        user.ue_id
                    ] = int(bs_id)

        return served_map

    return served_map


# ============================================================
# 10. One evaluation rollout
# ============================================================
@torch.no_grad()
def evaluate_one(
    algorithm: str,
    trainer,
    env,
    eval_seed: int,
):
    set_seed(eval_seed)

    local_obs, global_obs = (
        env.reset()
    )

    throughput_history = []
    slot_rates = []
    power_history = []
    handover_history = []
    request_history = []
    serving_event_history = []

    for step in range(
        STEPS_PER_EPISODE
    ):
        (
            ue_actions,
            bs_actions,
            cand_lists,
        ) = select_actions_actor_only(
            trainer=trainer,
            env=env,
            local_obs=local_obs,
            deterministic=DETERMINISTIC,
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
            float(
                info["total_throughput"]
            )
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

        handover_map = (
            extract_handover_map(
                info
            )
        )

        handover_history.append([
            float(
                handover_map[
                    user.ue_id
                ]
            )
            for user in env.users
        ])

        request_map = (
            extract_request_map(
                env=env,
                info=info,
                ue_actions=ue_actions,
            )
        )

        request_history.append([
            int(
                request_map[
                    user.ue_id
                ]
            )
            for user in env.users
        ])

        serving_event_map = (
            extract_serving_event_map(
                env=env,
                info=info,
            )
        )

        serving_event_history.append([
            int(
                serving_event_map[
                    user.ue_id
                ]
            )
            for user in env.users
        ])

        local_obs = next_local_obs
        global_obs = next_global_obs

        if (step + 1) % 1000 == 0:
            recent_throughput = float(
                np.mean(
                    throughput_history[-1000:]
                )
            )

            recent_on = float(
                np.mean(
                    np.asarray(
                        power_history[-1000:],
                        dtype=np.float64,
                    )
                    > 0.0
                )
            )

            recent_ho = float(
                np.mean(
                    np.asarray(
                        handover_history[-1000:],
                        dtype=np.float64,
                    )
                )
            )

            print(
                f"[{algorithm}] "
                f"step={step + 1:5d} | "
                f"Thr={recent_throughput:.4f} | "
                f"ON={recent_on:.4f} | "
                f"HO={recent_ho:.4f}"
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
    handover_arr = np.asarray(
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

    (
        req_dwell,
        req_switch,
    ) = compute_request_metrics(
        request_arr
    )

    cond_ho = compute_conditional_ho(
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
            np.mean(
                power_mat > 0.0
            )
        ),
        "ho_ratio": float(
            np.mean(handover_arr)
        ),
        "request_dwell": req_dwell,
        "request_switch": req_switch,
        "conditional_ho": cond_ho,
        "same_bs": 1.0 - cond_ho,
    }

    trajectory = {
        "throughput": throughput_arr,
        "slot_rates": slot_rates_arr,
        "power_mat": power_mat,
        "handover_mat": handover_arr,
        "request_bs_history": request_arr,
        "serving_bs_history":
            serving_event_arr,
    }

    return metrics, trajectory


# ============================================================
# 11. CSV helpers
# ============================================================
def save_csv(
    rows,
    path: str,
):
    if not rows:
        return

    os.makedirs(
        os.path.dirname(path),
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
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[CSV SAVED] {path}"
    )


# ============================================================
# 12. Summary
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
    raw_rows,
):
    per_train_rows = []

    for algorithm in ALGORITHMS:
        for train_seed in TRAIN_SEEDS:
            rows = [
                row
                for row in raw_rows
                if (
                    row["algorithm"]
                    == algorithm
                    and row["train_seed"]
                    == train_seed
                )
            ]

            if not rows:
                continue

            summary = {
                "algorithm": algorithm,
                "train_seed":
                    train_seed,
                "trained_num_users":
                    TRAINED_NUM_USERS,
                "eval_num_users":
                    EVAL_NUM_USERS,
                "kappa": KAPPA,
                "n_eval_seeds":
                    len(rows),
            }

            for metric in METRIC_NAMES:
                summary[metric] = float(
                    np.mean([
                        row[metric]
                        for row in rows
                    ])
                )

            per_train_rows.append(
                summary
            )

    final_rows = []

    for algorithm in ALGORITHMS:
        rows = [
            row
            for row in per_train_rows
            if row["algorithm"]
            == algorithm
        ]

        if not rows:
            continue

        summary = {
            "algorithm": algorithm,
            "trained_num_users":
                TRAINED_NUM_USERS,
            "eval_num_users":
                EVAL_NUM_USERS,
            "kappa": KAPPA,
            "n_train_seeds":
                len(rows),
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
                np.std(
                    values,
                    ddof=0,
                )
            )

        final_rows.append(
            summary
        )

    return (
        per_train_rows,
        final_rows,
    )


def print_final_summary(
    final_rows,
):
    print(
        "\n"
        + "=" * 170
    )

    print(
        "U=30 ZERO-SHOT SCALABILITY SUMMARY "
        "(trained with U=20, kappa=0.015)"
    )

    print(
        "Each train seed is first averaged over "
        "evaluation seeds; mean/std are across train seeds."
    )

    print(
        "=" * 170
    )

    for row in final_rows:
        print(
            f"{row['algorithm']:7s} | "
            f"Throughput="
            f"{row['throughput_mean']:.4f}"
            f" +/- "
            f"{row['throughput_std']:.4f} | "
            f"JFI="
            f"{row['jfi_mean']:.4f}"
            f" +/- "
            f"{row['jfi_std']:.4f} | "
            f"ON="
            f"{row['on_ratio_mean']:.4f}"
            f" +/- "
            f"{row['on_ratio_std']:.4f} | "
            f"HO="
            f"{row['ho_ratio_mean']:.4f}"
            f" +/- "
            f"{row['ho_ratio_std']:.4f} | "
            f"ReqSwitch="
            f"{row['request_switch_mean']:.4f}"
            f" +/- "
            f"{row['request_switch_std']:.4f} | "
            f"CondHO="
            f"{row['conditional_ho_mean']:.4f}"
            f" +/- "
            f"{row['conditional_ho_std']:.4f}"
        )

    print(
        "=" * 170
    )


# ============================================================
# 13. Main
# ============================================================
def main():
    raw_rows = []

    for algorithm in ALGORITHMS:
        print(
            "\n"
            + "#" * 120
        )
        print(
            f"START ALGORITHM: {algorithm}"
        )
        print(
            "#" * 120
        )

        for train_seed in TRAIN_SEEDS:
            model_path = get_model_path(
                algorithm=algorithm,
                train_seed=train_seed,
            )

            for eval_seed in EVAL_SEEDS:
                print(
                    "\n"
                    + "-" * 110
                )

                print(
                    f"[EVAL] "
                    f"algorithm={algorithm} | "
                    f"train_seed={train_seed} | "
                    f"eval_seed={eval_seed} | "
                    f"trained_U={TRAINED_NUM_USERS} | "
                    f"eval_U={EVAL_NUM_USERS} | "
                    f"kappa={KAPPA}"
                )

                env, trainer = (
                    build_env_and_trainer(
                        algorithm=algorithm,
                        eval_seed=eval_seed,
                    )
                )

                load_actor_only(
                    trainer=trainer,
                    model_path=model_path,
                )

                (
                    metrics,
                    trajectory,
                ) = evaluate_one(
                    algorithm=algorithm,
                    trainer=trainer,
                    env=env,
                    eval_seed=eval_seed,
                )

                seed_dir = os.path.join(
                    SAVE_ROOT,
                    algorithm,
                    f"seed_{train_seed}",
                )

                os.makedirs(
                    seed_dir,
                    exist_ok=True,
                )

                npz_path = os.path.join(
                    seed_dir,
                    f"eval_seed_{eval_seed}.npz",
                )

                np.savez_compressed(
                    npz_path,
                    tag=np.asarray(
                        f"{algorithm}_U30_kappa_0.015"
                    ),
                    algorithm=np.asarray(algorithm),
                    trained_num_users=np.asarray(
                        TRAINED_NUM_USERS
                    ),
                    eval_num_users=np.asarray(
                        EVAL_NUM_USERS
                    ),
                    kappa=np.asarray(KAPPA),
                    train_seed=np.asarray(train_seed),
                    eval_seed=np.asarray(eval_seed),

                    # Slot-wise trajectories
                    throughput=trajectory["throughput"],
                    slot_rates=trajectory["slot_rates"],
                    power_mat=trajectory["power_mat"],
                    handover_mat=trajectory["handover_mat"],
                    request_bs_history=trajectory[
                        "request_bs_history"
                    ],
                    serving_bs_history=trajectory[
                        "serving_bs_history"
                    ],

                    # Final scalar metrics
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

                print(
                    f"[FINAL RESULT] "
                    f"{algorithm} | "
                    f"train={train_seed} | "
                    f"eval={eval_seed} | "
                    f"Throughput="
                    f"{metrics['throughput']:.4f} | "
                    f"JFI="
                    f"{metrics['jfi']:.4f} | "
                    f"ON="
                    f"{metrics['on_ratio']:.4f} | "
                    f"HO="
                    f"{metrics['ho_ratio']:.4f} | "
                    f"ReqSwitch="
                    f"{metrics['request_switch']:.4f} | "
                    f"CondHO="
                    f"{metrics['conditional_ho']:.4f}"
                )

                print(
                    f"[NPZ SAVED] "
                    f"{npz_path}"
                )

                raw_rows.append({
                    "algorithm": algorithm,
                    "train_seed":
                        train_seed,
                    "eval_seed":
                        eval_seed,
                    "trained_num_users":
                        TRAINED_NUM_USERS,
                    "eval_num_users":
                        EVAL_NUM_USERS,
                    "kappa": KAPPA,
                    **metrics,
                })

                del trainer
                del env

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raw_csv_path = os.path.join(
        SAVE_ROOT,
        "all_raw_results.csv",
    )

    save_csv(
        raw_rows,
        raw_csv_path,
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