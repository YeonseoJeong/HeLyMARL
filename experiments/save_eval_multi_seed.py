# save_eval_multi_seed.py

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import re
import numpy as np

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.env_happo import HAPPOEnvironment
from HeLyMARL.trainer_happo import HAPPOTrainer


# ============================================================
# 1. 실험 설정
# ============================================================
EVAL_SEEDS = [0, 1, 2, 3, 4]

STEPS_PER_EPISODE = 10000
EVAL_EPISODES = 1

AREA_SIZE = 100
NUM_USERS = 20
NUM_BS = 3

POWER_BUDGET_RATIO = 0.6
KAPPA = 0.03

HELYMARL_V = 5.0
HELYMARL_LAMBDA_E = 0.0

OVERWRITE_EXISTING = False

SAVE_ROOT = "results/results_multi_seed"
os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
# 2. 학습 모델 경로
# ============================================================
MODEL_PATHS = {
    "PF-HAPPO": (
        "results/results_baselines/"
        "ConstrainedHAPPO_pf_model_kappa_0.03_use_dimensionless.pt"
    ),
    "Jensen-HAPPO": (
        "results/results_baselines/"
        "ConstrainedHAPPO_jensen_model_kappa_0.03_use_dimensionless.pt"
    ),
    "HeLyMARL": (
        "results/results_kappa/"
        "HeLyMARL_model_kappa_0.03.pt"
    ),
}


# ============================================================
# 3. 공통 topology 생성
# ============================================================
def make_network(seed):
    """
    seed에 따라 초기 UE 위치를 변경합니다.
    """
    set_seed(seed)

    sbs_positions = generate_triangle_coverage(
        AREA_SIZE,
        35,
    )

    base_stations = [
        SmallCellBaseStation(
            i + 1,
            position,
            10,
            35,
        )
        for i, position in enumerate(sbs_positions)
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
# 4. HeLyMARL 환경 및 trainer
# ============================================================
def make_helymarl_env(seed):
    base_stations, users = make_network(seed)

    env = HAPPOEnvironment(
        base_stations=base_stations,
        users=users,
        V=HELYMARL_V,
        power_budget_ratio=POWER_BUDGET_RATIO,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=100,
        bs_top_k=5,
        hard_window_len=STEPS_PER_EPISODE,
        bs_over_penalty=100.0,
        use_hard_constraint=True,
        lambda_E=HELYMARL_LAMBDA_E,
        kappa=KAPPA,
    )

    return env


def make_helymarl_trainer(env):
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


# ============================================================
# 5. HeLyMARL seed별 평가
# ============================================================
def evaluate_helymarl(
    seed,
    save_npz_path,
):
    model_path = MODEL_PATHS["HeLyMARL"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"HeLyMARL model not found: {model_path}"
        )

    set_seed(seed)

    env = make_helymarl_env(seed)
    trainer = make_helymarl_trainer(env)

    trainer.load_model(model_path)

    # load_model 과정에서 RNG를 사용하는 경우를 방지하기 위해
    # 실제 evaluation 직전에 seed를 다시 설정합니다.
    set_seed(seed)

    trainer.evaluate(
        n_episodes=EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        save_npz_path=save_npz_path,
    )


# ============================================================
# 6. Constrained HAPPO 평가 adapter
# ============================================================
def evaluate_constrained_happo(
    algorithm,
    seed,
    save_npz_path,
):
    """
    PF-HAPPO와 Jensen-HAPPO의 기존 single-seed 평가 코드를
    이 함수에 연결합니다.

    algorithm:
        "PF-HAPPO"
        "Jensen-HAPPO"
    """

    if algorithm not in [
        "PF-HAPPO",
        "Jensen-HAPPO",
    ]:
        raise ValueError(
            f"Unknown constrained HAPPO algorithm: {algorithm}"
        )

    model_path = MODEL_PATHS[algorithm]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{algorithm} model not found: {model_path}"
        )

    set_seed(seed)

    base_stations, users = make_network(seed)

    # ========================================================
    # 아래 import와 환경 클래스 이름을 현재 프로젝트에 맞게 수정
    # ========================================================
    try:
        from baselines.env_constrainedhappo import (
            JensenHAPPOEnvironment,
            PFHAPPOEnvironment,
        )

        from HeLyMARL.trainer_happo import (
            HAPPOTrainer,
        )

    except ImportError as error:
        raise ImportError(
            "\nConstrained HAPPO import 경로를 현재 프로젝트에 "
            "맞게 수정해야 합니다.\n"
            "evaluate_constrained_happo() 내부의 import 부분을 "
            "확인하세요."
        ) from error

    if algorithm == "PF-HAPPO":
        env_class = PFHAPPOEnvironment
    else:
        env_class = JensenHAPPOEnvironment

    # ========================================================
    # 기존 PF/Jensen single-seed 평가 환경 생성 코드와
    # 동일하게 맞추면 됩니다.
    # ========================================================
    env = env_class(
        base_stations=base_stations,
        users=users,
        power_budget_ratio=POWER_BUDGET_RATIO,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=100,
        bs_top_k=5,
        hard_window_len=STEPS_PER_EPISODE,
        use_hard_constraint=True,
        kappa=KAPPA,
        use_dimensionless=True,
    )

    trainer = HAPPOTrainer(
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

    trainer.load_model(model_path)

    set_seed(seed)

    trainer.evaluate(
        n_episodes=EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        save_npz_path=save_npz_path,
    )


# ============================================================
# 7. DDPP 평가 adapter
# ============================================================
def evaluate_ddpp(
    seed,
    save_npz_path,
):
    set_seed(seed)

    from baselines.DDPP import DDPPAlgorithm

    base_stations, users = make_network(seed)

    ddpp = DDPPAlgorithm(
        base_stations=base_stations,
        users=users,
        V=5.0,
        power_budget_ratio=POWER_BUDGET_RATIO,
        max_slots=STEPS_PER_EPISODE,
        enable_mobility=True,
        enable_channel_variation=True,
        seed=seed,
        use_hard_constraint=True,
        hard_window_len=STEPS_PER_EPISODE,
        lambda_E=0.0,
        kappa=KAPPA,
    )

    ddpp.run_simulation()

    ddpp.save_results_npz(
        save_npz_path,
        tag=f"DDPP_seed_{seed}",
    )


# ============================================================
# 8. MaxSNR 평가 adapter
# ============================================================
def evaluate_maxsnr(
    seed,
    save_npz_path,
):
    set_seed(seed)

    from baselines.baselineMaxSNR import MaxSNRBaseline

    base_stations, users = make_network(seed)

    maxsnr = MaxSNRBaseline(
        base_stations=base_stations,
        users=users,
        power_budget_ratio=POWER_BUDGET_RATIO,
        max_slots=STEPS_PER_EPISODE,
        enable_mobility=True,
        enable_channel_variation=True,
        seed=seed,
        hard_window_len=STEPS_PER_EPISODE,
        lambda_E=0.0,
        kappa=KAPPA,
    )

    maxsnr.run_simulation()

    maxsnr.save_results_npz(
        save_npz_path,
        tag=f"MaxSNR_seed_{seed}",
    )


# ============================================================
# 9. 알고리즘별 평가 실행
# ============================================================
def run_single_evaluation(
    algorithm,
    seed,
    save_npz_path,
):
    if algorithm == "DDPP":
        evaluate_ddpp(
            seed=seed,
            save_npz_path=save_npz_path,
        )

    elif algorithm == "MaxSNR":
        evaluate_maxsnr(
            seed=seed,
            save_npz_path=save_npz_path,
        )

    elif algorithm in [
        "PF-HAPPO",
        "Jensen-HAPPO",
    ]:
        evaluate_constrained_happo(
            algorithm=algorithm,
            seed=seed,
            save_npz_path=save_npz_path,
        )

    elif algorithm == "HeLyMARL":
        evaluate_helymarl(
            seed=seed,
            save_npz_path=save_npz_path,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}"
        )


# ============================================================
# 10. power_mat에서 BS ON/OFF matrix 추출
# ============================================================
def load_bs_on_matrix(npz_path):
    """
    반환:
        bs_on_mat: [B, T]

    지원 key:
        power_mat
        bs_on_mat
        power_bs1, power_bs2, ...
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Evaluation result not found: {npz_path}"
        )

    with np.load(npz_path, allow_pickle=True) as data:

        # ----------------------------------------------------
        # Case 1: power_mat
        # ----------------------------------------------------
        if "power_mat" in data.files:
            power_mat = np.asarray(
                data["power_mat"],
                dtype=float,
            )

            power_mat = np.squeeze(power_mat)

            if power_mat.ndim != 2:
                raise ValueError(
                    f"'power_mat' must be 2-D, "
                    f"but got {power_mat.shape}"
                )

            # [T, B] -> [B, T]
            if (
                power_mat.shape[0] > power_mat.shape[1]
                and power_mat.shape[1] <= 20
            ):
                power_mat = power_mat.T

            return (
                power_mat > 0.0
            ).astype(np.float32)

        # ----------------------------------------------------
        # Case 2: bs_on_mat
        # ----------------------------------------------------
        if "bs_on_mat" in data.files:
            bs_on_mat = np.asarray(
                data["bs_on_mat"],
                dtype=float,
            )

            bs_on_mat = np.squeeze(bs_on_mat)

            if bs_on_mat.ndim != 2:
                raise ValueError(
                    f"'bs_on_mat' must be 2-D, "
                    f"but got {bs_on_mat.shape}"
                )

            if (
                bs_on_mat.shape[0] > bs_on_mat.shape[1]
                and bs_on_mat.shape[1] <= 20
            ):
                bs_on_mat = bs_on_mat.T

            return (
                bs_on_mat > 0.0
            ).astype(np.float32)

        # ----------------------------------------------------
        # Case 3: power_bs1, power_bs2, ...
        # ----------------------------------------------------
        power_keys = [
            key
            for key in data.files
            if key.startswith("power_bs")
        ]

        power_keys.sort(
            key=lambda key: int(
                re.search(r"\d+", key).group()
            )
        )

        if power_keys:
            power_list = []

            for key in power_keys:
                power = np.asarray(
                    data[key],
                    dtype=float,
                ).reshape(-1)

                power_list.append(power)

            min_length = min(
                len(power)
                for power in power_list
            )

            power_mat = np.stack(
                [
                    power[:min_length]
                    for power in power_list
                ],
                axis=0,
            )

            return (
                power_mat > 0.0
            ).astype(np.float32)

        raise KeyError(
            f"No usable BS ON/OFF data in {npz_path}\n"
            f"Available keys: {data.files}"
        )
    
# ============================================================
# 10-1. Seed별 evaluation 결과에서
#       cumulative handover ratio trajectory 추출
# ============================================================
def load_handover_trajectory(npz_path):
    """
    반환:
        handover_trajectory: [T]

    우선순위:
        1. handover_ratio_trajectory
           이미 cumulative ratio인 경우

        2. handover_ratio
           저장된 값이 [T] trajectory인 경우

        3. handover_ratio_step
           매 slot의 handover 비율인 경우 cumulative로 변환

        4. handover_indicator_mat
           [T, U] 또는 [U, T]에서 cumulative로 변환
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Evaluation result not found: {npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:

        # ----------------------------------------------------
        # Case 1: 이미 cumulative trajectory
        # ----------------------------------------------------
        cumulative_keys = [
            "handover_ratio_trajectory",
            "cumulative_handover_ratio",
            "handover_ratio_running",
        ]

        for key in cumulative_keys:
            if key not in data.files:
                continue

            trajectory = np.asarray(
                data[key],
                dtype=float,
            ).squeeze()

            if trajectory.ndim != 1:
                raise ValueError(
                    f"'{key}' must be 1-D, "
                    f"but got {trajectory.shape}"
                )

            return trajectory.astype(
                np.float32
            )

        # ----------------------------------------------------
        # Case 2: 기존 handover_ratio
        #
        # 저장된 handover_ratio가 slot별 사용자 평균 HO 값이라면
        # cumulative average trajectory로 변환한다.
        # ----------------------------------------------------
        if "handover_ratio" in data.files:
            handover_ratio_step = np.asarray(
                data["handover_ratio"],
                dtype=float,
            ).squeeze()

            if (
                handover_ratio_step.ndim == 1
                and handover_ratio_step.size > 1
            ):
                handover_ratio_step = np.nan_to_num(
                    handover_ratio_step,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                original_length = len(
                    handover_ratio_step
                )

                # 모든 알고리즘의 evaluation horizon을 10K로 통일
                if original_length < STEPS_PER_EPISODE:
                    handover_ratio_step = np.pad(
                        handover_ratio_step,
                        (
                            0,
                            STEPS_PER_EPISODE
                            - original_length,
                        ),
                        mode="constant",
                        constant_values=0.0,
                    )

                handover_ratio_step = (
                    handover_ratio_step[
                        :STEPS_PER_EPISODE
                    ]
                )

                denominator = np.arange(
                    1,
                    len(handover_ratio_step) + 1,
                    dtype=float,
                )

                handover_trajectory = (
                    np.cumsum(handover_ratio_step)
                    / denominator
                )

                print(
                    f"[HO LOAD] {npz_path}: "
                    f"raw length={original_length}, "
                    f"final length={len(handover_trajectory)}, "
                    f"final HO={handover_trajectory[-1]:.6f}"
                )

                return handover_trajectory.astype(
                    np.float32
                )
            
        # ----------------------------------------------------
        # Case 3: slot별 mean handover ratio
        # ----------------------------------------------------
        step_ratio_keys = [
            "handover_ratio_step",
            "handover_ratio_per_step",
            "handover_step_ratio",
        ]

        for key in step_ratio_keys:
            if key not in data.files:
                continue

            step_ratio = np.asarray(
                data[key],
                dtype=float,
            ).reshape(-1)

            denominator = np.arange(
                1,
                len(step_ratio) + 1,
                dtype=float,
            )

            trajectory = (
                np.cumsum(step_ratio)
                / denominator
            )

            return trajectory.astype(
                np.float32
            )

        # ----------------------------------------------------
        # Case 4: [T, U] 사용자별 handover indicator
        # ----------------------------------------------------
        indicator_keys = [
            "handover_indicator_mat",
            "handover_flags",
            "handover_mat",
        ]

        for key in indicator_keys:
            if key not in data.files:
                continue

            indicator_mat = np.asarray(
                data[key],
                dtype=float,
            ).squeeze()

            if indicator_mat.ndim != 2:
                raise ValueError(
                    f"'{key}' must be 2-D, "
                    f"but got {indicator_mat.shape}"
                )

            # [U, T] -> [T, U]
            if (
                indicator_mat.shape[0]
                < indicator_mat.shape[1]
            ):
                indicator_mat = (
                    indicator_mat.T
                )

            step_ratio = np.mean(
                indicator_mat,
                axis=1,
            )

            denominator = np.arange(
                1,
                len(step_ratio) + 1,
                dtype=float,
            )

            trajectory = (
                np.cumsum(step_ratio)
                / denominator
            )

            return trajectory.astype(
                np.float32
            )

        handover_keys = [
            key
            for key in data.files
            if (
                "handover" in key.lower()
                or key.lower().startswith("ho_")
            )
        ]

        raise KeyError(
            f"No usable handover trajectory in {npz_path}\n"
            f"Handover-related keys: {handover_keys}\n"
            f"Available keys: {data.files}"
        )

def to_finite_mean(values):
    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan

    return float(np.mean(values))


def calculate_jfi(user_rates, eps=1e-12):
    user_rates = np.asarray(
        user_rates,
        dtype=float,
    ).reshape(-1)

    user_rates = np.nan_to_num(
        user_rates,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    user_rates = np.maximum(
        user_rates,
        0.0,
    )

    numerator = np.sum(user_rates) ** 2
    denominator = (
        len(user_rates)
        * np.sum(user_rates ** 2)
        + eps
    )

    return float(
        numerator / denominator
    )

def block_jain_fairness(
    slot_rates,
    block_size=1000,
    eps=1e-12,
):
    """
    1000-slot 블록별 JFI를 계산한 뒤 평균.

    모든 UE의 평균 rate가 0인 all-off 블록은 제외한다.
    """
    rates = np.asarray(
        slot_rates,
        dtype=float,
    )

    rates = np.squeeze(rates)

    if rates.size == 0 or rates.ndim != 2:
        return np.nan

    # [U, T] -> [T, U]
    if rates.shape[0] < rates.shape[1]:
        rates = rates.T

    rates = np.nan_to_num(
        rates,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    rates = np.maximum(
        rates,
        0.0,
    )

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

        # 해당 블록에서 UE별 평균 rate
        block_user_rates = np.mean(
            block,
            axis=0,
        )

        # 완전 all-off 블록 제외
        if np.sum(
            block_user_rates ** 2
        ) <= eps:
            continue

        block_jfi = calculate_jfi(
            block_user_rates,
            eps=eps,
        )

        if np.isfinite(block_jfi):
            block_jfis.append(
                block_jfi
            )

    if len(block_jfis) == 0:
        return np.nan

    return float(
        np.mean(block_jfis)
    )

def load_eval_performance(npz_path):
    """
    하나의 seed evaluation 파일에서

        mean throughput
        mean fairness

    를 scalar로 반환합니다.
    """
    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:

        # ====================================================
        # Throughput
        # ====================================================
        throughput = np.nan

        throughput_keys = [
            "throughput",
            "throughput_history",
            "episode_throughput_mean",
            "mean_throughput",
        ]

        for key in throughput_keys:
            if key not in data.files:
                continue

            throughput = to_finite_mean(
                data[key]
            )

            if np.isfinite(throughput):
                break

        # ====================================================
        # Fairness
        # 1순위: slot_rates 기반 1000-slot block JFI
        #         all-off 블록 제외
        # ====================================================
        fairness = np.nan

        if "slot_rates" in data.files:
            fairness = block_jain_fairness(
                data["slot_rates"],
                block_size=1000,
            )

        # 저장된 block JFI가 있으면 fallback
        if (
            not np.isfinite(fairness)
            and "fairness_block_jfis" in data.files
        ):
            block_jfis = np.asarray(
                data["fairness_block_jfis"],
                dtype=float,
            ).reshape(-1)

            valid_mask = (
                np.isfinite(block_jfis)
                & (block_jfis > 0.0)
            )

            if np.any(valid_mask):
                fairness = float(
                    np.mean(
                        block_jfis[valid_mask]
                    )
                )

        # 기존 파일 호환용 fallback
        if not np.isfinite(fairness):
            fairness_keys = [
                "fairness",
                "episode_fairness_last",
                "mean_fairness",
                "jfi",
            ]

            for key in fairness_keys:
                if key not in data.files:
                    continue

                fairness = to_finite_mean(
                    data[key]
                )

                if np.isfinite(fairness):
                    break

        if (
            not np.isfinite(fairness)
            and "avg_user_rates" in data.files
        ):
            fairness = calculate_jfi(
                data["avg_user_rates"]
            )
    return throughput, fairness


def calculate_seed_statistics(values):
    """
    seed별 scalar 값의 평균, 분산, 표준편차.

    seed가 2개 이상이면 sample variance/std를 사용합니다.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return np.nan, np.nan, np.nan

    ddof = (
        1
        if finite_values.size > 1
        else 0
    )

    mean_value = float(
        np.mean(finite_values)
    )

    variance_value = float(
        np.var(
            finite_values,
            ddof=ddof,
        )
    )

    std_value = float(
        np.std(
            finite_values,
            ddof=ddof,
        )
    )

    return (
        mean_value,
        variance_value,
        std_value,
    )

# ============================================================
# 11. 알고리즘별 multi-seed summary 저장
# ============================================================
def save_algorithm_summary(
    algorithm,
    eval_paths,
    seeds,
):
    bs_on_matrices = []
    on_ratio_trajectories = []

    handover_trajectories = []

    throughput_per_seed = []
    fairness_per_seed = []

    successful_seeds = []

    for seed, eval_path in zip(
        seeds,
        eval_paths,
    ):
        try:
            bs_on_mat = load_bs_on_matrix(
                eval_path
            )

            handover_trajectory = (
                load_handover_trajectory(
                    eval_path
                )
            )

            T = min(
                STEPS_PER_EPISODE,
                bs_on_mat.shape[1],
                len(handover_trajectory),
            )

            bs_on_mat = bs_on_mat[:, :T]

            handover_trajectory = (
                handover_trajectory[:T]
            )

            mean_on_per_slot = np.mean(
                bs_on_mat,
                axis=0,
            )

            # 해당 seed의 throughput, fairness 추출
            throughput_value, fairness_value = (
                load_eval_performance(eval_path)
            )

            bs_on_matrices.append(
                bs_on_mat
            )

            on_ratio_trajectories.append(
                mean_on_per_slot
            )

            handover_trajectories.append(
                handover_trajectory
            )

            throughput_per_seed.append(
                throughput_value
            )

            fairness_per_seed.append(
                fairness_value
            )

            successful_seeds.append(
                seed
            )

            print(
                f"[{algorithm} | seed={seed}] "
                f"ON={np.mean(mean_on_per_slot):.6f}, "
                f"Final HO={handover_trajectory[-1]:.6f}, "
                f"Throughput={throughput_value:.6f}, "
                f"Fairness={fairness_value:.6f}"
            )

        except Exception as error:
            print(
                f"[WARNING] {algorithm}, seed={seed}: "
                f"{error}"
            )

    if not on_ratio_trajectories:
        raise RuntimeError(
            f"No valid trajectories for {algorithm}"
        )

    min_time_length = min(
        min(
            len(trajectory)
            for trajectory
            in on_ratio_trajectories
        ),
        min(
            len(trajectory)
            for trajectory
            in handover_trajectories
        ),
    )

    min_bs_count = min(
        matrix.shape[0]
        for matrix in bs_on_matrices
    )

    trajectory_per_seed = np.stack(
        [
            trajectory[:min_time_length]
            for trajectory in on_ratio_trajectories
        ],
        axis=0,
    ).astype(np.float32)
    # [S, T]


    handover_trajectory_per_seed = np.stack(
        [
            trajectory[:min_time_length]
            for trajectory in handover_trajectories
        ],
        axis=0,
    ).astype(np.float32)
    # [S, T]


    # ========================================================
    # Seed 통계의 자유도
    # 반드시 ON/HO 통계 계산 전에 정의
    # ========================================================
    ddof = (
        1
        if len(successful_seeds) > 1
        else 0
    )


    # ========================================================
    # Handover trajectory 통계
    # ========================================================
    handover_trajectory_mean = np.mean(
        handover_trajectory_per_seed,
        axis=0,
    ).astype(np.float32)

    handover_trajectory_var = np.var(
        handover_trajectory_per_seed,
        axis=0,
        ddof=ddof,
    ).astype(np.float32)

    handover_trajectory_std = np.std(
        handover_trajectory_per_seed,
        axis=0,
        ddof=ddof,
    ).astype(np.float32)


    overall_handover_ratio_per_seed = (
        handover_trajectory_per_seed[:, -1]
    )

    overall_handover_ratio_mean = float(
        np.mean(
            overall_handover_ratio_per_seed
        )
    )

    overall_handover_ratio_var = float(
        np.var(
            overall_handover_ratio_per_seed,
            ddof=ddof,
        )
    )

    overall_handover_ratio_std = float(
        np.std(
            overall_handover_ratio_per_seed,
            ddof=ddof,
        )
    )


    # ========================================================
    # BS ON matrix
    # ========================================================
    bs_on_mat_per_seed = np.stack(
        [
            matrix[
                :min_bs_count,
                :min_time_length,
            ]
            for matrix in bs_on_matrices
        ],
        axis=0,
    )
    # [S, B, T]


    # ========================================================
    # ON-ratio trajectory 통계
    # ========================================================
    trajectory_mean = np.mean(
        trajectory_per_seed,
        axis=0,
    )

    trajectory_var = np.var(
        trajectory_per_seed,
        axis=0,
        ddof=ddof,
    )

    trajectory_std = np.std(
        trajectory_per_seed,
        axis=0,
        ddof=ddof,
    )

    overall_on_ratio_per_seed = np.mean(
        trajectory_per_seed,
        axis=1,
    )

    overall_on_ratio_mean = np.mean(
        overall_on_ratio_per_seed
    )

    overall_on_ratio_var = np.var(
        overall_on_ratio_per_seed,
        ddof=ddof,
    )

    overall_on_ratio_std = np.std(
        overall_on_ratio_per_seed,
        ddof=ddof,
    )


    # ========================================================
    # Throughput/Fairness seed 통계
    # ========================================================
    throughput_per_seed = np.asarray(
        throughput_per_seed,
        dtype=float,
    )

    fairness_per_seed = np.asarray(
        fairness_per_seed,
        dtype=float,
    )

    (
        throughput_mean,
        throughput_var,
        throughput_std,
    ) = calculate_seed_statistics(
        throughput_per_seed
    )

    (
        fairness_mean,
        fairness_var,
        fairness_std,
    ) = calculate_seed_statistics(
        fairness_per_seed
    )

    safe_algorithm_name = (
        algorithm
        .replace("-", "_")
        .replace(" ", "_")
        .lower()
    )

    summary_path = os.path.join(
        SAVE_ROOT,
        (
            f"{safe_algorithm_name}_"
            f"{len(successful_seeds)}seeds_"
            "evaluation_summary.npz"
        ),
    )

    np.savez(
        summary_path,

        algorithm=np.asarray(
            algorithm
        ),

        eval_seeds=np.asarray(
            successful_seeds,
            dtype=int,
        ),

        eval_paths=np.asarray(
            [
                str(path)
                for path in eval_paths
            ],
        ),

        steps_per_episode=np.asarray(
            min_time_length,
            dtype=int,
        ),

        bs_on_mat_per_seed=(
            bs_on_mat_per_seed
        ),

        on_ratio_trajectory_per_seed=(
            trajectory_per_seed
        ),

        on_ratio_trajectory_mean=(
            trajectory_mean
        ),

        on_ratio_trajectory_var=(
            trajectory_var
        ),

        on_ratio_trajectory_std=(
            trajectory_std
        ),

        overall_on_ratio_per_seed=(
            overall_on_ratio_per_seed
        ),

        overall_on_ratio_mean=np.asarray(
            overall_on_ratio_mean,
            dtype=float,
        ),

        overall_on_ratio_var=np.asarray(
            overall_on_ratio_var,
            dtype=float,
        ),

        overall_on_ratio_std=np.asarray(
            overall_on_ratio_std,
            dtype=float,
        ),

                # Seed별 throughput 및 통계
        throughput_per_seed=(
            throughput_per_seed
        ),

        throughput_mean=np.asarray(
            throughput_mean,
            dtype=float,
        ),

        throughput_var=np.asarray(
            throughput_var,
            dtype=float,
        ),

        throughput_std=np.asarray(
            throughput_std,
            dtype=float,
        ),

        # Seed별 fairness 및 통계
        fairness_per_seed=(
            fairness_per_seed
        ),

        fairness_mean=np.asarray(
            fairness_mean,
            dtype=float,
        ),

        fairness_var=np.asarray(
            fairness_var,
            dtype=float,
        ),

        fairness_std=np.asarray(
            fairness_std,
            dtype=float,
        ),
        # ====================================================
        # Handover trajectory
        # ====================================================
        handover_ratio_trajectory_per_seed=(
            handover_trajectory_per_seed
        ),

        handover_ratio_trajectory_mean=(
            handover_trajectory_mean
        ),

        handover_ratio_trajectory_var=(
            handover_trajectory_var
        ),

        handover_ratio_trajectory_std=(
            handover_trajectory_std
        ),

        overall_handover_ratio_per_seed=(
            overall_handover_ratio_per_seed
        ),

        overall_handover_ratio_mean=np.asarray(
            overall_handover_ratio_mean,
            dtype=float,
        ),

        overall_handover_ratio_var=np.asarray(
            overall_handover_ratio_var,
            dtype=float,
        ),

        overall_handover_ratio_std=np.asarray(
            overall_handover_ratio_std,
            dtype=float,
        ),
    )

    print(
        f"\n[{algorithm}] summary saved:\n"
        f"{summary_path}"
    )

    return summary_path


# ============================================================
# 12. Main
# ============================================================
if __name__ == "__main__":

    algorithms = [
        "DDPP",
        "MaxSNR",
        "PF-HAPPO",
        "Jensen-HAPPO",
        "HeLyMARL",
    ]

    all_summary_paths = {}

    for algorithm in algorithms:
        print("\n")
        print("=" * 100)
        print(f"Multi-seed evaluation: {algorithm}")
        print("=" * 100)

        safe_algorithm_name = (
            algorithm
            .replace("-", "_")
            .replace(" ", "_")
            .lower()
        )

        algorithm_save_dir = os.path.join(
            SAVE_ROOT,
            safe_algorithm_name,
        )

        os.makedirs(
            algorithm_save_dir,
            exist_ok=True,
        )

        eval_paths = []
        completed_seeds = []

        for seed in EVAL_SEEDS:
            eval_npz_path = os.path.join(
                algorithm_save_dir,
                (
                    f"{safe_algorithm_name}_"
                    f"eval_seed_{seed}.npz"
                ),
            )

            # =================================================
            # 1. Seed별 evaluation 실행
            # =================================================
            should_run = (
                OVERWRITE_EXISTING
                or not os.path.exists(eval_npz_path)
            )

            if should_run:
                print(
                    f"\n[RUN] {algorithm}, seed={seed}"
                )
                print(
                    f"Save path: {eval_npz_path}"
                )

                try:
                    run_single_evaluation(
                        algorithm=algorithm,
                        seed=seed,
                        save_npz_path=eval_npz_path,
                    )

                except Exception as error:
                    print(
                        f"[ERROR] {algorithm}, seed={seed}: "
                        f"{type(error).__name__}: {error}"
                    )
                    continue

            else:
                print(
                    f"[SKIP] Existing result: "
                    f"{algorithm}, seed={seed}"
                )

            # =================================================
            # 2. 결과 파일 생성 여부 확인
            # =================================================
            if not os.path.exists(eval_npz_path):
                print(
                    f"[WARNING] Evaluation result was not saved: "
                    f"{eval_npz_path}"
                )
                continue

            print(
                f"[LOAD] {algorithm}, seed={seed}: "
                f"{eval_npz_path}"
            )

            eval_paths.append(
                eval_npz_path
            )

            completed_seeds.append(
                seed
            )

        # =====================================================
        # 3. Algorithm별 multi-seed summary 생성
        # =====================================================
        if not eval_paths:
            print(
                f"[WARNING] No successful results "
                f"for {algorithm}"
            )
            continue

        summary_path = save_algorithm_summary(
            algorithm=algorithm,
            eval_paths=eval_paths,
            seeds=completed_seeds,
        )

        all_summary_paths[algorithm] = (
            summary_path
        )

    print("\n")
    print("=" * 100)
    print("Completed multi-seed evaluations")
    print("=" * 100)

    for algorithm, summary_path in (
        all_summary_paths.items()
    ):
        print(
            f"{algorithm:<16}: {summary_path}"
        )

    print("=" * 100)