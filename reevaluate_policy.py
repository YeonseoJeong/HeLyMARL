import os
import re
import glob
import shutil
import tempfile
import numpy as np

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.env_happo import HAPPOEnvironment
from baselines.env_constrainedhappo import (
    JensenHAPPOEnvironment,
    PFHAPPOEnvironment,
)
from HeLyMARL.trainer_happo import HAPPOTrainer


# ============================================================
# 1. Environment
# ============================================================
def make_env(
    seed,
    variant,
    V,
    lambda_E,
    kappa,
    use_hard_constraint,
    hard_window_len=10000,
):
    set_seed(seed)

    area_size = 100
    num_users = 20

    sbs_positions = generate_triangle_coverage(
        area_size,
        35,
    )

    sbs_list = [
        SmallCellBaseStation(
            i + 1,
            pos,
            10,
            35,
        )
        for i, pos in enumerate(sbs_positions)
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

    if variant == "helymarl":
        env = HAPPOEnvironment(
            base_stations=sbs_list,
            users=users,
            V=V,
            power_budget_ratio=0.6,
            enable_mobility=True,
            enable_channel_variation=True,
            on_window=100,
            bs_top_k=5,
            hard_window_len=hard_window_len,
            bs_over_penalty=100.0,
            use_hard_constraint=use_hard_constraint,
            lambda_E=lambda_E,
            kappa=kappa,
        )

    elif variant == "jensen":
        env = JensenHAPPOEnvironment(
            base_stations=sbs_list,
            users=users,
            power_budget_ratio=0.6,
            enable_mobility=True,
            enable_channel_variation=True,
            on_window=100,
            bs_top_k=5,
            hard_window_len=hard_window_len,
            use_hard_constraint=use_hard_constraint,
            kappa=kappa,
            use_dimensionless=True,
        )

    elif variant == "pf":
        env = PFHAPPOEnvironment(
            base_stations=sbs_list,
            users=users,
            power_budget_ratio=0.6,
            enable_mobility=True,
            enable_channel_variation=True,
            on_window=100,
            bs_top_k=5,
            hard_window_len=hard_window_len,
            use_hard_constraint=use_hard_constraint,
            kappa=kappa,
            use_dimensionless=True,
            pf_avg_beta=0.99,  # 학습 당시 값과 동일하게
        )

    else:
        raise ValueError(
            f"Unknown variant: {variant}"
        )

    return env


# ============================================================
# 2. Trainer
# ============================================================
def make_trainer(env):
    trainer = HAPPOTrainer(
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

    return trainer


# ============================================================
# 3. Checkpoint step 추출
# ============================================================
def extract_checkpoint_step(checkpoint_path):
    """
    checkpoint 파일명에서 training environment step을 추출한다.

    지원 예시
    ---------
    checkpoint_step_10000.pt
    policy_step_10000.pt
    checkpoint_10000.pt
    step10000.pt
    step_10000.pt
    update_80_step_10240.pt

    step 정보가 없으면 None 반환.
    """
    filename = os.path.basename(checkpoint_path)

    patterns = [
        r"(?:env_)?step[_-]?(\d+)",
        r"steps[_-]?(\d+)",
        r"checkpoint[_-]?(\d+)",
        r"policy[_-]?(\d+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            filename,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


def find_checkpoint_files(checkpoint_dir):
    """
    checkpoint_dir 아래의 .pt, .pth 파일을 재귀적으로 탐색한다.
    step 정보를 추출할 수 있는 checkpoint만 반환한다.
    """
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    candidate_files = []

    for extension in ("*.pt", "*.pth"):
        candidate_files.extend(
            glob.glob(
                os.path.join(
                    checkpoint_dir,
                    "**",
                    extension,
                ),
                recursive=True,
            )
        )

    checkpoint_entries = []

    for checkpoint_path in candidate_files:
        step = extract_checkpoint_step(
            checkpoint_path
        )

        if step is None:
            print(
                "[Warning] Training step을 추출할 수 없어 제외:"
                f"\n  {checkpoint_path}"
            )
            continue

        checkpoint_entries.append(
            {
                "step": step,
                "path": checkpoint_path,
            }
        )

    if not checkpoint_entries:
        raise RuntimeError(
            "step 정보를 포함한 checkpoint 모델을 찾지 못했습니다.\n"
            f"Checkpoint directory: {checkpoint_dir}"
        )

    # 같은 step 파일이 여러 개 있으면 가장 최근 수정 파일 사용
    step_to_entry = {}

    for entry in checkpoint_entries:
        step = entry["step"]
        path = entry["path"]

        if step not in step_to_entry:
            step_to_entry[step] = entry
            continue

        old_path = step_to_entry[step]["path"]

        if os.path.getmtime(path) > os.path.getmtime(old_path):
            step_to_entry[step] = entry

    checkpoint_entries = sorted(
        step_to_entry.values(),
        key=lambda item: item["step"],
    )

    return checkpoint_entries


# ============================================================
# 4. Evaluation NPZ에서 slot rate 불러오기
# ============================================================
def load_slot_rates(eval_npz_path, expected_num_users=20):
    """
    evaluation 결과 NPZ에서 slot_rates를 불러와 [U, T] 형태로 반환한다.

    지원 형태
    ---------
    [U, T]
    [T, U]
    [episode, U, T]
    [episode, T, U]
    """
    if not os.path.exists(eval_npz_path):
        raise FileNotFoundError(
            f"Evaluation NPZ not found: {eval_npz_path}"
        )

    with np.load(
        eval_npz_path,
        allow_pickle=True,
    ) as data:
        if "slot_rates" not in data.files:
            raise KeyError(
                f"'slot_rates' key not found in {eval_npz_path}\n"
                f"Available keys: {data.files}"
            )

        slot_rates = np.asarray(
            data["slot_rates"],
            dtype=np.float64,
        )

    # n_episodes=1일 때 첫 차원이 episode일 수 있음
    if slot_rates.ndim == 3:
        if slot_rates.shape[0] != 1:
            raise ValueError(
                "이 스크립트는 checkpoint별 n_episodes=1 평가를 "
                "가정합니다.\n"
                f"slot_rates shape: {slot_rates.shape}"
            )

        slot_rates = slot_rates[0]

    if slot_rates.ndim != 2:
        raise ValueError(
            "'slot_rates' must be 2-D after squeezing the "
            f"episode axis, but got shape {slot_rates.shape}"
        )

    # [T, U]이면 [U, T]로 transpose
    if (
        slot_rates.shape[0] != expected_num_users
        and slot_rates.shape[1] == expected_num_users
    ):
        slot_rates = slot_rates.T

    if slot_rates.shape[0] != expected_num_users:
        raise ValueError(
            "slot_rates의 user 축을 확인할 수 없습니다.\n"
            f"Expected users: {expected_num_users}\n"
            f"Actual shape: {slot_rates.shape}"
        )

    return slot_rates


# ============================================================
# 5. Eq. (7) objective 계산
# ============================================================
def compute_eq7_objective(
    slot_rates,
    eps=1e-12,
):
    """
    Eq. (7):
        sum_u log((1/T) sum_t R_u(t))

    Parameters
    ----------
    slot_rates : np.ndarray
        shape [U, T]

    eps : float
        log(0) 방지용 작은 값

    Returns
    -------
    objective : float
    """
    slot_rates = np.asarray(
        slot_rates,
        dtype=np.float64,
    )

    if slot_rates.ndim != 2:
        raise ValueError(
            f"slot_rates must be [U, T], got {slot_rates.shape}"
        )

    mean_user_rates = np.mean(
        slot_rates,
        axis=1,
    )

    objective = np.sum(
        np.log(
            np.maximum(
                mean_user_rates,
                eps,
            )
        )
    )

    return float(objective)


# ============================================================
# 6. 하나의 checkpoint를 여러 seed에서 평가
# ============================================================
def evaluate_one_checkpoint(
    checkpoint_path,
    checkpoint_step,
    eval_seeds,
    V,
    lambda_E,
    kappa,
    steps_per_episode,
    temporary_dir,
    expected_num_users=20,
):
    """
    동일 checkpoint를 여러 evaluation seed에서 평가하고
    Eq. (7) objective 목록을 반환한다.
    """
    objectives = []

    for eval_seed in eval_seeds:
        print(
            f"\n  Eval checkpoint step={checkpoint_step} "
            f"| seed={eval_seed}"
        )

        # seed마다 환경을 새로 생성해야
        # 사용자 위치, mobility, channel realization이 독립적으로 초기화됨
        eval_env = make_env(
            seed=eval_seed,
            variant="jensen",
            V=V,
            lambda_E=lambda_E,
            kappa=kappa,
            use_hard_constraint=True,
            hard_window_len=steps_per_episode,
        )

        trainer = make_trainer(
            eval_env
        )

        trainer.load_model(
            checkpoint_path
        )

        # evaluate 직전에도 동일 seed 고정
        set_seed(eval_seed)

        temporary_npz_path = os.path.join(
            temporary_dir,
            (
                f"checkpoint_step_{checkpoint_step}_"
                f"seed_{eval_seed}.npz"
            ),
        )

        trainer.evaluate(
            n_episodes=1,
            steps_per_episode=steps_per_episode,
            save_npz_path=temporary_npz_path,
        )

        slot_rates = load_slot_rates(
            temporary_npz_path,
            expected_num_users=expected_num_users,
        )

        objective = compute_eq7_objective(
            slot_rates
        )

        objectives.append(
            objective
        )

        print(
            f"    Eq. (7) objective = {objective:.6f}"
        )

    return np.asarray(
        objectives,
        dtype=np.float64,
    )


# ============================================================
# 7. 전체 checkpoint 재평가
# ============================================================
def reevaluate_policy_checkpoints(
    checkpoint_dir,
    output_npz_path,
    eval_seeds,
    V,
    lambda_E,
    kappa,
    steps_per_episode=10000,
    expected_num_users=20,
    include_step_zero=True,
):
    """
    저장된 모든 checkpoint를 5개 evaluation seed에서 재평가한다.

    저장 key
    --------
    policy_eval_steps
    policy_eval_objective_mean
    policy_eval_objective_std
    policy_eval_objective_all
    policy_eval_seeds
    policy_eval_episodes
    policy_eval_updates
    checkpoint_paths
    """
    checkpoint_entries = find_checkpoint_files(
        checkpoint_dir
    )

    if not include_step_zero:
        checkpoint_entries = [
            entry
            for entry in checkpoint_entries
            if entry["step"] > 0
        ]

    print("\n" + "=" * 100)
    print("Checkpoint evaluation configuration")
    print("=" * 100)
    print(f"Checkpoint directory : {checkpoint_dir}")
    print(f"Output NPZ           : {output_npz_path}")
    print(f"Number of checkpoints: {len(checkpoint_entries)}")
    print(f"Evaluation seeds     : {eval_seeds}")
    print(f"Evaluation horizon   : {steps_per_episode}")
    print(f"kappa                : {kappa}")
    print(f"lambda_E             : {lambda_E}")
    print("=" * 100)

    for entry in checkpoint_entries:
        print(
            f"Step {entry['step']:7d} | "
            f"{entry['path']}"
        )

    policy_eval_steps = []
    policy_eval_objective_mean = []
    policy_eval_objective_std = []
    policy_eval_objective_all = []
    checkpoint_paths = []

    temporary_dir = tempfile.mkdtemp(
        prefix="helymarl_checkpoint_eval_"
    )

    try:
        for checkpoint_index, entry in enumerate(
            checkpoint_entries,
            start=1,
        ):
            checkpoint_step = entry["step"]
            checkpoint_path = entry["path"]

            print("\n" + "=" * 100)
            print(
                f"[{checkpoint_index}/{len(checkpoint_entries)}] "
                f"Checkpoint step={checkpoint_step}"
            )
            print(f"Model: {checkpoint_path}")
            print("=" * 100)

            objectives = evaluate_one_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_step=checkpoint_step,
                eval_seeds=eval_seeds,
                V=V,
                lambda_E=lambda_E,
                kappa=kappa,
                steps_per_episode=steps_per_episode,
                temporary_dir=temporary_dir,
                expected_num_users=expected_num_users,
            )

            objective_mean = float(
                np.mean(objectives)
            )

            # sample standard deviation
            if len(objectives) >= 2:
                objective_std = float(
                    np.std(
                        objectives,
                        ddof=1,
                    )
                )
            else:
                objective_std = 0.0

            print(
                f"\n  Checkpoint step={checkpoint_step}"
                f"\n  Objectives = {objectives}"
                f"\n  Mean       = {objective_mean:.6f}"
                f"\n  Std        = {objective_std:.6f}"
            )

            policy_eval_steps.append(
                checkpoint_step
            )

            policy_eval_objective_mean.append(
                objective_mean
            )

            policy_eval_objective_std.append(
                objective_std
            )

            policy_eval_objective_all.append(
                objectives
            )

            checkpoint_paths.append(
                checkpoint_path
            )

    finally:
        shutil.rmtree(
            temporary_dir,
            ignore_errors=True,
        )

    policy_eval_steps = np.asarray(
        policy_eval_steps,
        dtype=np.int64,
    )

    policy_eval_objective_mean = np.asarray(
        policy_eval_objective_mean,
        dtype=np.float64,
    )

    policy_eval_objective_std = np.asarray(
        policy_eval_objective_std,
        dtype=np.float64,
    )

    policy_eval_objective_all = np.stack(
        policy_eval_objective_all,
        axis=0,
    )

    # plotting 코드와 호환되도록 저장
    policy_eval_episodes = (
        policy_eval_steps
        // steps_per_episode
    ).astype(np.int32)

    # checkpoint filename에서 정확한 update 수를 알 수 없으므로 -1
    policy_eval_updates = np.full(
        len(policy_eval_steps),
        -1,
        dtype=np.int32,
    )

    output_dir = (
        os.path.dirname(output_npz_path)
        if os.path.dirname(output_npz_path)
        else "."
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    np.savez_compressed(
        output_npz_path,
        policy_eval_steps=policy_eval_steps,
        policy_eval_objective_mean=policy_eval_objective_mean,
        policy_eval_objective_std=policy_eval_objective_std,
        policy_eval_objective_all=policy_eval_objective_all,
        policy_eval_seeds=np.asarray(
            eval_seeds,
            dtype=np.int64,
        ),
        policy_eval_episodes=policy_eval_episodes,
        policy_eval_updates=policy_eval_updates,
        checkpoint_paths=np.asarray(
            checkpoint_paths,
            dtype=object,
        ),
        V=np.asarray(V),
        lambda_E=np.asarray(lambda_E),
        kappa=np.asarray(kappa),
        eval_steps_per_episode=np.asarray(
            steps_per_episode
        ),
    )

    print("\n" + "=" * 100)
    print("Checkpoint evaluation summary")
    print("=" * 100)

    for step, mean, std in zip(
        policy_eval_steps,
        policy_eval_objective_mean,
        policy_eval_objective_std,
    ):
        print(
            f"Step {step:7d} | "
            f"Objective = {mean:.6f} ± {std:.6f}"
        )

    print("=" * 100)
    print(
        f"\n✅ Saved checkpoint objective results:\n"
        f"   {output_npz_path}"
    )


# ============================================================
# 8. Main
# ============================================================
if __name__ == "__main__":

    # --------------------------------------------------------
    # 동일 checkpoint를 평가할 환경 seed
    # --------------------------------------------------------
    CHECKPOINT_EVAL_SEEDS = [
        1000,
        1001,
        1002,
        1003,
        1004,
    ]

    V = 5.0
    LAMBDA_E = 0.0
    KAPPA = 0.03

    STEPS_PER_EPISODE = 10000
    EXPECTED_NUM_USERS = 20

    # --------------------------------------------------------
    # 실제 checkpoint 저장 폴더로 수정
    #
    # 예:
    # results/results_kappa/checkpoints
    # results/policy_improvement/HeLyMARL/checkpoints
    # --------------------------------------------------------
    CHECKPOINT_DIR = (
        "results/policy_improvement/jensen/"
        "checkpoints"
    )

    # plotting 코드에서 읽을 새 NPZ
    OUTPUT_NPZ_PATH = (
        "results/policy_improvement/jensen/"
        "jensen_policy_improvement_eval5seeds_"
        "lambda_0.0_kappa_0.03.npz"
    )

    reevaluate_policy_checkpoints(
        checkpoint_dir=CHECKPOINT_DIR,
        output_npz_path=OUTPUT_NPZ_PATH,
        eval_seeds=CHECKPOINT_EVAL_SEEDS,
        V=V,
        lambda_E=LAMBDA_E,
        kappa=KAPPA,
        steps_per_episode=STEPS_PER_EPISODE,
        expected_num_users=EXPECTED_NUM_USERS,

        # step=0 checkpoint가 실제로 저장돼 있으면 포함
        include_step_zero=True,
    )