import os
import csv
import gc
from collections import defaultdict
import numpy as np
import torch

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.env_happo import HAPPOEnvironment
from HeLyMARL.trainer_happo import HAPPOTrainer
from HeLyMARL.trainer_mappo import MAPPOTrainer

# ============================================================
# Experiment settings
# ============================================================
ALGORITHMS = ["MAPPO", "HAPPO"]
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

V = 5.0
LAMBDA_E = 0.0
KAPPA_LIST = [0.03]

STEPS_PER_EPISODE = 10000
TRAIN_EPISODES = 10
UPDATE_INTERVAL = 128

OBJECTIVE_WINDOW = 10000
OBJECTIVE_EPS = 1e-12

RUN_TRAIN = False
RUN_EVAL = True

SAVE_DIR = "results/results_mappo_happo"

# ============================================================
# Environment
# ============================================================
def make_env(
    seed, 
    V, 
    lambda_E, 
    kappa, 
    use_hard_constraint, 
    hard_window_len=10000
):
    set_seed(seed)

    area_size = 100
    num_users = 20

    sbs_positions = generate_triangle_coverage(area_size, 35)
    sbs_list = [SmallCellBaseStation(i + 1, pos, 10, 35) for i, pos in enumerate(sbs_positions)]
    
    users = [
        UserEquipment(i + 1, (np.random.uniform(10, 90), np.random.uniform(10, 90)))
        for i in range(num_users)
    ]

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
        use_hard_constraint=use_hard_constraint,   # training: no hard constraint
        lambda_E=lambda_E,
        kappa=kappa
    )
    return env

# ============================================================
# Trainer
# ============================================================
def make_trainer(env, algorithm, eval_env = None):
    if algorithm == "MAPPO":
        trainer_class = MAPPOTrainer
    elif algorithm == "HAPPO":
        trainer_class = HAPPOTrainer
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    return trainer_class(
        env=env,
        eval_env=eval_env,
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
        minibatch_size=256
    )

# ============================================================
# File paths
# ============================================================
def make_run_dir(algorithm, kappa, train_seed):
    return os.path.join(SAVE_DIR, f"{algorithm}_kappa_{kappa:.2f}_seed_{train_seed}")

def make_model_path(algorithm, kappa, train_seed):
    run_dir = make_run_dir(algorithm, kappa, train_seed)
    return os.path.join(run_dir, "model.pt")

def make_train_npz_path(algorithm, kappa, train_seed):
    run_dir = make_run_dir(algorithm, kappa, train_seed)
    return os.path.join(run_dir, "train.npz")

def make_eval_npz_path(algorithm, kappa, train_seed, eval_seed):
    run_dir = make_run_dir(algorithm, kappa, train_seed)
    return os.path.join(run_dir, f"eval_seed_{eval_seed}.npz")

# ============================================================
# Metric helpers
# ============================================================
# ============================================================
# Standalone association-stability evaluation
# trainer 수정 없이 main.py에서 계산
# ============================================================
@torch.no_grad()
def evaluate_stability_metrics(
    trainer,
    env,
    eval_seed,
    steps_per_episode,
    deterministic=False,
):
    """
    Returns
    -------
    request_dwell_time:
        UE가 같은 BS를 연속 요청한 평균 슬롯 수

    request_switch_ratio:
        연속 슬롯 사이 UE 요청 BS가 변경된 비율

    conditional_handover_ratio:
        미서비스 슬롯을 제외한 service-event 사이에서
        실제 serving BS가 변경된 비율

    same_bs_retention_ratio:
        service-event 사이에서 같은 BS가 유지된 비율

    service_dwell_events:
        같은 serving BS가 연속된 평균 service-event 수
    """
    set_seed(int(eval_seed))

    trainer.ue_actor.eval()
    trainer.bs_actor.eval()
    trainer.critic.eval()

    local_obs, global_obs = env.reset()

    request_history = []
    served_bs_history = []

    for _ in range(steps_per_episode):
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
            deterministic=deterministic,
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

        # ----------------------------------------------------
        # 각 UE가 요청한 BS action
        # shape after rollout: [T, N]
        # ----------------------------------------------------
        request_history.append(
            [
                int(ue_actions[user.ue_id])
                for user in env.users
            ]
        )

        # ----------------------------------------------------
        # 각 UE를 실제로 서비스한 BS
        # 0 = 해당 슬롯에 서비스받지 않음
        # ----------------------------------------------------
        served_bs_of_user = info.get(
            "served_bs_of_user",
            {},
        )

        served_bs_history.append(
            [
                (
                    0
                    if served_bs_of_user.get(
                        user.ue_id,
                        None,
                    ) is None
                    else int(
                        served_bs_of_user[user.ue_id]
                    )
                )
                for user in env.users
            ]
        )

        local_obs = next_local_obs
        global_obs = next_global_obs

        if done:
            break

    request_arr = np.asarray(
        request_history,
        dtype=np.int64,
    )

    served_arr = np.asarray(
        served_bs_history,
        dtype=np.int64,
    )

    # ========================================================
    # 1. Request-BS stability
    # ========================================================
    if (
        request_arr.ndim == 2
        and request_arr.shape[0] > 1
    ):
        request_switch_flags = (
            request_arr[1:] != request_arr[:-1]
        )  # [T-1, N]

        request_switch_ratio = float(
            np.mean(request_switch_flags)
        )

        # UE별 연속 request run 개수
        request_runs_per_user = (
            np.sum(
                request_switch_flags,
                axis=0,
            )
            + 1
        )

        # UE별 평균 dwell을 계산한 뒤 사용자 평균
        request_dwell_per_user = (
            request_arr.shape[0]
            / np.maximum(
                request_runs_per_user,
                1,
            )
        )

        request_dwell_time = float(
            np.mean(request_dwell_per_user)
        )

    elif request_arr.shape[0] == 1:
        request_switch_ratio = 0.0
        request_dwell_time = 1.0

    else:
        request_switch_ratio = np.nan
        request_dwell_time = np.nan

    # ========================================================
    # 2. Actual serving-BS stability
    # 미서비스 슬롯은 제외
    # ========================================================
    total_service_transitions = 0
    total_service_switches = 0
    service_dwell_per_user = []

    if served_arr.ndim == 2:
        num_users = served_arr.shape[1]

        for user_idx in range(num_users):
            service_sequence = served_arr[:, user_idx]

            # 0: 미서비스 슬롯 제거
            service_sequence = service_sequence[
                service_sequence > 0
            ]

            if service_sequence.size == 0:
                continue

            if service_sequence.size == 1:
                service_dwell_per_user.append(1.0)
                continue

            service_switch_flags = (
                service_sequence[1:]
                != service_sequence[:-1]
            )

            num_transitions = (
                service_sequence.size - 1
            )

            num_switches = int(
                np.sum(service_switch_flags)
            )

            total_service_transitions += (
                num_transitions
            )

            total_service_switches += (
                num_switches
            )

            num_service_runs = num_switches + 1

            service_dwell_per_user.append(
                float(
                    service_sequence.size
                    / num_service_runs
                )
            )

    if total_service_transitions > 0:
        conditional_handover_ratio = float(
            total_service_switches
            / total_service_transitions
        )

        same_bs_retention_ratio = float(
            1.0 - conditional_handover_ratio
        )
    else:
        conditional_handover_ratio = 0.0
        same_bs_retention_ratio = 1.0

    service_dwell_events = (
        float(np.mean(service_dwell_per_user))
        if service_dwell_per_user
        else np.nan
    )

    return {
        "request_dwell_time":
            request_dwell_time,

        "request_switch_ratio":
            request_switch_ratio,

        "conditional_handover_ratio":
            conditional_handover_ratio,

        "same_bs_retention_ratio":
            same_bs_retention_ratio,

        "service_dwell_events":
            service_dwell_events,
    }

def safe_mean(value):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else np.nan

def build_power_matrix(power_history):
    if not isinstance(power_history, dict) or len(power_history) == 0:
        return np.empty((0, 0), dtype=np.float32)
    rows = []
    for bs_id in sorted(power_history.keys()):
        row = np.asarray(power_history[bs_id], dtype=np.float64).reshape(-1)
        if row.size == 0:
            return np.empty((0, 0), dtype=np.float64)
        rows.append(row)

    # 혹시 BS별 저장 길이가 다르면 공통 길이만 사용
    common_length = min(row.size for row in rows)
    if common_length <= 0:
        return np.empty((0, 0), dtype=np.float64)

    return np.stack(
        [row[-common_length:] for row in rows],
        axis=0,
    )

def compute_objective_metric(results, lambda_E, objective_window =10000, eps =1e-12):
    # --------------------------------------------------------
    # 1. PF utility
    # --------------------------------------------------------
    slot_rates = np.asarray(
        results.get("slot_rates", []),
        dtype=np.float64,
    )

    if slot_rates.ndim == 2 and slot_rates.shape[0] > 0:
        rate_window = min(
            int(objective_window),
            int(slot_rates.shape[0]),
        )
        recent_slot_rates = slot_rates[-rate_window:]
        avg_user_rates = np.mean(recent_slot_rates, axis=0)
        pf_utility = float(
            np.sum(
                np.log(avg_user_rates + float(eps))
            )
        )
    else:
        avg_user_rates = np.asarray([], dtype=np.float64)
        pf_utility = np.nan

    # --------------------------------------------------------
    # 2. Average energy cost
    # --------------------------------------------------------
    power_mat = build_power_matrix(
        results.get("power_history", {})
    )

    if power_mat.ndim == 2 and power_mat.size > 0:
        power_window = min(
            int(objective_window),
            int(power_mat.shape[1]),
        )
        recent_power_mat = power_mat[:, -power_window:]
        energy_per_slot = np.sum(recent_power_mat, axis=0)
        avg_energy_cost = float(np.mean(energy_per_slot))
    else:
        energy_per_slot = np.asarray([], dtype=np.float64)
        avg_energy_cost = np.nan

    # --------------------------------------------------------
    # 3. Energy-aware performance metric
    # --------------------------------------------------------
    if np.isfinite(pf_utility) and np.isfinite(avg_energy_cost):
        performance_metric = float(
            pf_utility
            - float(lambda_E) * avg_energy_cost
        )
    else:
        performance_metric = np.nan

    return {
        "pf_utility": pf_utility,
        "avg_energy_cost": avg_energy_cost,
        "performance_metric": performance_metric,
        "avg_user_rates": avg_user_rates,
        "energy_per_slot": energy_per_slot,
    }

def extract_eval_metrics(results, lambda_E, objective_window=10000):
    objective_metrics = compute_objective_metric(
        results = results,
        lambda_E = lambda_E,
        objective_window = objective_window,
        eps = OBJECTIVE_EPS
    )
    return {
        "throughput": safe_mean(
            results.get("episode_throughput_mean", [])
        ),
        "fairness": safe_mean(
            results.get("episode_fairness_last", [])
        ),
        "on_ratio": safe_mean(
            results.get("episode_on_ratio_mean", [])
        ),
        "handover_ratio": safe_mean(
            results.get("episode_handover_ratio_mean", [])
        ),
        "served_ratio": safe_mean(
            results.get("episode_served_ratio_mean", [])
        ),
        "outage_ratio": safe_mean(
            results.get("episode_outage_ratio_mean", [])
        ),
        "reward": safe_mean(
            results.get("episode_reward_mean", [])
        ),
        "pf_utility": objective_metrics["pf_utility"],
        "avg_energy_cost": objective_metrics["avg_energy_cost"],
        "performance_metric": objective_metrics["performance_metric"],
    }

METRIC_NAMES = [
    "throughput",
    "fairness",
    "on_ratio",
    "handover_ratio",
    "request_dwell_time",
    "request_switch_ratio",
    "conditional_handover_ratio",
    "same_bs_retention_ratio",
    "service_dwell_events",
    "served_ratio",
    "outage_ratio",
    "reward",
    "pf_utility",
    "avg_energy_cost",
    "performance_metric",
]


# ============================================================
# CSV helpers
# ============================================================
def save_csv(rows, path):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


def aggregate_results(raw_rows):
    """
    1) 동일 train seed의 여러 eval seed 결과를 먼저 평균
    2) train seed별 평균값에 대해 최종 mean/std 계산
    """
    per_train_groups = defaultdict(list)

    for row in raw_rows:
        key = (
            row["algorithm"],
            float(row["kappa"]),
            int(row["train_seed"]),
        )
        per_train_groups[key].append(row)

    per_train_rows = []

    for key, rows in sorted(per_train_groups.items()):
        algorithm, kappa, train_seed = key

        summary = {
            "algorithm": algorithm,
            "kappa": kappa,
            "train_seed": train_seed,
            "n_eval_seeds": len(rows),
        }

        for metric in METRIC_NAMES:
            summary[metric] = safe_mean(
                [row[metric] for row in rows]
            )

        per_train_rows.append(summary)

    final_groups = defaultdict(list)

    for row in per_train_rows:
        key = (
            row["algorithm"],
            float(row["kappa"]),
        )
        final_groups[key].append(row)

    final_rows = []

    for key, rows in sorted(final_groups.items()):
        algorithm, kappa = key

        summary = {
            "algorithm": algorithm,
            "kappa": kappa,
            "n_train_seeds": len(rows),
        }

        for metric in METRIC_NAMES:
            values = np.asarray(
                [row[metric] for row in rows],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]

            summary[f"{metric}_mean"] = (
                float(np.mean(values))
                if values.size > 0
                else np.nan
            )
            summary[f"{metric}_std"] = (
                float(np.std(values))
                if values.size > 0
                else np.nan
            )

        final_rows.append(summary)

    return per_train_rows, final_rows


def print_final_summary(final_rows):
    print("\n" + "=" * 125)
    print("MAPPO vs HAPPO FINAL SUMMARY")
    print(
        "Each train seed is first averaged over evaluation seeds; "
        "mean/std below are across train seeds."
    )
    print("=" * 125)

    for row in final_rows:
        print(
            f"kappa={row['kappa']:.2f} | "
            f"{row['algorithm']:5s} | "
            f"Throughput={row['throughput_mean']:.4f}"
            f" +/- {row['throughput_std']:.4f} | "
            f"JFI={row['fairness_mean']:.4f}"
            f" +/- {row['fairness_std']:.4f} | "
            f"Objective={row['performance_metric_mean']:.4f}"
            f" +/- {row['performance_metric_std']:.4f} | "
            f"ON={row['on_ratio_mean']:.4f}"
            f" +/- {row['on_ratio_std']:.4f} | "
            f"HO={row['handover_ratio_mean']:.4f}"
            f" +/- {row['handover_ratio_std']:.4f} | "
            f"ReqDwell={row['request_dwell_time_mean']:.2f}"
            f" +/- {row['request_dwell_time_std']:.2f} | "
            f"ReqSwitch={row['request_switch_ratio_mean']:.4f}"
            f" +/- {row['request_switch_ratio_std']:.4f} | "
            f"CondHO={row['conditional_handover_ratio_mean']:.4f}"
            f" +/- {row['conditional_handover_ratio_std']:.4f} | "
            f"SameBS={row['same_bs_retention_ratio_mean']:.4f}"
            f" +/- {row['same_bs_retention_ratio_std']:.4f}"
        )

    print("=" * 125 + "\n")

# ============================================================
# Train one model
# ============================================================
def train_one_model(algorithm, kappa, train_seed):
    run_dir = make_run_dir(algorithm, kappa, train_seed)
    os.makedirs(run_dir, exist_ok=True)
    model_path = make_model_path(algorithm, kappa, train_seed)
    train_npz_path = make_train_npz_path(algorithm, kappa, train_seed)

    print("\n" + "=" * 100)
    print(
        f"TRAIN | algorithm={algorithm} | "
        f"kappa={kappa:.2f} | "
        f"train_seed={train_seed}"
    )
    print("=" * 100)

    env_soft = make_env(seed=train_seed, V=V, lambda_E=LAMBDA_E, kappa=kappa, use_hard_constraint=False, hard_window_len=STEPS_PER_EPISODE)
    # Environment 생성 과정에서 RNG가 사용되므로
    # network initialization 직전에 다시 seed 고정
    set_seed(train_seed)
    trainer = make_trainer(env= env_soft, algorithm=algorithm, eval_env=None)
    trainer.train(
        n_episodes=TRAIN_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        update_interval=UPDATE_INTERVAL,
        save_npz_path=train_npz_path,
        eval_every=0,
        policy_improvement_dir=None,
        save_episode_end_checkpoint=False,
    )
    trainer.save_model(model_path)
    del trainer
    del env_soft
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# Evaluate one model
# ============================================================  
def evaluate_one_model(algorithm, kappa, train_seed, eval_seed):
    model_path = make_model_path(algorithm, kappa, train_seed)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Skipping evaluation.")
        
    eval_npz_path = make_eval_npz_path(algorithm, kappa, train_seed, eval_seed)

    print("\n" + "=" * 100)
    print(
        f"EVAL | algorithm={algorithm} | "
        f"kappa={kappa:.2f} | "
        f"train_seed={train_seed} | "
        f"eval_seed={eval_seed}"
    )
    print("=" * 100)

    env_hard = make_env(seed=eval_seed, V=V, lambda_E=LAMBDA_E, kappa=kappa, use_hard_constraint=True, hard_window_len=STEPS_PER_EPISODE)
    trainer = make_trainer(env=env_hard, algorithm=algorithm, eval_env=None)
    trainer.load_model(model_path)
    set_seed(eval_seed)

    results = trainer.evaluate(
        n_episodes=1,
        steps_per_episode=STEPS_PER_EPISODE,
        save_npz_path=eval_npz_path
    )
    metrics = extract_eval_metrics(results=results, lambda_E=LAMBDA_E, objective_window=OBJECTIVE_WINDOW)
    stability_metrics = evaluate_stability_metrics(
        trainer=trainer,
        env=env_hard,
        eval_seed=eval_seed,
        steps_per_episode=STEPS_PER_EPISODE,
        deterministic=False,
    )
    print(
        f"[OBJECTIVE] "
        f"PF={metrics['pf_utility']:.6f} | "
        f"AvgEnergy={metrics['avg_energy_cost']:.6f} | "
        f"Metric={metrics['performance_metric']:.6f}"
    )

    row = {
        "algorithm": algorithm,
        "kappa": float(kappa),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        **metrics,
        **stability_metrics,
    }

    del trainer
    del env_hard
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    raw_rows = []

    for kappa in KAPPA_LIST:
        for train_seed in TRAIN_SEEDS:
            for algorithm in ALGORITHMS:
                if RUN_TRAIN:
                    train_one_model(algorithm, kappa, train_seed)
                if RUN_EVAL:
                    for eval_seed in EVAL_SEEDS:
                        row = evaluate_one_model(algorithm, kappa, train_seed, eval_seed)
                        raw_rows.append(row)
                        save_csv(raw_rows, os.path.join(SAVE_DIR, "raw_results.csv"))

    if raw_rows:
        per_train_rows, final_rows = aggregate_results(raw_rows)
        save_csv(per_train_rows, os.path.join(SAVE_DIR, "per_train_summary.csv"))
        save_csv(final_rows, os.path.join(SAVE_DIR, "final_summary.csv"))
        print_final_summary(final_rows)
    print("\nMAPPO-HAPPO comparison completed!\n")

if __name__ == "__main__":
    main()