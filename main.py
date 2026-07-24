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

# 알고리즘 자체 차이만 먼저 보려면 [0.03] 권장
# 전체 kappa 비교 시 [0.01, 0.02, 0.03]으로 변경
KAPPA_LIST = [0.03]

STEPS_PER_EPISODE = 10000
TRAIN_EPISODES = 10
UPDATE_INTERVAL = 128

# 이미 학습한 모델만 평가하려면 False
RUN_TRAIN = True
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
             hard_window_len=10000):
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
def safe_mean(value):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else np.nan

def extract_eval_metrics(results):
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
        "pf_utility": safe_mean(
            results.get("pf_utility", [])
        ),
        "avg_energy_cost": safe_mean(
            results.get("avg_energy_cost", [])
        ),
        "performance_metric": safe_mean(
            results.get("performance_metric", [])
        ),
    }

METRIC_NAMES = [
    "throughput",
    "fairness",
    "on_ratio",
    "handover_ratio",
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
            f" +/- {row['handover_ratio_std']:.4f}"
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
    metrics = extract_eval_metrics(results)

    row = {
        "algorithm": algorithm,
        "kappa": float(kappa),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        **metrics,
        "model_path": model_path,
        "eval_npz_path": eval_npz_path,
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