import os
import sys
import csv
import gc
from collections import defaultdict

import numpy as np
import torch

# Add the project root so this script also works from experiments/.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.env_happo import HAPPOEnvironment
from HeLyMARL.trainer_happo import HAPPOTrainer
from HeLyMARL.trainer_mappo import MAPPOTrainer


# ============================================================
# 1. Evaluation settings
# ============================================================
ALGORITHMS = ["HAPPO"]
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

NUM_USERS = 20
V = 5.0
LAMBDA_E = 0.0

# ------------------------------------------------------------
# The model path is determined ONLY by the training condition.
# Existing models:
# results/results_kappa/HAPPO_kappa_0.015_seed_{0,1,2}/model.pt
# ------------------------------------------------------------
TRAIN_KAPPA = 0.015
TRAIN_ETA = 0.6  # recorded for labeling only; not used in model path
MODEL_ROOT = "results/results_kappa"

# ------------------------------------------------------------
# Evaluate the SAME trained model under these two hard budgets.
# eta corresponds to HAPPOEnvironment(power_budget_ratio=eta).
# ------------------------------------------------------------
EVAL_SETTINGS = [
    {
        "name": "tight_0.4",
        "kappa": 0.015,
        "eta": 0.4,
    },
    {
        "name": "original",
        "kappa": 0.015,
        "eta": 0.6,
    },
    {
        "name": "relaxed",
        "kappa": 0.020,
        "eta": 0.7,
    },
    {
        "name": "relaxed_0.8",
        "kappa": 0.020,
        "eta": 0.8,
    },
]

RUN_EVAL_NAMES = {"tight_0.4"}

STEPS_PER_EPISODE = 10000
OBJECTIVE_WINDOW = 10000
OBJECTIVE_EPS = 1e-12

# False: stochastic policy evaluation, matching the supplied code.
# True: deterministic action selection, if trainer.evaluate supports it.
STABILITY_DETERMINISTIC = False

# Evaluation results are saved separately from training folders,
# so existing eval_seed_*.npz files are never overwritten.
OUTPUT_ROOT = "results/eval_same_model_two_budgets"


# ============================================================
# 2. Environment / trainer
# ============================================================
def make_env(
    seed,
    V,
    lambda_E,
    kappa,
    eta,
    use_hard_constraint,
    hard_window_len=10000,
):
    set_seed(seed)

    area_size = 100
    sbs_positions = generate_triangle_coverage(area_size, 35)
    sbs_list = [
        SmallCellBaseStation(i + 1, pos, 10, 35)
        for i, pos in enumerate(sbs_positions)
    ]

    users = [
        UserEquipment(
            i + 1,
            (np.random.uniform(10, 90), np.random.uniform(10, 90)),
        )
        for i in range(NUM_USERS)
    ]

    return HAPPOEnvironment(
        base_stations=sbs_list,
        users=users,
        V=V,
        power_budget_ratio=eta,
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


def make_trainer(env, algorithm, eval_env=None):
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
        minibatch_size=256,
    )


# ============================================================
# 3. Paths
# ============================================================
def make_model_path(algorithm, train_seed):
    model_dir = os.path.join(
        MODEL_ROOT,
        f"{algorithm}_kappa_{TRAIN_KAPPA:.3f}_seed_{train_seed}",
    )
    return os.path.join(model_dir, "model.pt")


def make_setting_tag(eval_name, eval_kappa, eval_eta):
    return (
        f"{eval_name}_"
        f"eval_kappa_{eval_kappa:.3f}_"
        f"eta_{eval_eta:.3f}"
    )


def make_eval_dir(
    algorithm,
    train_seed,
    eval_name,
    eval_kappa,
    eval_eta,
):
    train_tag = (
        f"{algorithm}_trained_kappa_{TRAIN_KAPPA:.3f}_"
        f"eta_{TRAIN_ETA:.3f}_seed_{train_seed}"
    )
    setting_tag = make_setting_tag(eval_name, eval_kappa, eval_eta)
    return os.path.join(OUTPUT_ROOT, train_tag, setting_tag)


def make_eval_npz_path(
    algorithm,
    train_seed,
    eval_seed,
    eval_name,
    eval_kappa,
    eval_eta,
):
    eval_dir = make_eval_dir(
        algorithm=algorithm,
        train_seed=train_seed,
        eval_name=eval_name,
        eval_kappa=eval_kappa,
        eval_eta=eval_eta,
    )
    os.makedirs(eval_dir, exist_ok=True)
    return os.path.join(eval_dir, f"eval_seed_{eval_seed}.npz")


# ============================================================
# 4. Metric helpers
# ============================================================
def safe_mean(value):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else np.nan


def build_power_matrix(power_history):
    if not isinstance(power_history, dict) or len(power_history) == 0:
        return np.empty((0, 0), dtype=np.float64)

    rows = []
    for bs_id in sorted(power_history.keys()):
        row = np.asarray(power_history[bs_id], dtype=np.float64).reshape(-1)
        if row.size == 0:
            return np.empty((0, 0), dtype=np.float64)
        rows.append(row)

    common_length = min(row.size for row in rows)
    if common_length <= 0:
        return np.empty((0, 0), dtype=np.float64)

    return np.stack(
        [row[-common_length:] for row in rows],
        axis=0,
    )


def compute_objective_metric(
    results,
    lambda_E,
    objective_window=10000,
    eps=1e-12,
):
    slot_rates = np.asarray(
        results.get("slot_rates", []),
        dtype=np.float64,
    )

    if slot_rates.ndim == 2 and slot_rates.shape[0] > 0:
        rate_window = min(int(objective_window), int(slot_rates.shape[0]))
        recent_slot_rates = slot_rates[-rate_window:]
        avg_user_rates = np.mean(recent_slot_rates, axis=0)
        pf_utility = float(np.sum(np.log(avg_user_rates + float(eps))))
    else:
        avg_user_rates = np.asarray([], dtype=np.float64)
        pf_utility = np.nan

    power_mat = build_power_matrix(results.get("power_history", {}))

    if power_mat.ndim == 2 and power_mat.size > 0:
        power_window = min(int(objective_window), int(power_mat.shape[1]))
        recent_power_mat = power_mat[:, -power_window:]
        energy_per_slot = np.sum(recent_power_mat, axis=0)
        avg_energy_cost = float(np.mean(energy_per_slot))
    else:
        energy_per_slot = np.asarray([], dtype=np.float64)
        avg_energy_cost = np.nan

    if np.isfinite(pf_utility) and np.isfinite(avg_energy_cost):
        performance_metric = float(
            pf_utility - float(lambda_E) * avg_energy_cost
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
        results=results,
        lambda_E=lambda_E,
        objective_window=objective_window,
        eps=OBJECTIVE_EPS,
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


@torch.no_grad()
def evaluate_stability_metrics(
    trainer,
    env,
    eval_seed,
    steps_per_episode,
    deterministic=False,
):
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

        next_local_obs, next_global_obs, info, done = env.step_joint(
            ue_actions=ue_actions,
            bs_actions=bs_actions,
            cand_lists=cand_lists,
        )

        request_history.append(
            [int(ue_actions[user.ue_id]) for user in env.users]
        )

        served_bs_of_user = info.get("served_bs_of_user", {})
        served_bs_history.append(
            [
                0
                if served_bs_of_user.get(user.ue_id, None) is None
                else int(served_bs_of_user[user.ue_id])
                for user in env.users
            ]
        )

        local_obs = next_local_obs
        global_obs = next_global_obs

        if done:
            break

    request_arr = np.asarray(request_history, dtype=np.int64)
    served_arr = np.asarray(served_bs_history, dtype=np.int64)

    if request_arr.ndim == 2 and request_arr.shape[0] > 1:
        request_switch_flags = request_arr[1:] != request_arr[:-1]
        request_switch_ratio = float(np.mean(request_switch_flags))
        request_runs_per_user = np.sum(request_switch_flags, axis=0) + 1
        request_dwell_per_user = request_arr.shape[0] / np.maximum(
            request_runs_per_user,
            1,
        )
        request_dwell_time = float(np.mean(request_dwell_per_user))
    elif request_arr.ndim == 2 and request_arr.shape[0] == 1:
        request_switch_ratio = 0.0
        request_dwell_time = 1.0
    else:
        request_switch_ratio = np.nan
        request_dwell_time = np.nan

    total_service_transitions = 0
    total_service_switches = 0
    service_dwell_per_user = []

    if served_arr.ndim == 2:
        for user_idx in range(served_arr.shape[1]):
            service_sequence = served_arr[:, user_idx]
            service_sequence = service_sequence[service_sequence > 0]

            if service_sequence.size == 0:
                continue
            if service_sequence.size == 1:
                service_dwell_per_user.append(1.0)
                continue

            service_switch_flags = (
                service_sequence[1:] != service_sequence[:-1]
            )
            num_transitions = service_sequence.size - 1
            num_switches = int(np.sum(service_switch_flags))

            total_service_transitions += num_transitions
            total_service_switches += num_switches
            service_dwell_per_user.append(
                float(service_sequence.size / (num_switches + 1))
            )

    if total_service_transitions > 0:
        conditional_handover_ratio = float(
            total_service_switches / total_service_transitions
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
        "request_dwell_time": request_dwell_time,
        "request_switch_ratio": request_switch_ratio,
        "conditional_handover_ratio": conditional_handover_ratio,
        "same_bs_retention_ratio": same_bs_retention_ratio,
        "service_dwell_events": service_dwell_events,
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
# 5. CSV and aggregation
# ============================================================
def save_csv(rows, path):
    if not rows:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


def load_csv(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # The previous script used the name "relaxed" for eta=0.7.
    # Rename only the CSV label; the existing NPZ directories are untouched.
    for row in rows:
        try:
            is_old_relaxed = (
                row.get("eval_name") == "relaxed"
                and abs(float(row.get("eval_kappa", "nan")) - 0.020) < 1e-12
                and abs(float(row.get("eval_eta", "nan")) - 0.7) < 1e-12
            )
        except (TypeError, ValueError):
            is_old_relaxed = False

        if is_old_relaxed:
            row["eval_name"] = "relaxed_0.7"

    print(f"Loaded existing CSV: {path} ({len(rows)} rows)")
    return rows


def make_result_key(
    algorithm,
    eval_name,
    eval_kappa,
    eval_eta,
    train_seed,
    eval_seed,
):
    return (
        str(algorithm),
        str(eval_name),
        round(float(eval_kappa), 9),
        round(float(eval_eta), 9),
        int(train_seed),
        int(eval_seed),
    )


def row_result_key(row):
    return make_result_key(
        algorithm=row["algorithm"],
        eval_name=row["eval_name"],
        eval_kappa=row["eval_kappa"],
        eval_eta=row["eval_eta"],
        train_seed=row["train_seed"],
        eval_seed=row["eval_seed"],
    )


def aggregate_results(raw_rows):
    """
    1) Average eval seeds within each train seed and eval setting.
    2) Compute mean/std across train seeds.
    """
    per_train_groups = defaultdict(list)

    for row in raw_rows:
        key = (
            row["algorithm"],
            row["eval_name"],
            float(row["eval_kappa"]),
            float(row["eval_eta"]),
            int(row["train_seed"]),
        )
        per_train_groups[key].append(row)

    per_train_rows = []

    for key, rows in sorted(per_train_groups.items()):
        algorithm, eval_name, eval_kappa, eval_eta, train_seed = key

        summary = {
            "algorithm": algorithm,
            "train_kappa": TRAIN_KAPPA,
            "train_eta": TRAIN_ETA,
            "eval_name": eval_name,
            "eval_kappa": eval_kappa,
            "eval_eta": eval_eta,
            "train_seed": train_seed,
            "n_eval_seeds": len(rows),
        }

        for metric in METRIC_NAMES:
            summary[metric] = safe_mean([row[metric] for row in rows])

        per_train_rows.append(summary)

    final_groups = defaultdict(list)

    for row in per_train_rows:
        key = (
            row["algorithm"],
            row["eval_name"],
            float(row["eval_kappa"]),
            float(row["eval_eta"]),
        )
        final_groups[key].append(row)

    final_rows = []

    for key, rows in sorted(final_groups.items()):
        algorithm, eval_name, eval_kappa, eval_eta = key

        summary = {
            "algorithm": algorithm,
            "train_kappa": TRAIN_KAPPA,
            "train_eta": TRAIN_ETA,
            "eval_name": eval_name,
            "eval_kappa": eval_kappa,
            "eval_eta": eval_eta,
            "n_train_seeds": len(rows),
        }

        for metric in METRIC_NAMES:
            values = np.asarray(
                [row[metric] for row in rows],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]

            summary[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size > 0 else np.nan
            )
            summary[f"{metric}_std"] = (
                float(np.std(values)) if values.size > 0 else np.nan
            )

        final_rows.append(summary)

    return per_train_rows, final_rows


def print_final_summary(final_rows):
    print("\n" + "=" * 150)
    print(
        "SAME TRAINED MODEL / MULTIPLE EVALUATION BUDGETS\n"
        f"Train condition: kappa={TRAIN_KAPPA:.3f}, eta={TRAIN_ETA:.3f}\n"
        "Each train seed is first averaged over evaluation seeds; "
        "mean/std are then computed across train seeds."
    )
    print("=" * 150)

    for row in final_rows:
        print(
            f"{row['eval_name']:8s} | "
            f"eval kappa={row['eval_kappa']:.3f}, "
            f"eta={row['eval_eta']:.3f} | "
            f"Throughput={row['throughput_mean']:.4f}"
            f" +/- {row['throughput_std']:.4f} | "
            f"JFI={row['fairness_mean']:.4f}"
            f" +/- {row['fairness_std']:.4f} | "
            f"ON={row['on_ratio_mean']:.4f}"
            f" +/- {row['on_ratio_std']:.4f} | "
            f"HO={row['handover_ratio_mean']:.4f}"
            f" +/- {row['handover_ratio_std']:.4f} | "
            f"Served={row['served_ratio_mean']:.4f}"
            f" +/- {row['served_ratio_std']:.4f} | "
            f"Outage={row['outage_ratio_mean']:.4f}"
            f" +/- {row['outage_ratio_std']:.4f} | "
            f"CondHO={row['conditional_handover_ratio_mean']:.4f}"
            f" +/- {row['conditional_handover_ratio_std']:.4f} | "
            f"SameBS={row['same_bs_retention_ratio_mean']:.4f}"
            f" +/- {row['same_bs_retention_ratio_std']:.4f}"
        )

    print("=" * 150 + "\n")


# ============================================================
# 6. Evaluate one existing model
# ============================================================
def evaluate_one_model(
    algorithm,
    train_seed,
    eval_seed,
    eval_name,
    eval_kappa,
    eval_eta,
):
    model_path = make_model_path(algorithm, train_seed)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Check TRAIN_KAPPA and MODEL_ROOT:\n"
            f"  {model_path}"
        )

    eval_npz_path = make_eval_npz_path(
        algorithm=algorithm,
        train_seed=train_seed,
        eval_seed=eval_seed,
        eval_name=eval_name,
        eval_kappa=eval_kappa,
        eval_eta=eval_eta,
    )

    print("\n" + "=" * 110)
    print(
        f"EVAL | {algorithm} | "
        f"trained(kappa={TRAIN_KAPPA:.3f}, eta={TRAIN_ETA:.3f}) | "
        f"eval={eval_name}(kappa={eval_kappa:.3f}, eta={eval_eta:.3f}) | "
        f"train_seed={train_seed} | eval_seed={eval_seed}"
    )
    print("=" * 110)

    env_hard = make_env(
        seed=eval_seed,
        V=V,
        lambda_E=LAMBDA_E,
        kappa=eval_kappa,
        eta=eval_eta,
        use_hard_constraint=True,
        hard_window_len=STEPS_PER_EPISODE,
    )

    # Re-seed before network construction for reproducibility.
    set_seed(eval_seed)
    trainer = make_trainer(
        env=env_hard,
        algorithm=algorithm,
        eval_env=None,
    )
    trainer.load_model(model_path)
    set_seed(eval_seed)

    results = trainer.evaluate(
        n_episodes=1,
        steps_per_episode=STEPS_PER_EPISODE,
        save_npz_path=eval_npz_path,
    )

    metrics = extract_eval_metrics(
        results=results,
        lambda_E=LAMBDA_E,
        objective_window=OBJECTIVE_WINDOW,
    )

    # This performs a second rollout for request/serving stability metrics.
    stability_metrics = evaluate_stability_metrics(
        trainer=trainer,
        env=env_hard,
        eval_seed=eval_seed,
        steps_per_episode=STEPS_PER_EPISODE,
        deterministic=STABILITY_DETERMINISTIC,
    )

    print(
        f"[RESULT] Throughput={metrics['throughput']:.6f} | "
        f"JFI={metrics['fairness']:.6f} | "
        f"ON={metrics['on_ratio']:.6f} | "
        f"HO={metrics['handover_ratio']:.6f} | "
        f"Served={metrics['served_ratio']:.6f} | "
        f"Outage={metrics['outage_ratio']:.6f}"
    )

    row = {
        "algorithm": algorithm,
        "train_kappa": float(TRAIN_KAPPA),
        "train_eta": float(TRAIN_ETA),
        "eval_name": eval_name,
        "eval_kappa": float(eval_kappa),
        "eval_eta": float(eval_eta),
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "model_path": model_path,
        "eval_npz_path": eval_npz_path,
        **metrics,
        **stability_metrics,
    }

    del trainer
    del env_hard
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# ============================================================
# 7. Main: evaluation only
# ============================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    raw_csv_path = os.path.join(OUTPUT_ROOT, "raw_results.csv")

    # Preserve previous original / eta=0.7 evaluations and append eta=0.8.
    raw_rows = load_csv(raw_csv_path)
    existing_keys = {row_result_key(row) for row in raw_rows}

    # Check all models before running long evaluations.
    missing_models = []
    for algorithm in ALGORITHMS:
        for train_seed in TRAIN_SEEDS:
            model_path = make_model_path(algorithm, train_seed)
            if not os.path.exists(model_path):
                missing_models.append(model_path)

    if missing_models:
        print("Missing model files:")
        for path in missing_models:
            print(f"  - {path}")
        raise FileNotFoundError(
            "Evaluation stopped because one or more model files are missing."
        )

    for setting in EVAL_SETTINGS:
        eval_name = str(setting["name"])
        eval_kappa = float(setting["kappa"])
        eval_eta = float(setting["eta"])

        # Keep all settings in EVAL_SETTINGS for aggregation/labels,
        # but run only the newly requested eta=0.8 setting.
        if eval_name not in RUN_EVAL_NAMES:
            print(
                f"[KEEP] {eval_name}: existing evaluation is not rerun "
                f"(kappa={eval_kappa:.3f}, eta={eval_eta:.3f})"
            )
            continue

        for train_seed in TRAIN_SEEDS:
            for algorithm in ALGORITHMS:
                for eval_seed in EVAL_SEEDS:
                    result_key = make_result_key(
                        algorithm=algorithm,
                        eval_name=eval_name,
                        eval_kappa=eval_kappa,
                        eval_eta=eval_eta,
                        train_seed=train_seed,
                        eval_seed=eval_seed,
                    )

                    # Safe against accidentally launching the same run twice.
                    if result_key in existing_keys:
                        print(
                            f"[SKIP] Already in raw_results.csv | "
                            f"{algorithm} | {eval_name} | "
                            f"train_seed={train_seed} | eval_seed={eval_seed}"
                        )
                        continue

                    row = evaluate_one_model(
                        algorithm=algorithm,
                        train_seed=train_seed,
                        eval_seed=eval_seed,
                        eval_name=eval_name,
                        eval_kappa=eval_kappa,
                        eval_eta=eval_eta,
                    )
                    raw_rows.append(row)
                    existing_keys.add(result_key)

                    # Rewrite the combined CSV: old rows + newly completed row.
                    save_csv(raw_rows, raw_csv_path)

    if not raw_rows:
        print("No evaluation rows are available.")
        return

    # Aggregate both old conditions and the newly appended eta=0.8 condition.
    per_train_rows, final_rows = aggregate_results(raw_rows)

    save_csv(
        per_train_rows,
        os.path.join(OUTPUT_ROOT, "per_train_summary.csv"),
    )
    save_csv(
        final_rows,
        os.path.join(OUTPUT_ROOT, "final_summary.csv"),
    )

    print_final_summary(final_rows)
    print("Evaluation-only comparison completed.\n")


if __name__ == "__main__":
    main()
