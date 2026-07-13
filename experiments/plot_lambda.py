import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Utility
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_get(data, key, default=None):
    if key in data.files:
        return data[key]
    return default

def safe_get_any(data, keys, default=None):
    for key in keys:
        if key in data.files:
            return data[key]
    return default


def moving_average(x, window=100):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def parse_lambda(path):
    """
    Supports:
    LyMARL_train_rewards_lambda_0.1.npz
    LyMARL_eval_rewards_lambda_1.0.npz
    """
    name = os.path.basename(path)

    patterns = [
        r"lambda_([0-9.]+)",
        r"lambdaE_([0-9.]+)",
        r"lambda_E_([0-9.]+)",
        r"lam_([0-9.]+)",
    ]

    for p in patterns:
        m = re.search(p, name)
        if m is not None:
            return float(m.group(1).rstrip("."))

    raise ValueError(f"Cannot parse lambda from filename: {name}")

def infer_eval_variant(path):
    name = os.path.basename(path).lower()

    if "soft" in name:
        return "soft"
    if "hard" in name:
        return "hard"

    # 기존 LyMARL_eval_rewards_lambda_*.npz 같은 파일이 남아있는 경우
    return "unknown"

def sorted_npz_files(result_dir, prefix, eval_variant="soft"):
    """
    prefix: 'train' or 'eval'

    eval_variant:
        'soft' -> only eval soft files
        'hard' -> only eval hard files
        'all'  -> all eval files
    """
    all_npz = glob.glob(os.path.join(result_dir, "**", "*.npz"), recursive=True)

    files = []
    for f in all_npz:
        name = os.path.basename(f).lower()

        if prefix.lower() not in name:
            continue
        if "lambda_" not in name:
            continue

        if prefix.lower() == "eval":
            variant = infer_eval_variant(f)

            if eval_variant in ["soft", "hard"]:
                if variant != eval_variant:
                    continue

            # eval_variant == "all"이면 hard/soft 모두 포함

        files.append(f)

    files = sorted(files, key=parse_lambda)
    return files


def load_npz(path):
    return np.load(path, allow_pickle=True)


# =========================================================
# Plot helpers
# =========================================================
def save_lineplot_multi(
    x_list, y_list, labels,
    xlabel, ylabel, title, save_path,
    hline=None, hline_label=None
):
    plt.figure(figsize=(7, 5))

    for x, y, label in zip(x_list, y_list, labels):
        if len(y) == 0:
            continue
        plt.plot(x, y, linewidth=2, label=label)

    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=2, label=hline_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_bar_summary(lams, values, xlabel, ylabel, title, save_path, budget=None, budget_label=None):
    plt.figure(figsize=(7, 5))
    x = np.arange(len(lams))
    plt.bar(x, values)
    plt.xticks(x, [str(v) for v in lams])
    if budget is not None:
        plt.axhline(budget, linestyle="--", linewidth=2, label=budget_label)
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# Train plots: one figure per metric, 5 lambda lines
# =========================================================
def plot_train_curves_all_lambdas(train_files, plot_dir, reward_window=10000):
    labels = [rf"$\lambda_E={parse_lambda(f)}$" for f in train_files]

    # -----------------------------------------------------
    # 1) Global reward
    # -----------------------------------------------------
    x_list, y_list = [], []
    for f in train_files:
        data = load_npz(f)

        reward_x_1000 = safe_get(data, "reward_x_1000", None)
        global_reward_1000 = safe_get(data, "global_reward_1000", None)
        global_reward = safe_get(data, "global_reward", np.array([]))

        if reward_x_1000 is not None and global_reward_1000 is not None and len(global_reward_1000) > 0:
            x = reward_x_1000
            y = global_reward_1000
        else:
            y = moving_average(global_reward, reward_window)
            x = np.arange(len(y)) + reward_window - 1 if len(global_reward) >= reward_window else np.arange(len(y))

        x_list.append(x)
        y_list.append(y)

    save_lineplot_multi(
        x_list, y_list, labels,
        xlabel="Training Step",
        ylabel="Global Reward",
        title="Training Global Reward",
        save_path=os.path.join(plot_dir, "train_global_reward_all_lambdas.png")
    )

    # -----------------------------------------------------
    # 2) Critic loss
    # -----------------------------------------------------
    x_list, y_list = [], []
    for f in train_files:
        data = load_npz(f)
        update_steps = safe_get(data, "update_steps", np.array([]))
        critic_loss = safe_get(data, "critic_loss", np.array([]))

        if len(critic_loss) == 0:
            continue
        if len(update_steps) == 0:
            update_steps = np.arange(len(critic_loss))

        x_list.append(update_steps[:len(critic_loss)])
        y_list.append(critic_loss)

    save_lineplot_multi(
        x_list, y_list, labels[:len(x_list)] if len(x_list) != len(labels) else labels,
        xlabel="Training Step",
        ylabel="Critic Loss",
        title="Training Critic Loss",
        save_path=os.path.join(plot_dir, "train_critic_loss_all_lambdas.png")
    )

    # -----------------------------------------------------
    # 3) UE actor loss
    # -----------------------------------------------------
    x_list, y_list, used_labels = [], [], []
    for f in train_files:
        data = load_npz(f)
        lam = parse_lambda(f)

        update_steps = safe_get(data, "update_steps", np.array([]))
        actor_ue_loss = safe_get(data, "actor_ue_loss", np.array([]))

        if len(actor_ue_loss) == 0:
            continue
        if len(update_steps) == 0:
            update_steps = np.arange(len(actor_ue_loss))

        x_list.append(update_steps[:len(actor_ue_loss)])
        y_list.append(actor_ue_loss)
        used_labels.append(rf"$\lambda_E={lam}$")

    save_lineplot_multi(
        x_list, y_list, used_labels,
        xlabel="Training Step",
        ylabel="UE Actor Loss",
        title="Training UE Actor Loss",
        save_path=os.path.join(plot_dir, "train_ue_actor_loss_all_lambdas.png")
    )

    # -----------------------------------------------------
    # 4) BS actor loss
    # -----------------------------------------------------
    x_list, y_list, used_labels = [], [], []
    for f in train_files:
        data = load_npz(f)
        lam = parse_lambda(f)

        update_steps = safe_get(data, "update_steps", np.array([]))
        actor_bs_loss = safe_get(data, "actor_bs_loss", np.array([]))

        if len(actor_bs_loss) == 0:
            continue
        if len(update_steps) == 0:
            update_steps = np.arange(len(actor_bs_loss))

        x_list.append(update_steps[:len(actor_bs_loss)])
        y_list.append(actor_bs_loss)
        used_labels.append(rf"$\lambda_E={lam}$")

    save_lineplot_multi(
        x_list, y_list, used_labels,
        xlabel="Training Step",
        ylabel="BS Actor Loss",
        title="Training BS Actor Loss",
        save_path=os.path.join(plot_dir, "train_bs_actor_loss_all_lambdas.png")
    )

    # -----------------------------------------------------
    # 5) Entropy (UE)
    # -----------------------------------------------------
    x_list, y_list, used_labels = [], [], []
    for f in train_files:
        data = load_npz(f)
        lam = parse_lambda(f)

        update_steps = safe_get(data, "update_steps", np.array([]))
        entropy_ue = safe_get(data, "entropy_ue", np.array([]))

        if len(entropy_ue) == 0:
            continue
        if len(update_steps) == 0:
            update_steps = np.arange(len(entropy_ue))

        x_list.append(update_steps[:len(entropy_ue)])
        y_list.append(entropy_ue)
        used_labels.append(rf"$\lambda_E={lam}$")

    save_lineplot_multi(
        x_list, y_list, used_labels,
        xlabel="Training Step",
        ylabel="UE Entropy Loss",
        title="Training UE Entropy",
        save_path=os.path.join(plot_dir, "train_entropy_ue_all_lambdas.png")
    )

    # -----------------------------------------------------
    # 6) Entropy (BS)
    # -----------------------------------------------------
    x_list, y_list, used_labels = [], [], []
    for f in train_files:
        data = load_npz(f)
        lam = parse_lambda(f)

        update_steps = safe_get(data, "update_steps", np.array([]))
        entropy_bs = safe_get(data, "entropy_bs", np.array([]))

        if len(entropy_bs) == 0:
            continue
        if len(update_steps) == 0:
            update_steps = np.arange(len(entropy_bs))

        x_list.append(update_steps[:len(entropy_bs)])
        y_list.append(entropy_bs)
        used_labels.append(rf"$\lambda_E={lam}$")

    save_lineplot_multi(
        x_list, y_list, used_labels,
        xlabel="Training Step",
        ylabel="BS Entropy Loss",
        title="Training BS Entropy",
        save_path=os.path.join(plot_dir, "train_entropy_bs_all_lambdas.png")
    )


# =========================================================
# Eval time-series plots: one figure per metric, 5 lambda lines
# =========================================================
def plot_queue_timeseries_all_lambdas(files, plot_dir, prefix, smooth_window=1000, eval_variant=None):
    """
    Plot Q, Z, G mean queue trajectories.

    One figure per lambda.
    x-axis: step
    y-axis: mean queue value
    legend: Q, Z, G
    """

    for f in files:
        data = load_npz(f)
        lam = parse_lambda(f)

        Q = safe_get(data, "Q_mean", np.array([]))
        Z = safe_get(data, "Z_mean", np.array([]))
        G = safe_get(data, "G_mean", np.array([]))

        if len(Q) == 0 and len(Z) == 0 and len(G) == 0:
            print(f"[Warning] No Q_mean/Z_mean/G_mean found in {f}")
            print(f"Available keys: {data.files}")
            continue

        plt.figure(figsize=(7, 5))

        for y, name in [(Q, "Q"), (Z, "Z"), (G, "G")]:
            if len(y) == 0:
                continue

            y_smooth = moving_average(y, smooth_window)

            if len(y) >= smooth_window:
                x = np.arange(len(y_smooth)) + smooth_window - 1
            else:
                x = np.arange(len(y_smooth))

            plt.plot(x, y_smooth, linewidth=2, label=name)

        plt.xlabel("Step")
        plt.ylabel("Mean Queue Value")
        plt.title(prefix.capitalize() + " Mean Queue Trajectories " + rf"$(\lambda_E={lam})$")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if eval_variant is not None:
            save_name = f"{prefix}_{eval_variant}_mean_queue_QZG_lambda_{lam:g}.png"
        else:
            save_name = f"{prefix}_mean_queue_QZG_lambda_{lam:g}.png"
        plt.savefig(os.path.join(plot_dir, save_name), dpi=300)
        plt.close()

# =========================================================
# Lambda sensitivity metrics
# =========================================================
def normalize_power_mat(power_mat, num_bs=None):
    """
    Convert power_mat to shape: (num_bs, num_steps)

    Expected common formats:
        (num_bs, num_steps)
        (num_steps, num_bs)
    """
    power_mat = np.asarray(power_mat)

    if power_mat.ndim != 2 or power_mat.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if num_bs is not None:
        if power_mat.shape[0] == num_bs:
            return power_mat
        if power_mat.shape[1] == num_bs:
            return power_mat.T

    # 일반적으로 BS 수보다 time-step 수가 훨씬 큼
    if power_mat.shape[0] > power_mat.shape[1]:
        power_mat = power_mat.T

    return power_mat


def compute_tracking_rmse(
    power_mat,
    target_ratio,
    window=500,
    num_bs=None
):
    """
    Sliding-window BS ON-ratio와 target_ratio 사이의 RMSE.

    BS별 sliding-window ON-ratio:
        rho_hat_b(t) = (1/W) sum y_b(tau)

    Tracking RMSE:
        sqrt(mean_{b,t}[(rho_hat_b(t) - rho)^2])

    작은 값일수록 각 BS가 목표 ON-ratio를 시간적으로
    안정적으로 추종한다는 의미.
    """
    power_mat = normalize_power_mat(power_mat, num_bs=num_bs)

    if power_mat.size == 0:
        return np.nan

    bs_on_mat = (power_mat > 0.0).astype(np.float32)
    n_bs, n_steps = bs_on_mat.shape

    if n_steps == 0:
        return np.nan

    window = max(1, min(int(window), n_steps))
    kernel = np.ones(window, dtype=np.float32) / float(window)

    window_on_ratios = []

    for b in range(n_bs):
        local_ratio = np.convolve(
            bs_on_mat[b],
            kernel,
            mode="valid"
        )
        window_on_ratios.append(local_ratio)

    window_on_ratios = np.asarray(
        window_on_ratios,
        dtype=np.float32
    )

    tracking_rmse = np.sqrt(
        np.mean((window_on_ratios - target_ratio) ** 2)
    )

    return float(tracking_rmse)


def compute_budget_lifetime(
    power_mat,
    energy_budget_ratio,
    num_bs=None
):
    """
    BS별 에너지 예산 소진 시점을 계산하고 평균하여 정규화.

    fixed per-slot energy consumption을 가정하면,
    BS b의 허용 ON 횟수는 대략

        floor(rho * T)

    이다.

    lifetime_b:
        cumulative ON count가 budget에 도달한 시점 / T

    system-level budget lifetime:
        BS별 lifetime의 평균

    반환값:
        0 ~ 1

    1에 가까울수록 BS들이 horizon 끝까지 예산을 유지.
    """
    power_mat = normalize_power_mat(power_mat, num_bs=num_bs)

    if power_mat.size == 0:
        return np.nan

    bs_on_mat = (power_mat > 0.0).astype(np.int64)
    n_bs, n_steps = bs_on_mat.shape

    if n_steps == 0 or np.isnan(energy_budget_ratio):
        return np.nan

    # 각 BS가 horizon 동안 사용할 수 있는 최대 ON 횟수
    max_on_count = int(
        np.floor(float(energy_budget_ratio) * n_steps)
    )

    if max_on_count <= 0:
        return 0.0

    lifetime_per_bs = []

    for b in range(n_bs):
        cumulative_on = np.cumsum(bs_on_mat[b])

        exhausted_indices = np.where(
            cumulative_on >= max_on_count
        )[0]

        if len(exhausted_indices) == 0:
            # horizon 내에서 예산을 모두 사용하지 않음
            exhaustion_step = n_steps
        else:
            # index가 0부터 시작하므로 실제 step 수는 +1
            exhaustion_step = int(exhausted_indices[0] + 1)

        normalized_lifetime = exhaustion_step / float(n_steps)
        lifetime_per_bs.append(normalized_lifetime)

    return float(np.mean(lifetime_per_bs))


def summarize_lambda_sensitivity(
    path,
    tracking_window=500,
    metric_window=None
):
    """
    하나의 eval npz 파일에서 다음 지표를 계산:
        1. average throughput
        2. average JFI
        3. tracking RMSE
        4. budget lifetime

    metric_window:
        None  -> 전체 evaluation horizon 평균
        정수값 -> 마지막 metric_window 구간 평균
    """
    data = load_npz(path)

    throughput = safe_get(
        data,
        "throughput",
        np.array([])
    )

    fairness = safe_get(
        data,
        "fairness",
        np.array([])
    )

    power_mat = safe_get(
        data,
        "power_mat",
        np.zeros((0, 0), dtype=np.float32)
    )

    energy_budget_arr = safe_get(
        data,
        "energy_budget_ratio",
        np.array([np.nan])
    )

    if energy_budget_arr is None or len(energy_budget_arr) == 0:
        energy_budget_ratio = np.nan
    else:
        energy_budget_ratio = float(
            np.asarray(energy_budget_arr).reshape(-1)[0]
        )

    # -----------------------------------------------------
    # 평균을 계산할 evaluation 구간
    # -----------------------------------------------------
    if metric_window is None:
        throughput_used = throughput
        fairness_used = fairness
    else:
        throughput_used = throughput[-metric_window:]
        fairness_used = fairness[-metric_window:]

    avg_throughput = (
        float(np.mean(throughput_used))
        if np.asarray(throughput_used).size > 0
        else np.nan
    )

    avg_jfi = (
        float(np.mean(fairness_used))
        if np.asarray(fairness_used).size > 0
        else np.nan
    )

    tracking_rmse = compute_tracking_rmse(
        power_mat=power_mat,
        target_ratio=energy_budget_ratio,
        window=tracking_window
    )

    budget_lifetime = compute_budget_lifetime(
        power_mat=power_mat,
        energy_budget_ratio=energy_budget_ratio
    )

    return {
        "lambda": parse_lambda(path),
        "avg_throughput": avg_throughput,
        "avg_jfi": avg_jfi,
        "tracking_rmse": tracking_rmse,
        "budget_lifetime": budget_lifetime,
        "energy_budget_ratio": energy_budget_ratio
    }


def plot_lambda_sensitivity_four_metrics(
    eval_files,
    plot_dir,
    tracking_window=500,
    metric_window=None,
    eval_variant="hard"
):
    """
    2x2 subplot:

        (a) Average throughput
        (b) Average JFI
        (c) Energy-budget tracking RMSE
        (d) Budget lifetime
    """
    if len(eval_files) == 0:
        print("[Warning] No evaluation files.")
        return

    summaries = [
        summarize_lambda_sensitivity(
            path=f,
            tracking_window=tracking_window,
            metric_window=metric_window
        )
        for f in eval_files
    ]

    summaries = sorted(
        summaries,
        key=lambda item: item["lambda"]
    )

    lambdas = np.asarray(
        [s["lambda"] for s in summaries],
        dtype=np.float32
    )

    avg_throughput = np.asarray(
        [s["avg_throughput"] for s in summaries],
        dtype=np.float32
    )

    avg_jfi = np.asarray(
        [s["avg_jfi"] for s in summaries],
        dtype=np.float32
    )

    tracking_rmse = np.asarray(
        [s["tracking_rmse"] for s in summaries],
        dtype=np.float32
    )

    budget_lifetime = np.asarray(
        [s["budget_lifetime"] for s in summaries],
        dtype=np.float32
    )

    # -----------------------------------------------------
    # 결과 출력
    # -----------------------------------------------------
    print("\n" + "=" * 90)
    print("Lambda sensitivity summary")
    print("=" * 90)
    print(
        f"{'lambda_E':>10} | "
        f"{'Throughput':>12} | "
        f"{'JFI':>10} | "
        f"{'Tracking RMSE':>15} | "
        f"{'Budget Lifetime':>16}"
    )
    print("-" * 90)

    for s in summaries:
        print(
            f"{s['lambda']:>10.4g} | "
            f"{s['avg_throughput']:>12.4f} | "
            f"{s['avg_jfi']:>10.4f} | "
            f"{s['tracking_rmse']:>15.4f} | "
            f"{s['budget_lifetime']:>16.4f}"
        )

    print("=" * 90)

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 12
    })

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9)
    )

    marker_style = {
        "marker": "o",
        "linewidth": 2,
        "markersize": 7
    }

    # -----------------------------------------------------
    # (a) Average throughput
    # -----------------------------------------------------
    axes[0, 0].plot(
        lambdas,
        avg_throughput,
        **marker_style
    )

    axes[0, 0].set_xlabel(r"$\lambda_E$")
    axes[0, 0].set_ylabel("Average Throughput [Gbps]")
    axes[0, 0].set_title("(a) Average Throughput")
    axes[0, 0].grid(True, alpha=0.3)

    # -----------------------------------------------------
    # (b) Average JFI
    # -----------------------------------------------------
    axes[0, 1].plot(
        lambdas,
        avg_jfi,
        **marker_style
    )

    axes[0, 1].set_xlabel(r"$\lambda_E$")
    axes[0, 1].set_ylabel("Jain's Fairness Index")
    axes[0, 1].set_title("(b) Average JFI")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    # -----------------------------------------------------
    # (c) Tracking RMSE
    # -----------------------------------------------------
    axes[1, 0].plot(
        lambdas,
        tracking_rmse,
        **marker_style
    )

    axes[1, 0].set_xlabel(r"$\lambda_E$")
    axes[1, 0].set_ylabel("Tracking RMSE")
    axes[1, 0].set_title(
        rf"(c) Energy-Budget Tracking RMSE ($W={tracking_window}$)"
    )
    axes[1, 0].grid(True, alpha=0.3)

    # -----------------------------------------------------
    # (d) Budget lifetime
    # -----------------------------------------------------
    axes[1, 1].plot(
        lambdas,
        budget_lifetime,
        **marker_style
    )

    axes[1, 1].axhline(
        1.0,
        linestyle="--",
        linewidth=1.8,
        label="Full Horizon"
    )

    axes[1, 1].set_xlabel(r"$\lambda_E$")
    axes[1, 1].set_ylabel("Normalized Budget Lifetime")
    axes[1, 1].set_title("(d) Budget Lifetime")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()

    suffix = (
        f"_{eval_variant}"
        if eval_variant in ["soft", "hard"]
        else ""
    )

    save_path = os.path.join(
        plot_dir,
        f"summary{suffix}_lambda_sensitivity_four_metrics.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved lambda sensitivity plot: {save_path}")

def plot_eval_timeseries_all_lambdas(eval_files, plot_dir, smooth_window=1000, eval_variant="soft"):
    labels = [rf"$\lambda_E={parse_lambda(f):g}$" for f in eval_files]
    suffix = f"_{eval_variant}" if eval_variant in ["soft", "hard"] else ""

    # -----------------------------------------------------
    # 1) Throughput
    # -----------------------------------------------------
    x_list, y_list = [], []
    for f in eval_files:
        data = load_npz(f)
        throughput = safe_get(data, "throughput", np.array([]))

        y = moving_average(throughput, smooth_window)
        x = np.arange(len(y)) + smooth_window - 1 if len(throughput) >= smooth_window else np.arange(len(y))

        x_list.append(x)
        y_list.append(y)

    save_lineplot_multi(
        x_list, y_list, labels,
        xlabel="Evaluation Step",
        ylabel="Throughput [Gbps]",
        title="Evaluation Throughput",
        save_path=os.path.join(plot_dir, f"eval{suffix}_throughput_all_lambdas.png")
    )

    # -----------------------------------------------------
    # 2) JFI
    # -----------------------------------------------------
    x_list, y_list = [], []
    for f in eval_files:
        data = load_npz(f)
        fairness = safe_get(data, "fairness", np.array([]))

        y = moving_average(fairness, smooth_window)
        x = np.arange(len(y)) + smooth_window - 1 if len(fairness) >= smooth_window else np.arange(len(y))

        x_list.append(x)
        y_list.append(y)

    save_lineplot_multi(
        x_list, y_list, labels,
        xlabel="Evaluation Step",
        ylabel="Jain's Fairness Index",
        title="Evaluation JFI",
        save_path=os.path.join(plot_dir, f"eval{suffix}_jfi_all_lambdas.png")
    )

    # -----------------------------------------------------
    # 3) Handover ratio
    # -----------------------------------------------------
    x_list, y_list = [], []
    h_budget = None

    for f in eval_files:
        data = load_npz(f)
        handover_ratio = safe_get(data, "handover_ratio", np.array([]))

        if h_budget is None:
            hb = safe_get(data, "handover_budget_ratio", np.array([np.nan]))
            h_budget = float(hb[0]) if hb is not None and len(hb) > 0 else None

        y = moving_average(handover_ratio, smooth_window)
        x = np.arange(len(y)) + smooth_window - 1 if len(handover_ratio) >= smooth_window else np.arange(len(y))

        x_list.append(x)
        y_list.append(y)

    save_lineplot_multi(
        x_list, y_list, labels,
        xlabel="Evaluation Step",
        ylabel="Handover Ratio",
        title="Evaluation Handover Ratio",
        save_path=os.path.join(plot_dir, f"eval{suffix}_handover_ratio_all_lambdas.png"),
        hline=h_budget,
        hline_label="Handover Budget"
    )

    # -----------------------------------------------------
    # 4) BS ON ratio
    # -----------------------------------------------------
    x_list, y_list = [], []
    e_budget = None

    for f in eval_files:
        data = load_npz(f)
        power_mat = safe_get(data, "power_mat", np.zeros((0, 0), dtype=np.float32))

        if e_budget is None:
            eb = safe_get(data, "energy_budget_ratio", np.array([np.nan]))
            e_budget = float(eb[0]) if eb is not None and len(eb) > 0 else None

        if power_mat.size == 0:
            continue

        bs_on_mat = (power_mat > 0.0).astype(np.float32)
        on_ratio_step = np.mean(bs_on_mat, axis=0)

        y = moving_average(on_ratio_step, smooth_window)
        x = np.arange(len(y)) + smooth_window - 1 if len(on_ratio_step) >= smooth_window else np.arange(len(y))

        x_list.append(x)
        y_list.append(y)

    used_labels = labels[:len(x_list)] if len(x_list) != len(labels) else labels

    save_lineplot_multi(
        x_list, y_list, used_labels,
        xlabel="Evaluation Step",
        ylabel="BS ON Ratio",
        title="Evaluation BS ON Ratio",
        save_path=os.path.join(plot_dir, f"eval{suffix}_on_ratio_all_lambdas.png"),
        hline=e_budget,
        hline_label="Energy Budget"
    )


# =========================================================
# Eval summary plots over lambda
# =========================================================
def summarize_eval_npz(path, last_window=10000):
    data = load_npz(path)

    throughput = safe_get(data, "throughput", np.array([]))
    fairness = safe_get(data, "fairness", np.array([]))
    handover_ratio = safe_get(data, "handover_ratio", np.array([]))
    power_mat = safe_get(data, "power_mat", np.zeros((0, 0), dtype=np.float32))

    thr_mean = float(np.mean(throughput[-last_window:])) if throughput.size > 0 else np.nan
    fair_mean = float(np.mean(fairness[-last_window:])) if fairness.size > 0 else np.nan
    ho_mean = float(np.mean(handover_ratio[-last_window:])) if handover_ratio.size > 0 else np.nan

    if power_mat.size > 0:
        recent_power = power_mat[:, -last_window:]
        on_ratio_mean = float(np.mean(recent_power > 0.0))
    else:
        on_ratio_mean = np.nan

    e_budget_arr = safe_get(data, "energy_budget_ratio", np.array([np.nan]))
    h_budget_arr = safe_get(data, "handover_budget_ratio", np.array([np.nan]))

    e_budget = float(e_budget_arr[0]) if len(e_budget_arr) > 0 else np.nan
    h_budget = float(h_budget_arr[0]) if len(h_budget_arr) > 0 else np.nan

    return {
        "lambda": parse_lambda(path),
        "throughput": thr_mean,
        "fairness": fair_mean,
        "on_ratio": on_ratio_mean,
        "handover_ratio": ho_mean,
        "energy_budget": e_budget,
        "handover_budget": h_budget,
        "energy_violation": max(0.0, on_ratio_mean - e_budget) if not np.isnan(on_ratio_mean) else np.nan,
        "handover_violation": max(0.0, ho_mean - h_budget) if not np.isnan(ho_mean) else np.nan,
    }


def plot_eval_summary_vs_lambda(eval_files, plot_dir, last_window=10000, eval_variant="soft"):
    summaries = [summarize_eval_npz(f, last_window) for f in eval_files]
    summaries = sorted(summaries, key=lambda x: x["lambda"])
    suffix = f"_{eval_variant}" if eval_variant in ["soft", "hard"] else ""

    lams = np.array([s["lambda"] for s in summaries], dtype=np.float32)
    thr = np.array([s["throughput"] for s in summaries], dtype=np.float32)
    jfi = np.array([s["fairness"] for s in summaries], dtype=np.float32)
    on = np.array([s["on_ratio"] for s in summaries], dtype=np.float32)
    ho = np.array([s["handover_ratio"] for s in summaries], dtype=np.float32)
    e_budget = summaries[0]["energy_budget"] if len(summaries) > 0 else None
    h_budget = summaries[0]["handover_budget"] if len(summaries) > 0 else None
    e_violation = np.array([s["energy_violation"] for s in summaries], dtype=np.float32)
    h_violation = np.array([s["handover_violation"] for s in summaries], dtype=np.float32)

    # 1) Throughput vs lambda
    plt.figure(figsize=(7, 5))
    plt.plot(lams, thr, marker="o", linewidth=2)
    plt.xlabel(r"$\lambda_E$")
    plt.ylabel("Throughput [Gbps]")
    plt.title("Throughput vs " + r"$\lambda_E$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"summary{suffix}_throughput_vs_lambda.png"), dpi=300)
    plt.close()

    # 2) JFI vs lambda
    plt.figure(figsize=(7, 5))
    plt.plot(lams, jfi, marker="o", linewidth=2)
    plt.xlabel(r"$\lambda_E$")
    plt.ylabel("Jain's Fairness Index")
    plt.title("JFI vs " + r"$\lambda_E$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"summary{suffix}_jfi_vs_lambda.png"), dpi=300)
    plt.close()

    # 3) ON ratio vs lambda
    plt.figure(figsize=(7, 5))
    plt.plot(lams, on, marker="o", linewidth=2, label="Actual")
    if e_budget is not None and not np.isnan(e_budget):
        plt.axhline(e_budget, linestyle="--", linewidth=2, label="Energy Budget")
    plt.xlabel(r"$\lambda_E$")
    plt.ylabel("BS ON Ratio")
    plt.title("BS ON Ratio vs " + r"$\lambda_E$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"summary{suffix}_on_ratio_vs_lambda.png"), dpi=300)
    plt.close()

    # 4) Handover ratio vs lambda
    plt.figure(figsize=(7, 5))
    plt.plot(lams, ho, marker="o", linewidth=2, label="Actual")
    if h_budget is not None and not np.isnan(h_budget):
        plt.axhline(h_budget, linestyle="--", linewidth=2, label="Handover Budget")
    plt.xlabel(r"$\lambda_E$")
    plt.ylabel("Handover Ratio")
    plt.title("Handover Ratio vs " + r"$\lambda_E$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"summary{suffix}_handover_ratio_vs_lambda.png"), dpi=300)
    plt.close()

    # 5) Constraint violation bar
    x = np.arange(len(lams))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, e_violation, width, label="Energy Violation")
    plt.bar(x + width / 2, h_violation, width, label="Handover Violation")
    plt.xticks(x, [str(v) for v in lams])
    plt.xlabel(r"$\lambda_E$")
    plt.ylabel("Violation Amount")
    plt.title("Constraint Violations vs " + r"$\lambda_E$")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"summary{suffix}_constraint_violations_vs_lambda.png"), dpi=300)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results/results_lambda")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "eval", "summary"])
    parser.add_argument("--last_window", type=int, default=10000)
    parser.add_argument("--smooth_window", type=int, default=1000)
    parser.add_argument(
        "--eval_variant",
        type=str,
        default="hard",
        choices=["soft", "hard", "all"],
        help="Which eval files to plot. Use soft for natural policy behavior, hard for hard-constrained evaluation."
    )
    args = parser.parse_args()

    result_dir = args.result_dir
    plot_dir = os.path.join(result_dir, "plots")
    ensure_dir(plot_dir)

    train_files = sorted_npz_files(result_dir, "train")
    eval_files = sorted_npz_files(result_dir, "eval", eval_variant=args.eval_variant)

    print(f"Result dir: {result_dir}")
    print(f"Plot dir:   {plot_dir}")
    print(f"Train files: {len(train_files)}")
    for f in train_files:
        print("  ", f)
    print(f"Eval files: {len(eval_files)}")
    print(f"Eval variant: {args.eval_variant}")
    for f in eval_files:
        print("  ", f)

    if args.mode in ["all", "train"]:
        if len(train_files) > 0:
            plot_train_curves_all_lambdas(train_files, plot_dir)
        else:
            print("[Warning] No train npz files found.")

    if args.mode in ["all", "eval"]:
        if len(eval_files) > 0:
            plot_eval_timeseries_all_lambdas(
                eval_files,
                plot_dir,
                smooth_window=args.smooth_window,
                eval_variant=args.eval_variant
            )
            plot_queue_timeseries_all_lambdas(
                eval_files,
                plot_dir,
                prefix="eval",
                smooth_window=args.smooth_window,
                eval_variant=args.eval_variant
            )
        else:
            print("[Warning] No eval npz files found.")
    
    if args.mode in ["all", "summary"]:
        if len(eval_files) > 0:
            plot_eval_summary_vs_lambda(
                eval_files,
                plot_dir,
                last_window=args.last_window,
                eval_variant=args.eval_variant
            )
            plot_lambda_sensitivity_four_metrics(
                eval_files=eval_files,
                plot_dir=plot_dir,
                tracking_window=500,
                metric_window=None,
                eval_variant=args.eval_variant
            )
        else:
            print("[Warning] No eval npz files found.")

    print("\n✅ Plotting completed.")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()