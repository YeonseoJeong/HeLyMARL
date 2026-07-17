import os
import matplotlib
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 22,
    # "font.weight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    # "axes.labelweight": "bold",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def moving_average(x, window=1000):
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return x

    if window <= 1:
        return x

    if len(x) < window:
        window = len(x)

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")

# def moving_average(x, window=1000):
#     """
#     처음 window-1 step은 현재까지의 누적 평균,
#     이후에는 trailing moving average를 계산한다.
#     출력 길이는 입력 길이와 동일하다.
#     """
#     x = np.asarray(x, dtype=np.float32).reshape(-1)

#     if x.size == 0:
#         return x

#     if window <= 1:
#         return x.copy()

#     window = min(window, len(x))

#     cumsum = np.cumsum(
#         np.insert(x.astype(np.float64), 0, 0.0)
#     )

#     result = np.empty(len(x), dtype=np.float32)

#     for t in range(len(x)):
#         start = max(0, t - window + 1)
#         count = t - start + 1

#         result[t] = (
#             cumsum[t + 1] - cumsum[start]
#         ) / count

#     return result

def episodic_moving_average(
    x,
    episode_length=10000,
    window=1000,
):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    x_parts = []
    y_parts = []

    for episode_start in range(0, len(x), episode_length):
        episode_end = min(
            episode_start + episode_length,
            len(x),
        )

        episode_data = x[episode_start:episode_end]

        if len(episode_data) == 0:
            continue

        episode_ma = moving_average(
            episode_data,
            window=window,
        )

        episode_x = np.arange(
            episode_start,
            episode_end,
        )

        x_parts.append(episode_x)
        y_parts.append(episode_ma)

        # 에피소드 사이의 선을 끊기 위한 NaN
        x_parts.append(np.array([np.nan]))
        y_parts.append(np.array([np.nan]))

    if len(x_parts) == 0:
        return np.array([]), np.array([])

    return (
        np.concatenate(x_parts),
        np.concatenate(y_parts),
    )

def plot_train_curve(
    result_dir="results/results_kappa",
    save_dir="results/results_kappa/plots",
    kappas=(0.01, 0.02, 0.03),
    episode_length=10000,
    window=1000,
):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    colors = ["C0", "C2", "C3"]
    found_train = False

    for i, kappa in enumerate(kappas):
        train_path = os.path.join(
            result_dir,
            f"HeLyMARL_train_rewards_kappa_{kappa}.npz"
        )

        if not os.path.exists(train_path):
            print(f"[Warning] Train file not found: {train_path}")
            continue

        data = np.load(train_path, allow_pickle=True)

        if "handover_ratio" not in data:
            print(f"[Warning] 'handover_ratio' not found in {train_path}")
            continue

        ho_ratio = np.asanyarray(data["handover_ratio"], dtype=np.float32).reshape(-1)
        ho_ma = moving_average(ho_ratio, window=window)

        x_step = np.arange(window -1, window - 1 + len(ho_ma))
        x_episode = (x_step+1) / episode_length

        if "handover_budget_ratio" in data:
            budget = float(data["handover_budget_ratio"][0])
        else:
            budget = float(kappa)

        # x = np.arange(window - 1, window - 1 + len(ho_ma))
        color = colors[i % len(colors)]

        ax.plot(
            x_episode,
            ho_ma,
            linewidth=2.0,
            color=color,
            label=f"Train $\\kappa$ = {budget:.2f}"
        )

        ax.axhline(
            y=budget,
            linestyle="--",
            linewidth=1.3,
            color=color,
            alpha=0.9
        )

        found_train = True

    n_episodes = len(ho_ratio) / episode_length

    ax.set_xlabel("Training Episode")
    ax.set_ylabel(f"Handover Ratio (MA{window})")
    if found_train:
        ax.set_xlim(0, n_episodes)
        ax.set_xticks(np.arange(0, int(np.floor(n_episodes)) + 1, 1))
        ax.legend()
    
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        "train_handover_curve_kappa.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print(f"✅ Saved: {save_path}")


def plot_eval_bar(
    result_dir="results/results_kappa",
    save_dir="results/results_kappa/plots",
    kappas=(0.01, 0.02, 0.03),
):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    colors = ["C0", "C2", "C3"]

    eval_means = []
    eval_labels = []

    found_eval = False

    for i, kappa in enumerate(kappas):
        eval_path = os.path.join(
            result_dir,
            f"HeLyMARL_eval_hard_kappa_{kappa}.npz"
        )

        if not os.path.exists(eval_path):
            print(f"[Warning] Eval file not found: {eval_path}")
            continue

        data = np.load(eval_path, allow_pickle=True)

        if "handover_ratio" not in data:
            print(f"[Warning] 'handover_ratio' not found in {eval_path}")
            continue

        ho_ratio = data["handover_ratio"].astype(np.float32)

        if "handover_budget_ratio" in data:
            budget = float(data["handover_budget_ratio"][0])
        else:
            budget = float(kappa)

        mean_val = float(np.mean(ho_ratio))

        eval_means.append(mean_val)
        eval_labels.append(f"$\\kappa$ = {budget:.2f}")

        found_eval = True

    if found_eval:
        x_bar = np.arange(len(eval_means))
        bar_colors = [colors[i % len(colors)] for i in range(len(eval_means))]

        ax.bar(
            x_bar,
            eval_means,
            color=bar_colors,
            alpha=0.85,
            width=0.6
        )

        ax.set_xticks(x_bar)
        ax.set_xticklabels(eval_labels)
        ax.set_ylabel("Average Handover Ratio")
        ax.set_ylim(0, 0.03)
        ax.grid(True, axis="y", alpha=0.3)

        for i, val in enumerate(eval_means):
            ax.text(
                i,
                val + 0.001,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=17,
                fontweight="bold"
            )

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        "eval_handover_bar_kappa.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    kappas = (0.01, 0.02, 0.03)

    plot_train_curve(
        result_dir="results/results_kappa",
        save_dir="results/results_kappa/plots",
        kappas=kappas,
        episode_length=10000,
        window=2000,
    )

    plot_eval_bar(
        result_dir="results/results_kappa",
        save_dir="results/results_kappa/plots",
        kappas=kappas,
    )