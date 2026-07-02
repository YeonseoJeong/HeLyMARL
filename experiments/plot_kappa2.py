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
    "font.weight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "axes.labelweight": "bold",
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


def plot_train_curve(
    result_dir="results_kappa",
    save_dir="results_kappa/plots",
    kappas=(0.01, 0.02, 0.03),
    window=1000,
):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    colors = ["C0", "C2", "C3"]
    found_train = False

    for i, kappa in enumerate(kappas):
        train_path = os.path.join(
            result_dir,
            f"LyMARL_train_rewards_kappa_{kappa}.npz"
        )

        if not os.path.exists(train_path):
            print(f"[Warning] Train file not found: {train_path}")
            continue

        data = np.load(train_path, allow_pickle=True)

        if "handover_ratio" not in data:
            print(f"[Warning] 'handover_ratio' not found in {train_path}")
            continue

        ho_ratio = data["handover_ratio"].astype(np.float32)
        ho_ma = moving_average(ho_ratio, window=window)

        if "handover_budget_ratio" in data:
            budget = float(data["handover_budget_ratio"][0])
        else:
            budget = float(kappa)

        x = np.arange(window - 1, window - 1 + len(ho_ma))
        color = colors[i % len(colors)]

        ax.plot(
            x,
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

    ax.set_xlabel("Training Step")
    ax.set_ylabel(f"Handover Ratio (MA{window})")
    ax.grid(True, alpha=0.3)

    if found_train:
        ax.legend()

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        "train_handover_curve_kappa.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print(f"✅ Saved: {save_path}")


def plot_eval_bar(
    result_dir="results_kappa",
    save_dir="results_kappa/plots",
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
            f"LyMARL_eval_hard_kappa_{kappa}.npz"
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
        result_dir="results_kappa",
        save_dir="results_kappa/plots",
        kappas=kappas,
        window=1000,
    )

    plot_eval_bar(
        result_dir="results_kappa",
        save_dir="results_kappa/plots",
        kappas=kappas,
    )