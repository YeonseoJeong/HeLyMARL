import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Experiment settings and result paths
# ============================================================
KAPPA = 0.015
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

# DDPP and MaxSNR use the existing multi-seed summary files.
SUMMARY_FILES = {
    "DDPP": (
        "results/results_baselines/"
        "ddpp_5seeds_evaluation_summary.npz"
    ),
    "MaxSNR": (
        "results/results_baselines/"
        "maxsnr_5seeds_evaluation_summary.npz"
    ),
}

# PF-HAPPO, Jensen-HAPPO, and HeLyMARL are read directly
# from train-seed/eval-seed raw evaluation files.
#
# Expected paths:
#   PF-HAPPO:
#     results/results_baselines/pf/
#       kappa_0.015_seed_{train_seed}/eval_seed_{eval_seed}.npz
#
#   Jensen-HAPPO:
#     results/results_baselines/jensen/
#       kappa_0.015_seed_{train_seed}/eval_seed_{eval_seed}.npz
#
#   HeLyMARL:
#     results/results_kappa/
#       HAPPO_kappa_0.015_seed_{train_seed}/eval_seed_{eval_seed}.npz
RAW_CONFIGS = {
    "PF-HAPPO": {
        "root": "results/results_baselines/pf",
        "folder_prefix": None,
    },
    "Jensen-HAPPO": {
        "root": "results/results_baselines/jensen",
        "folder_prefix": None,
    },
    "HeLyMARL": {
        "root": "results/results_kappa",
        "folder_prefix": "HAPPO",
    },
}


# ============================================================
# 2. Save directory
# ============================================================
SAVE_DIR = "eval_compare_plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 3. Display names
# ============================================================
DISPLAY_NAMES = {
    "DDPP": "DDPP",
    "MaxSNR": "MaxSNR",
    "PF-HAPPO": "PF-HAPPO",
    "Jensen-HAPPO": "Jensen-HAPPO",
    "HeLyMARL": "HeLyMARL",
}


# ============================================================
# 4. Plot style
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.4,
    "hatch.linewidth": 1.4,
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})


# ============================================================
# 5. Metric helpers
# ============================================================
def to_scalar_mean(x):
    arr = np.asarray(x, dtype=float)

    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan

    return float(np.nanmean(arr))


def jain_fairness(rates, eps=1e-12):
    rates = np.asarray(rates, dtype=float)

    if rates.size == 0:
        return np.nan

    if rates.ndim == 1:
        user_rates = rates
    elif rates.ndim == 2:
        # Usually [T, U]. Convert either orientation to user averages.
        if rates.shape[0] >= rates.shape[1]:
            user_rates = np.nanmean(rates, axis=0)
        else:
            user_rates = np.nanmean(rates, axis=1)
    else:
        return np.nan

    user_rates = np.nan_to_num(
        user_rates,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    numerator = np.sum(user_rates) ** 2
    denominator = len(user_rates) * np.sum(user_rates ** 2)

    if denominator <= eps:
        return 0.0

    return float(numerator / (denominator + eps))


def block_jain_fairness(slot_rates, block_size=1000, eps=1e-12):
    """
    Divide slot_rates into blocks and calculate JFI in each block.

    - Input: [T, U] or [U, T]
    - Calculate each user's mean rate within a block
    - Exclude all-off blocks
    - Include the final shorter block
    """
    rates = np.asarray(slot_rates, dtype=float)
    rates = np.squeeze(rates)

    if rates.size == 0 or rates.ndim != 2:
        return np.nan

    # Convert [U, T] to [T, U].
    if rates.shape[0] < rates.shape[1]:
        rates = rates.T

    rates = np.nan_to_num(
        rates,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    rates = np.maximum(rates, 0.0)

    block_jfis = []

    for start in range(0, rates.shape[0], block_size):
        block = rates[start:start + block_size]
        block_user_rates = np.mean(block, axis=0)

        # Exclude a block in which all users have zero rate.
        if np.sum(block_user_rates ** 2) <= eps:
            continue

        block_jfi = jain_fairness(
            block_user_rates,
            eps=eps,
        )

        if np.isfinite(block_jfi):
            block_jfis.append(block_jfi)

    if not block_jfis:
        return np.nan

    return float(np.mean(block_jfis))


def get_fairness(data, block_size=1000):
    """
    Priority:
      1) Recalculate block JFI from slot_rates
      2) fairness_block_jfis
      3) fairness
      4) episode_fairness_last
      5) avg_user_rates
    """
    if "slot_rates" in data.files:
        value = block_jain_fairness(
            data["slot_rates"],
            block_size=block_size,
        )
        if np.isfinite(value):
            return value

    if "fairness_block_jfis" in data.files:
        block_jfis = np.asarray(
            data["fairness_block_jfis"],
            dtype=float,
        ).reshape(-1)

        valid_mask = (
            np.isfinite(block_jfis)
            & (block_jfis > 0.0)
        )

        if np.any(valid_mask):
            return float(
                np.mean(block_jfis[valid_mask])
            )

    if "fairness" in data.files:
        value = to_scalar_mean(data["fairness"])
        if np.isfinite(value):
            return value

    if "episode_fairness_last" in data.files:
        value = to_scalar_mean(
            data["episode_fairness_last"]
        )
        if np.isfinite(value):
            return value

    if "avg_user_rates" in data.files:
        return jain_fairness(
            data["avg_user_rates"]
        )

    return np.nan


def get_throughput(data):
    candidate_keys = [
        "throughput",
        "throughput_history",
        "episode_throughput_mean",
    ]

    for key in candidate_keys:
        if key in data.files:
            value = to_scalar_mean(data[key])

            if np.isfinite(value):
                return value

    return np.nan


# ============================================================
# 6. Path helpers
# ============================================================
def get_raw_npz_path(
    algorithm,
    train_seed,
    eval_seed,
):
    config = RAW_CONFIGS[algorithm]
    kappa_tag = f"{KAPPA:.3f}"

    if config["folder_prefix"] is None:
        folder_name = (
            f"kappa_{kappa_tag}_seed_{train_seed}"
        )
    else:
        folder_name = (
            f"{config['folder_prefix']}_"
            f"kappa_{kappa_tag}_seed_{train_seed}"
        )

    return os.path.join(
        config["root"],
        folder_name,
        f"eval_seed_{eval_seed}.npz",
    )


# ============================================================
# 7. Load DDPP/MaxSNR summary results
# ============================================================
def load_summary_result(summary_path, algorithm):
    if not os.path.exists(summary_path):
        print(
            f"[WARNING] Summary file not found: "
            f"{summary_path}"
        )
        return None

    with np.load(
        summary_path,
        allow_pickle=True,
    ) as summary_data:
        required_keys = [
            "fairness_mean",
            "fairness_std",
            "throughput_mean",
            "throughput_std",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in summary_data.files
        ]

        if missing_keys:
            print(
                f"[WARNING] {algorithm}: "
                f"missing keys={missing_keys}"
            )
            print(
                f"Available keys: {summary_data.files}"
            )
            return None

        return (
            float(
                np.asarray(
                    summary_data["fairness_mean"]
                ).reshape(-1)[0]
            ),
            float(
                np.asarray(
                    summary_data["fairness_std"]
                ).reshape(-1)[0]
            ),
            float(
                np.asarray(
                    summary_data["throughput_mean"]
                ).reshape(-1)[0]
            ),
            float(
                np.asarray(
                    summary_data["throughput_std"]
                ).reshape(-1)[0]
            ),
        )


# ============================================================
# 8. Load PF/Jensen/HeLyMARL raw results
# ============================================================
def load_raw_result(algorithm):
    """
    Aggregation:
      1) Average the five eval seeds within each train seed.
      2) Calculate mean/std across the three train seeds.
    """
    train_fairness = []
    train_throughput = []

    for train_seed in TRAIN_SEEDS:
        eval_fairness = []
        eval_throughput = []

        for eval_seed in EVAL_SEEDS:
            npz_path = get_raw_npz_path(
                algorithm=algorithm,
                train_seed=train_seed,
                eval_seed=eval_seed,
            )

            if not os.path.exists(npz_path):
                print(
                    f"[WARNING] {algorithm} file not found: "
                    f"{npz_path}"
                )
                continue

            with np.load(
                npz_path,
                allow_pickle=True,
            ) as data:
                fairness = get_fairness(
                    data,
                    block_size=1000,
                )
                throughput = get_throughput(data)

            if np.isfinite(fairness):
                eval_fairness.append(fairness)
            else:
                print(
                    f"[WARNING] Invalid {algorithm} "
                    f"fairness: {npz_path}"
                )

            if np.isfinite(throughput):
                eval_throughput.append(throughput)
            else:
                print(
                    f"[WARNING] Invalid {algorithm} "
                    f"throughput: {npz_path}"
                )

        if eval_fairness:
            train_fairness.append(
                float(np.mean(eval_fairness))
            )

        if eval_throughput:
            train_throughput.append(
                float(np.mean(eval_throughput))
            )

        print(
            f"[{algorithm} train seed {train_seed}] "
            f"valid fairness evals={len(eval_fairness)}, "
            f"valid throughput evals={len(eval_throughput)}"
        )

    if not train_fairness or not train_throughput:
        print(
            f"[WARNING] No valid {algorithm} "
            f"kappa={KAPPA:.3f} results found."
        )
        return None

    return (
        float(np.mean(train_fairness)),
        float(np.std(train_fairness)),
        float(np.mean(train_throughput)),
        float(np.std(train_throughput)),
    )


# ============================================================
# 9. Read all results
# ============================================================
algorithms = []

fairness_values = []
fairness_stds = []

throughput_values = []
throughput_stds = []


for algorithm in DISPLAY_NAMES:
    if algorithm in SUMMARY_FILES:
        result = load_summary_result(
            SUMMARY_FILES[algorithm],
            algorithm,
        )
    else:
        result = load_raw_result(algorithm)

    if result is None:
        continue

    (
        fairness_mean,
        fairness_std,
        throughput_mean,
        throughput_std,
    ) = result

    algorithms.append(
        DISPLAY_NAMES[algorithm]
    )
    fairness_values.append(
        fairness_mean
    )
    fairness_stds.append(
        fairness_std
    )
    throughput_values.append(
        throughput_mean
    )
    throughput_stds.append(
        throughput_std
    )

    print(
        f"{algorithm:15s} | "
        f"Fairness={fairness_mean:.4f} "
        f"± {fairness_std:.4f} | "
        f"Throughput={throughput_mean:.4f} "
        f"± {throughput_std:.4f}"
    )


algorithms = np.asarray(algorithms)

fairness_values = np.asarray(
    fairness_values,
    dtype=float,
)
fairness_stds = np.asarray(
    fairness_stds,
    dtype=float,
)
throughput_values = np.asarray(
    throughput_values,
    dtype=float,
)
throughput_stds = np.asarray(
    throughput_stds,
    dtype=float,
)

# Replace NaN std only for plotting error bars.
fairness_stds_plot = np.nan_to_num(
    fairness_stds,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

throughput_stds_plot = np.nan_to_num(
    throughput_stds,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

x = np.arange(len(algorithms))


# ============================================================
# 10. Bar styles
# ============================================================
bar_colors = [
    "#8CB7D9",  # DDPP
    "#F4B978",  # MaxSNR
    "#8FCB8F",  # PF-HAPPO
    "#E88989",  # Jensen-HAPPO
    "#9467bd",  # HeLyMARL
]

bar_hatches = [
    "--",
    "//",
    "\\\\",
    "xx",
    "",
]


# ============================================================
# 11. Fairness figure
# ============================================================
fig, ax = plt.subplots(
    figsize=(8.2, 6.0)
)

bars = ax.bar(
    x,
    fairness_values,
    width=0.62,
    color=bar_colors[:len(x)],
    edgecolor="black",
    linewidth=1.2,
    yerr=fairness_stds_plot,
    capsize=5,
    error_kw={
        "ecolor": "black",
        "elinewidth": 1.4,
        "capthick": 1.4,
    },
)

for bar, hatch in zip(
    bars,
    bar_hatches,
):
    bar.set_hatch(hatch)

for bar, value, std in zip(
    bars,
    fairness_values,
    fairness_stds_plot,
):
    if np.isfinite(value):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + std + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )

ax.set_ylabel(
    "Jain's Fairness Index (JFI)"
)
ax.set_xticks(x)
ax.set_xticklabels(
    algorithms,
    rotation=30,
    ha="right",
)

finite_fairness_upper = (
    fairness_values
    + fairness_stds_plot
)
finite_fairness_upper = (
    finite_fairness_upper[
        np.isfinite(finite_fairness_upper)
    ]
)

if finite_fairness_upper.size > 0:
    fairness_ymax = max(
        1.05,
        np.max(finite_fairness_upper) + 0.08,
    )
    ax.set_ylim(0.0, fairness_ymax)
else:
    ax.set_ylim(0.0, 1.10)

ax.set_yticks(
    np.arange(0.0, 1.01, 0.2)
)
ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.7,
    alpha=0.25,
)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

fairness_path = os.path.join(
    SAVE_DIR,
    "fairness_comparison_kappa0015.png",
)

plt.savefig(
    fairness_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 12. Throughput figure
# ============================================================
fig, ax = plt.subplots(
    figsize=(8.2, 6.0)
)

bars = ax.bar(
    x,
    throughput_values,
    width=0.62,
    color=bar_colors[:len(x)],
    edgecolor="black",
    linewidth=1.2,
    yerr=throughput_stds_plot,
    capsize=5,
    error_kw={
        "ecolor": "black",
        "elinewidth": 1.4,
        "capthick": 1.4,
    },
)

for bar, hatch in zip(
    bars,
    bar_hatches,
):
    bar.set_hatch(hatch)

for bar, value, std in zip(
    bars,
    throughput_values,
    throughput_stds_plot,
):
    if np.isfinite(value):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + std + 0.08,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )

ax.set_ylabel(
    "Average Throughput (Gbps)"
)
ax.set_xticks(x)
ax.set_xticklabels(
    algorithms,
    rotation=30,
    ha="right",
)

finite_throughput_upper = (
    throughput_values
    + throughput_stds_plot
)
finite_throughput_upper = (
    finite_throughput_upper[
        np.isfinite(finite_throughput_upper)
    ]
)

if finite_throughput_upper.size > 0:
    ymax = np.max(
        finite_throughput_upper
    )
    ax.set_ylim(
        0.0,
        ymax * 1.15,
    )
else:
    ax.set_ylim(0.0, 1.0)

ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.7,
    alpha=0.25,
)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

throughput_path = os.path.join(
    SAVE_DIR,
    "throughput_comparison_kappa0015.png",
)

plt.savefig(
    throughput_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close()


print(
    f"\nSaved fairness figure:   "
    f"{fairness_path}"
)
print(
    f"Saved throughput figure: "
    f"{throughput_path}"
)
