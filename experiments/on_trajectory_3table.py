import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 설정
# ============================================================
KAPPA = 0.015
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

MAX_STEPS = 10000
SMOOTH_WINDOW = 100
TARGET_ON_RATIO = 0.6

SAVE_DIR = "eval_compare_plots"
os.makedirs(SAVE_DIR, exist_ok=True)


LYMARL_ROOT = "/home/wjddustj/LyMARL"
EXTENDED_ROOT = "/home/wjddustj/LyMARL_extended"


def get_npz_path(algorithm, train_seed, eval_seed):
    if algorithm in ["MAPPO", "HAPPO"]:
        return os.path.join(
            EXTENDED_ROOT,
            "results",
            "results_kappa",
            f"{algorithm}_kappa_0.015_seed_{train_seed}",
            f"eval_seed_{eval_seed}.npz",
        )

    if algorithm == "LyMARL":
        return os.path.join(
            LYMARL_ROOT,
            "results",
            "results_mappo_happo",
            "LyMARL_HO_kappa_0.015",
            f"seed_{train_seed}",
            f"eval_seed_{eval_seed}.npz",
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


ALGORITHMS = ["MAPPO", "HAPPO", "LyMARL"]


# ============================================================
# 2. Plot 스타일
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 17,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.2,
})


# ============================================================
# 3. Moving average
# ============================================================
def moving_average(x, window):
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if x.size == 0:
        return x

    window = max(1, int(window))
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    result = np.empty_like(x)

    for t in range(x.size):
        start = max(0, t - window + 1)
        count = t - start + 1
        result[t] = (
            cumsum[t + 1] - cumsum[start]
        ) / count

    return result


# ============================================================
# 4. NPZ에서 BS ON matrix 읽기
# ============================================================
def load_bs_on_matrix(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        if "power_mat" not in data.files:
            raise KeyError(
                f"'power_mat' not found in {npz_path}\n"
                f"Available keys: {data.files}"
            )

        power_mat = np.asarray(
            data["power_mat"],
            dtype=np.float64,
        )

    if power_mat.ndim != 2:
        raise ValueError(
            f"power_mat must be 2-D: {power_mat.shape}"
        )

    # [T, B]이면 [B, T]로 전환
    if (
        power_mat.shape[0] > power_mat.shape[1]
        and power_mat.shape[1] <= 20
    ):
        power_mat = power_mat.T

    return (power_mat > 0.0).astype(np.float64)


# ============================================================
# 5. 알고리즘별 15개 trajectory 집계
# ============================================================
summary = {}

for algorithm in ALGORITHMS:
    trajectories = []

    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            path = get_npz_path(
                algorithm,
                train_seed,
                eval_seed,
            )

            try:
                bs_on_mat = load_bs_on_matrix(path)

                T = min(
                    MAX_STEPS,
                    bs_on_mat.shape[1],
                )

                mean_on_per_slot = np.mean(
                    bs_on_mat[:, :T],
                    axis=0,
                )

                smoothed = moving_average(
                    mean_on_per_slot,
                    SMOOTH_WINDOW,
                )

                trajectories.append(smoothed)

            except Exception as error:
                print(
                    f"[WARNING] {algorithm}, "
                    f"train={train_seed}, eval={eval_seed}: "
                    f"{error}"
                )

    if not trajectories:
        continue

    common_T = min(len(x) for x in trajectories)

    trajectory_mat = np.stack(
        [x[:common_T] for x in trajectories],
        axis=0,
    )

    summary[algorithm] = {
        "mean": np.mean(trajectory_mat, axis=0),
        "std": np.std(trajectory_mat, axis=0),
        "n": trajectory_mat.shape[0],
    }

    print(
        f"[{algorithm}] "
        f"N={trajectory_mat.shape[0]} | "
        f"final overall ON="
        f"{np.mean(trajectory_mat[:, -1000:]):.4f}"
    )


# ============================================================
# 6. Plot
# ============================================================
line_styles = {
    "MAPPO": {
        "linestyle": "--",
        "marker": "s",
    },
    "HAPPO": {
        "linestyle": "-",
        "marker": "o",
    },
    "LyMARL": {
        "linestyle": "-.",
        "marker": "^",
    },
}


fig, ax = plt.subplots(figsize=(9.2, 5.4))

for algorithm, values in summary.items():
    mean = values["mean"]
    std = values["std"]
    x = np.arange(1, len(mean) + 1)

    style = line_styles[algorithm]
    marker_interval = max(1, len(mean) // 10)

    line = ax.plot(
        x,
        mean,
        label=algorithm,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markevery=marker_interval,
        markersize=7,
        linewidth=2.3,
    )[0]

    # 선과 같은 색으로 표준편차 음영
    ax.fill_between(
        x,
        np.clip(mean - std, 0.0, 1.0),
        np.clip(mean + std, 0.0, 1.0),
        color=line.get_color(),
        alpha=0.15,
        linewidth=0,
    )


ax.axhline(
    TARGET_ON_RATIO,
    linestyle="--",
    linewidth=1.8,
    color="black",
    alpha=0.8,
    label=r"Target $\eta=0.6$",
)

ax.set_xlabel("Time step")
ax.set_ylabel("Mean BS ON-ratio")

ax.set_xlim(0, MAX_STEPS)
ax.set_ylim(0.0, 1.05)

ax.set_xticks([
    0,
    2000,
    4000,
    6000,
    8000,
    10000,
])
ax.set_xticklabels([
    "0",
    "2K",
    "4K",
    "6K",
    "8K",
    "10K",
])

ax.set_yticks(np.arange(0.0, 1.01, 0.2))

ax.grid(
    True,
    linestyle="-",
    linewidth=0.7,
    alpha=0.3,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="lower left",
    ncol=2,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
)

plt.tight_layout()

png_path = os.path.join(
    SAVE_DIR,
    "on_ratio_trajectory_kappa_0.015.png",
)


plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved PNG: {png_path}")