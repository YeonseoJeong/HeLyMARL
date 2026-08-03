import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. NPZ 경로 및 실험 설정
# ============================================================

# DDPP / MaxSNR / PF-HAPPO / Jensen-HAPPO는 기존 summary 사용
MULTI_SEED_FILES = {
    "DDPP": (
        "results/results_multi_seed/"
        "ddpp_5seeds_evaluation_summary.npz"
    ),
    "MaxSNR": (
        "results/results_multi_seed/"
        "maxsnr_5seeds_evaluation_summary.npz"
    ),
    "PF-HAPPO": (
        "results/results_multi_seed/"
        "pf_happo_3train_5eval_summary.npz"
    ),
    "Jensen-HAPPO": (
        "results/results_multi_seed/"
        "jensen_happo_3train_5eval_summary.npz"
    ),
}

# HeLyMARL은 아래 HAPPO kappa=0.020 결과를 직접 읽음
HELYMARL_ROOT = "results/results_kappa"
HELYMARL_KAPPA = 0.030
HELYMARL_TRAIN_SEEDS = [0, 1, 2]
HELYMARL_EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

# LyMARL은 아래 HAPPO kappa=0.020 결과를 직접 읽음
LYMARL_ROOT = "results/results_kappa"
LYMARL_KAPPA = 0.030
LYMARL_TRAIN_SEEDS = [0, 1, 2]
LYMARL_EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]


def get_helymarl_npz_path(train_seed, eval_seed):
    return os.path.join(
        HELYMARL_ROOT,
        f"HAPPO_kappa_{HELYMARL_KAPPA:.3f}_seed_{train_seed}",
        f"eval_seed_{eval_seed}.npz",
    )

def get_lymarl_npz_path(train_seed, eval_seed):
    return os.path.join(
        LYMARL_ROOT,
        f"LyMARL_kappa_{LYMARL_KAPPA:.3f}_seed_{train_seed}",
        f"eval_seed_{eval_seed}.npz",
    )



# ============================================================
# 2. 저장 폴더
# ============================================================
SAVE_DIR = "eval_compare_plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 3. Figure에 표시할 알고리즘 이름
# ============================================================
DISPLAY_NAMES = {
    "DDPP": "DDPP",
    "MaxSNR": "MaxSNR",
    "PF-HAPPO": "PF-HAPPO",
    "Jensen-HAPPO": "Jensen-HAPPO",
    "HeLyMARL": "HeLyMARL",
    "LyMARL": "LyMARL",
}


# ============================================================
# 4. Plot 스타일
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
# 5. 공통 함수
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
        # 일반적으로 [T, U]
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
    slot_rates를 block_size 단위로 나누어 JFI를 계산한다.

    - 입력: [T, U] 또는 [U, T]
    - 각 블록에서 UE별 평균 rate를 구한 뒤 JFI 계산
    - 모든 UE의 평균 rate가 0인 all-off 블록은 제외
    - 마지막 블록이 block_size보다 짧아도 포함
    """
    rates = np.asarray(slot_rates, dtype=float)
    rates = np.squeeze(rates)

    if rates.size == 0 or rates.ndim != 2:
        return np.nan

    # [U, T] 형태라면 [T, U]로 변환
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

        # 모든 UE의 rate가 0인 all-off 블록 제외
        if np.sum(block_user_rates ** 2) <= eps:
            continue

        block_jfi = jain_fairness(block_user_rates, eps=eps)

        if np.isfinite(block_jfi):
            block_jfis.append(block_jfi)

    if not block_jfis:
        return np.nan

    return float(np.mean(block_jfis))


def get_fairness(data, block_size=1000):
    """
    우선순위:
    1) slot_rates를 1000-step 블록으로 직접 계산
    2) 저장된 fairness_block_jfis 사용
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

        valid_mask = np.isfinite(block_jfis) & (block_jfis > 0.0)

        if np.any(valid_mask):
            return float(np.mean(block_jfis[valid_mask]))

    if "fairness" in data.files:
        value = to_scalar_mean(data["fairness"])
        if np.isfinite(value):
            return value

    if "episode_fairness_last" in data.files:
        value = to_scalar_mean(data["episode_fairness_last"])
        if np.isfinite(value):
            return value

    if "avg_user_rates" in data.files:
        return jain_fairness(data["avg_user_rates"])

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


def load_summary_result(summary_path, algorithm):
    if not os.path.exists(summary_path):
        print(f"[WARNING] Multi-seed file not found: {summary_path}")
        return None

    with np.load(summary_path, allow_pickle=True) as summary_data:
        required_keys = [
            "fairness_mean",
            "fairness_std",
            "throughput_mean",
            "throughput_std",
        ]

        missing_keys = [
            key for key in required_keys
            if key not in summary_data.files
        ]

        if missing_keys:
            print(f"[WARNING] {algorithm}: missing keys={missing_keys}")
            print(f"Available keys: {summary_data.files}")
            return None

        return (
            float(np.asarray(summary_data["fairness_mean"]).reshape(-1)[0]),
            float(np.asarray(summary_data["fairness_std"]).reshape(-1)[0]),
            float(np.asarray(summary_data["throughput_mean"]).reshape(-1)[0]),
            float(np.asarray(summary_data["throughput_std"]).reshape(-1)[0]),
        )


def load_helymarl_result():
    """
    각 train seed에서 eval seed 결과를 먼저 평균한 뒤,
    train seed 평균과 표준편차를 계산한다.
    """
    train_fairness = []
    train_throughput = []

    for train_seed in HELYMARL_TRAIN_SEEDS:
        eval_fairness = []
        eval_throughput = []

        for eval_seed in HELYMARL_EVAL_SEEDS:
            npz_path = get_helymarl_npz_path(train_seed, eval_seed)

            if not os.path.exists(npz_path):
                print(f"[WARNING] HeLyMARL file not found: {npz_path}")
                continue

            with np.load(npz_path, allow_pickle=True) as data:
                fairness = get_fairness(data, block_size=1000)
                throughput = get_throughput(data)

            if np.isfinite(fairness):
                eval_fairness.append(fairness)
            else:
                print(f"[WARNING] Invalid HeLyMARL fairness: {npz_path}")

            if np.isfinite(throughput):
                eval_throughput.append(throughput)
            else:
                print(f"[WARNING] Invalid HeLyMARL throughput: {npz_path}")

        if eval_fairness:
            train_fairness.append(float(np.mean(eval_fairness)))

        if eval_throughput:
            train_throughput.append(float(np.mean(eval_throughput)))

        print(
            f"[HeLyMARL train seed {train_seed}] "
            f"valid fairness evals={len(eval_fairness)}, "
            f"valid throughput evals={len(eval_throughput)}"
        )

    if not train_fairness or not train_throughput:
        print("[WARNING] No valid HeLyMARL kappa=0.030 results found.")
        return None

    return (
        float(np.mean(train_fairness)),
        float(np.std(train_fairness)),
        float(np.mean(train_throughput)),
        float(np.std(train_throughput)),
    )


def load_lymarl_result():
    """
    각 train seed에서 eval seed 결과를 먼저 평균한 뒤,
    train seed 평균과 표준편차를 계산한다.
    """
    train_fairness = []
    train_throughput = []

    for train_seed in LYMARL_TRAIN_SEEDS:
        eval_fairness = []
        eval_throughput = []

        for eval_seed in LYMARL_EVAL_SEEDS:
            npz_path = get_lymarl_npz_path(train_seed, eval_seed)

            if not os.path.exists(npz_path):
                print(f"[WARNING] LyMARL file not found: {npz_path}")
                continue

            with np.load(npz_path, allow_pickle=True) as data:
                fairness = get_fairness(data, block_size=1000)
                throughput = get_throughput(data)

            if np.isfinite(fairness):
                eval_fairness.append(fairness)
            else:
                print(f"[WARNING] Invalid LyMARL fairness: {npz_path}")

            if np.isfinite(throughput):
                eval_throughput.append(throughput)
            else:
                print(f"[WARNING] Invalid LyMARL throughput: {npz_path}")

        if eval_fairness:
            train_fairness.append(float(np.mean(eval_fairness)))

        if eval_throughput:
            train_throughput.append(float(np.mean(eval_throughput)))

        print(
            f"[LyMARL train seed {train_seed}] "
            f"valid fairness evals={len(eval_fairness)}, "
            f"valid throughput evals={len(eval_throughput)}"
        )

    if not train_fairness or not train_throughput:
        print("[WARNING] No valid LyMARL kappa=0.030 results found.")
        return None

    return (
        float(np.mean(train_fairness)),
        float(np.std(train_fairness)),
        float(np.mean(train_throughput)),
        float(np.std(train_throughput)),
    )

# ============================================================
# 6. 결과 읽기
# ============================================================
algorithms = []

fairness_values = []
fairness_stds = []

throughput_values = []
throughput_stds = []


for algorithm in DISPLAY_NAMES:
    if algorithm == "HeLyMARL":
        result = load_helymarl_result()
    elif algorithm == "LyMARL":
        result = load_lymarl_result()
    else:
        result = load_summary_result(
            MULTI_SEED_FILES[algorithm],
            algorithm,
        )

    if result is None:
        continue

    fairness_mean, fairness_std, throughput_mean, throughput_std = result

    algorithms.append(DISPLAY_NAMES[algorithm])
    fairness_values.append(fairness_mean)
    fairness_stds.append(fairness_std)
    throughput_values.append(throughput_mean)
    throughput_stds.append(throughput_std)

    print(
        f"{algorithm:15s} | "
        f"Fairness={fairness_mean:.4f} ± {fairness_std:.4f} | "
        f"Throughput={throughput_mean:.4f} ± {throughput_std:.4f}"
    )


algorithms = np.asarray(algorithms)
fairness_values = np.asarray(fairness_values, dtype=float)
fairness_stds = np.asarray(fairness_stds, dtype=float)
throughput_values = np.asarray(throughput_values, dtype=float)
throughput_stds = np.asarray(throughput_stds, dtype=float)

# NaN std가 있으면 error bar만 0으로 처리
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
# 7. Bar 스타일
# ============================================================
bar_colors = [
    "#8CB7D9",  # DDPP
    "#F4B978",  # MaxSNR
    "#8FCB8F",  # PF-HAPPO
    "#E88989",  # Jensen-HAPPO
    "#9467bd",  # HeLyMARL
    "#FF7F0E",  # LyMARL
]

bar_hatches = [
    "--",
    "//",
    "\\\\",
    "xx",
    "",
    "",
]


# ============================================================
# 8. Fairness figure
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 6.0))

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

for bar, hatch in zip(bars, bar_hatches):
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

ax.set_ylabel("Jain's Fairness Index (JFI)")
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=30, ha="right")

finite_fairness_upper = fairness_values + fairness_stds_plot
finite_fairness_upper = finite_fairness_upper[
    np.isfinite(finite_fairness_upper)
]

if finite_fairness_upper.size > 0:
    fairness_ymax = max(
        1.05,
        np.max(finite_fairness_upper) + 0.08,
    )
    ax.set_ylim(0.0, fairness_ymax)
else:
    ax.set_ylim(0.0, 1.10)

ax.set_yticks(np.arange(0.0, 1.01, 0.2))
ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.25)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

fairness_path = os.path.join(
    SAVE_DIR,
    "fairness_comparison.png",
)

plt.savefig(
    fairness_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 9. Throughput figure
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 6.0))

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

for bar, hatch in zip(bars, bar_hatches):
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

ax.set_ylabel("Average Throughput (Gbps)")
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=30, ha="right")

finite_throughput_upper = throughput_values + throughput_stds_plot
finite_throughput_upper = finite_throughput_upper[
    np.isfinite(finite_throughput_upper)
]

if finite_throughput_upper.size > 0:
    ymax = np.max(finite_throughput_upper)
    ax.set_ylim(0.0, ymax * 1.15)
else:
    ax.set_ylim(0.0, 1.0)

ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.25)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

throughput_path = os.path.join(
    SAVE_DIR,
    "throughput_comparison.png",
)

plt.savefig(
    throughput_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close()


print(f"\nSaved fairness figure:   {fairness_path}")
print(f"Saved throughput figure: {throughput_path}")
