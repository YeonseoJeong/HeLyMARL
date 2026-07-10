import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. NPZ 경로
# ============================================================
NPZ_FILES = {
    "DDPP": (
        "results/results_compare/"
        "DDPP_eval_lambda_0.npz"
    ),
    "MaxSNR": (
        "results/results_compare/"
        "MaxSNR_eval_lambda_0.0.npz"
    ),
    "PF-HAPPO": (
        "results/results_baselines/"
        "ConstrainedHAPPO_pf_eval_hard_kappa_0.03.npz"
    ),
    "Jensen-HAPPO": (
        "results/results_baselines/"
        "ConstrainedHAPPO_jensen_eval_hard_kappa_0.03.npz"
    ),
    "HeLyMARL": (
        "results/results_kappa/"
        "HeLyMARL_eval_hard_kappa_0.03.npz"
    ),
}


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

    # 수식 글꼴
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

    if arr.size == 0:
        return np.nan

    if np.all(np.isnan(arr)):
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


def get_fairness(data):
    """
    우선순위:
    1) fairness
    2) episode_fairness_last
    3) fairness_block_jfis 평균
    4) avg_user_rates로 직접 계산
    5) slot_rates로 직접 계산
    """
    if "fairness" in data.files:
        value = to_scalar_mean(data["fairness"])

        if not np.isnan(value):
            return value

    if "episode_fairness_last" in data.files:
        value = to_scalar_mean(data["episode_fairness_last"])

        if not np.isnan(value):
            return value

    if "fairness_block_jfis" in data.files:
        value = to_scalar_mean(data["fairness_block_jfis"])

        if not np.isnan(value):
            return value

    if "avg_user_rates" in data.files:
        return jain_fairness(data["avg_user_rates"])

    if "slot_rates" in data.files:
        return jain_fairness(data["slot_rates"])

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

            if not np.isnan(value):
                return value

    return np.nan


# ============================================================
# 6. 결과 읽기
# ============================================================
algorithms = []
fairness_values = []
throughput_values = []

for algorithm, path in NPZ_FILES.items():

    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        continue

    with np.load(path, allow_pickle=True) as data:
        fairness = get_fairness(data)
        throughput = get_throughput(data)

        algorithms.append(DISPLAY_NAMES[algorithm])
        fairness_values.append(fairness)
        throughput_values.append(throughput)

        print(
            f"{algorithm:15s} | "
            f"Fairness={fairness:.4f} | "
            f"Throughput={throughput:.4f}"
        )


algorithms = np.asarray(algorithms)
fairness_values = np.asarray(fairness_values, dtype=float)
throughput_values = np.asarray(throughput_values, dtype=float)

x = np.arange(len(algorithms))


# ============================================================
# 7. Bar 스타일
# ============================================================
bar_colors = [
    "#8CB7D9",  # DDPP        : light blue
    "#F4B978",  # MaxSNR      : light orange
    "#8FCB8F",  # PF-HAPPO    : light green
    "#E88989",  # Jensen-HAPPO: light red
    "#9467bd",  # HeLyMARL    : purple
]

bar_hatches = [
    "--",
    "//",
    "\\\\",
    "xx",
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
    color=bar_colors,
    edgecolor="black",
    linewidth=1.2,
)

for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)


# 막대 위 값 표시
for bar, value in zip(bars, fairness_values):
    if np.isfinite(value):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )


ax.set_ylabel("Jain's Fairness Index (JFI)")

ax.set_xticks(x)
ax.set_xticklabels(
    algorithms,
    rotation=30,
    ha="right",
)

ax.set_ylim(0.0, 1.10)
ax.set_yticks(np.arange(0.0, 1.01, 0.2))

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
    color=bar_colors,
    edgecolor="black",
    linewidth=1.2,
)

for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)


# 막대 위 값 표시
for bar, value in zip(bars, throughput_values):
    if np.isfinite(value):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.10,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )


ax.set_ylabel("Average Throughput (Gbps)")

ax.set_xticks(x)
ax.set_xticklabels(
    algorithms,
    rotation=30,
    ha="right",
)

# 최대값에 따라 자동으로 y축 범위 설정
finite_throughput = throughput_values[
    np.isfinite(throughput_values)
]

if finite_throughput.size > 0:
    ymax = np.max(finite_throughput)
    ax.set_ylim(0.0, ymax * 1.18)
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