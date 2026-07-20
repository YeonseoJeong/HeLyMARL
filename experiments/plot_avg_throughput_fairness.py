import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. NPZ 경로
# ============================================================
NPZ_FILES = {
    "DDPP": (
        "results/results_compare/"
        "DDPP_eval_lambda_0.0.npz"
    ),
    "MaxSNR": (
        "results/results_compare/"
        "MaxSNR_eval_lambda_0.0.npz"
    ),
    "PF-HAPPO": (
        "results/policy_improvement/pf/"
        "ConstrainedHAPPO_pf_gamma_0.5_final_eval_kappa_0.03.npz"
    ),
    "Jensen-HAPPO": (
        "results/results_baselines/"
        "ConstrainedHAPPO_jensen_eval_hard_kappa_0.03_use_dimensionless.npz"
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

def block_jain_fairness(
    slot_rates,
    block_size=1000,
    eps=1e-12,
):
    """
    slot_rates를 block_size 단위로 나누어 JFI를 계산한다.

    - 입력: [T, U] 또는 [U, T]
    - 각 블록에서 UE별 평균 rate를 구한 뒤 JFI 계산
    - 모든 UE의 평균 rate가 0인 all-off 블록은 제외
    - 마지막 블록이 1000 step보다 짧아도 포함
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

        # 해당 1000-step 블록에서 UE별 평균 rate
        block_user_rates = np.mean(block, axis=0)

        # 모든 UE의 rate가 0인 all-off 블록 제외
        if np.sum(block_user_rates ** 2) <= eps:
            continue

        block_jfi = jain_fairness(
            block_user_rates,
            eps=eps,
        )

        if np.isfinite(block_jfi):
            block_jfis.append(block_jfi)

    if len(block_jfis) == 0:
        return np.nan

    return float(np.mean(block_jfis))


# def get_fairness(data):
#     """
#     우선순위:
#     1) fairness
#     2) episode_fairness_last
#     3) fairness_block_jfis 평균
#     4) avg_user_rates로 직접 계산
#     5) slot_rates로 직접 계산
#     """
#     if "fairness" in data.files:
#         value = to_scalar_mean(data["fairness"])

#         if not np.isnan(value):
#             return value

#     if "episode_fairness_last" in data.files:
#         value = to_scalar_mean(data["episode_fairness_last"])

#         if not np.isnan(value):
#             return value

#     if "fairness_block_jfis" in data.files:
#         value = to_scalar_mean(data["fairness_block_jfis"])

#         if not np.isnan(value):
#             return value

#     if "avg_user_rates" in data.files:
#         return jain_fairness(data["avg_user_rates"])

#     if "slot_rates" in data.files:
#         return jain_fairness(data["slot_rates"])

#     return np.nan

def get_fairness(data, block_size=1000):
    """
    우선순위:
    1) slot_rates를 1000-step 블록으로 직접 계산
       - 모든 UE rate가 0인 all-off 블록 제외
    2) 저장된 fairness_block_jfis 사용
       - 0 또는 NaN 블록 제외
    3) 기존 fairness
    4) episode_fairness_last
    5) avg_user_rates
    """

    # 새 방식: slot_rates에서 1000-step block JFI 직접 계산
    if "slot_rates" in data.files:
        value = block_jain_fairness(
            data["slot_rates"],
            block_size=block_size,
        )

        if np.isfinite(value):
            return value

    # slot_rates가 없을 때 저장된 block JFI 사용
    if "fairness_block_jfis" in data.files:
        block_jfis = np.asarray(
            data["fairness_block_jfis"],
            dtype=float,
        ).reshape(-1)

        # all-off 블록이 0으로 저장된 경우 제외
        valid_mask = (
            np.isfinite(block_jfis)
            & (block_jfis > 0.0)
        )

        if np.any(valid_mask):
            return float(
                np.mean(block_jfis[valid_mask])
            )

    # 아래는 기존 결과와의 호환성을 위한 fallback
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

            if not np.isnan(value):
                return value

    return np.nan

def get_avg_user_rates(data):
    if "avg_user_rates" in data.files:
        rates = np.asarray(data["avg_user_rates"], dtype=float)
        rates = np.squeeze(rates)

        if rates.ndim == 1:
            return np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        if rates.ndim == 2:
            if rates.shape[0] >= rates.shape[1]:
                user_rates = np.nanmean(rates, axis=0)
            else:
                user_rates = np.nanmean(rates, axis=1)
            return np.nan_to_num(user_rates, nan=0.0, posinf=0.0, neginf=0.0)
    
    if "slot_rates" in data.files:
        rates = np.asarray(data["slot_rates"], dtype=float)
        rates = np.squeeze(rates)

        if rates.ndim == 1:
            return np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        if rates.ndim == 2:
            if rates.shape[0] >= rates.shape[1]:
                user_rates = np.nanmean(rates, axis=0)
            else:
                user_rates = np.nanmean(rates, axis=1)
    return None

def get_network_utility(data, eps= 1e-8):
    avg_user_rates = get_avg_user_rates(data)
    if avg_user_rates is None:
        return np.nan
    avg_user_rates = np.asarray(avg_user_rates, dtype=float).reshape(-1)
    if avg_user_rates.size == 0:
        return np.nan
    avg_user_rates = np.nan_to_num(avg_user_rates, nan=0.0, posinf=0.0, neginf=0.0)
    avg_user_rates = np.maximum(avg_user_rates, 0.0)
    avg_user_rates_bps = avg_user_rates * 1e9
    return float(np.sum(np.log(avg_user_rates_bps + eps)))

# ============================================================
# 6. 결과 읽기
# ============================================================
algorithms = []
fairness_values = []
throughput_values = []
utility_values = []

for algorithm, path in NPZ_FILES.items():

    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        continue

    with np.load(path, allow_pickle=True) as data:
        fairness = get_fairness(data)
        throughput = get_throughput(data)
        utility = get_network_utility(data, eps=1e-8)

        algorithms.append(DISPLAY_NAMES[algorithm])
        fairness_values.append(fairness)
        throughput_values.append(throughput)
        utility_values.append(utility)

        print(
            f"{algorithm:15s} | "
            f"Fairness={fairness:.4f} | "
            f"Throughput={throughput:.4f} | "
            f"Objective={utility:.4f}"
        )


algorithms = np.asarray(algorithms)
fairness_values = np.asarray(fairness_values, dtype=float)
throughput_values = np.asarray(throughput_values, dtype=float)
utility_values = np.asarray(utility_values, dtype=float)
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

# ============================================================
# 10. Objective function figure
#     Equation (7):
#     sum_u log((1/T) sum_t R_u(t))
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 6.0))

utility_plot_values = utility_values / 1e2

def truncate(value, decimals=2):
    factor = 10 ** decimals
    return np.trunc(value * factor) / factor


bars = ax.bar(
    x,
    utility_plot_values,
    width=0.62,
    color=bar_colors,
    edgecolor="black",
    linewidth=1.2,
) 

for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)

finite_utility = utility_plot_values[
    np.isfinite(utility_plot_values)
]

if finite_utility.size > 0:
    utility_range = (
        np.max(finite_utility)
        - np.min(finite_utility)
    )

    text_offset = max(
        utility_range * 0.04,
        0.002,
    )
else:
    text_offset = 0.002

for bar, value in zip(bars, utility_plot_values):
    if np.isfinite(value):
        truncated_value = truncate(
            value,
            decimals=2,
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + text_offset,
            f"{truncated_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )


ax.set_ylabel(
    r"Network Utility "
    r"$\sum_{u=1}^{U}\log(\bar{R}_u)$ "
    r"$(\times 10^2)$"
)

ax.set_xticks(x)
ax.set_xticklabels(
    algorithms,
    rotation=30,
    ha="right",
)

if finite_utility.size > 0:
    ymin = np.min(finite_utility)
    ymax = np.max(finite_utility)

    margin = max(
        (ymax - ymin) * 0.25,
        0.01,
    )

    ax.set_ylim(
        ymin - margin,
        ymax + margin,
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

utility_path = os.path.join(
    SAVE_DIR,
    "objective_comparison.png",
)

plt.savefig(
    utility_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(f"\nSaved fairness figure:   {fairness_path}")
print(f"Saved throughput figure: {throughput_path}")
print(f"Saved objective figure:  {utility_path}")