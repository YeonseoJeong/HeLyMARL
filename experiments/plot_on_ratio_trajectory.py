import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 결과 파일 경로
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
        "results/results_baselines/"
        "ConstrainedHAPPO_pf_eval_hard_kappa_0.03_use_dimensionless.npz"
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
# 2. Plot 설정
# ============================================================
SAVE_DIR = "eval_compare_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_STEPS = 10000
SMOOTH_WINDOW = 100
TARGET_ON_RATIO = 0.6


# 논문용 스타일
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
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
    """
    Causal moving average.

    출력 길이는 입력과 동일하게 유지.
    초반 window 미만 구간은 현재까지 존재하는 값들로 평균.
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return x

    window = max(1, int(window))

    cumulative_sum = np.cumsum(
        np.insert(x, 0, 0.0)
    )

    result = np.empty_like(x, dtype=float)

    for t in range(len(x)):
        start = max(0, t - window + 1)
        count = t - start + 1

        result[t] = (
            cumulative_sum[t + 1] - cumulative_sum[start]
        ) / count

    return result


# ============================================================
# 4. power_mat 불러오기
# ============================================================
def load_bs_on_matrix(npz_path):
    """
    반환값:
        bs_on_mat: shape [B, T]

    지원 key:
        1) power_mat
        2) bs_on_mat
        3) power_bs1, power_bs2, ...
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"File not found: {npz_path}"
        )

    with np.load(npz_path, allow_pickle=True) as data:

        # ----------------------------------------------------
        # Case 1: power_mat
        # ----------------------------------------------------
        if "power_mat" in data.files:
            power_mat = np.asarray(
                data["power_mat"],
                dtype=float,
            )

            if power_mat.ndim != 2:
                raise ValueError(
                    f"'power_mat' must be 2-D, "
                    f"but got shape {power_mat.shape}"
                )

            # 일반적으로 [B, T]
            # 만약 [T, B] 형태이면 transpose
            if (
                power_mat.shape[0] > power_mat.shape[1]
                and power_mat.shape[1] <= 20
            ):
                power_mat = power_mat.T

            bs_on_mat = (
                power_mat > 0.0
            ).astype(np.float32)

            return bs_on_mat

        # ----------------------------------------------------
        # Case 2: 이미 binary ON matrix가 저장된 경우
        # ----------------------------------------------------
        if "bs_on_mat" in data.files:
            bs_on_mat = np.asarray(
                data["bs_on_mat"],
                dtype=float,
            )

            if bs_on_mat.ndim != 2:
                raise ValueError(
                    f"'bs_on_mat' must be 2-D, "
                    f"but got shape {bs_on_mat.shape}"
                )

            if (
                bs_on_mat.shape[0] > bs_on_mat.shape[1]
                and bs_on_mat.shape[1] <= 20
            ):
                bs_on_mat = bs_on_mat.T

            return (
                bs_on_mat > 0.0
            ).astype(np.float32)

        # ----------------------------------------------------
        # Case 3: power_bs1, power_bs2, ... 개별 저장
        # ----------------------------------------------------
        power_keys = sorted([
            key
            for key in data.files
            if key.startswith("power_bs")
        ])

        if len(power_keys) > 0:
            power_list = []

            for key in power_keys:
                power = np.asarray(
                    data[key],
                    dtype=float,
                ).reshape(-1)

                power_list.append(power)

            min_length = min(
                len(power)
                for power in power_list
            )

            power_mat = np.stack(
                [
                    power[:min_length]
                    for power in power_list
                ],
                axis=0,
            )

            return (
                power_mat > 0.0
            ).astype(np.float32)

        raise KeyError(
            f"No usable ON/OFF information found in {npz_path}\n"
            f"Available keys: {data.files}"
        )


# ============================================================
# 5. 알고리즘별 평균 ON-ratio trajectory 계산
# ============================================================
trajectories = {}

for algorithm, path in NPZ_FILES.items():
    try:
        bs_on_mat = load_bs_on_matrix(path)

        # 처음 MAX_STEPS만 사용
        T = min(
            MAX_STEPS,
            bs_on_mat.shape[1],
        )

        bs_on_mat = bs_on_mat[:, :T]

        # 매 슬롯에서 BS 평균 ON 상태
        mean_on_per_slot = np.mean(
            bs_on_mat,
            axis=0,
        )

        # moving average
        smoothed_on_ratio = moving_average(
            mean_on_per_slot,
            SMOOTH_WINDOW,
        )

        trajectories[algorithm] = smoothed_on_ratio

        print(
            f"[{algorithm}] "
            f"power shape={bs_on_mat.shape}, "
            f"overall ON mean={np.mean(mean_on_per_slot):.4f}"
        )

    except Exception as error:
        print(
            f"[WARNING] {algorithm}: {error}"
        )


# ============================================================
# 6. Plot 스타일
# ============================================================
line_styles = {
    "DDPP": {
        "linestyle": ":",
        "marker": "s",
    },
    "MaxSNR": {
        "linestyle": "--",
        "marker": "v",
    },
    "PF-HAPPO": {
        "linestyle": "-.",
        "marker": "^",
    },
    "Jensen-HAPPO": {
        "linestyle": "--",
        "marker": "D",
    },
    "HeLyMARL": {
        "linestyle": "-",
        "marker": "o",
    },
}


# ============================================================
# 7. Figure 생성
# ============================================================
fig, ax = plt.subplots(
    figsize=(9.2, 5.4)
)

for algorithm, trajectory in trajectories.items():
    x = np.arange(1, len(trajectory) + 1)

    style = line_styles.get(
        algorithm,
        {
            "linestyle": "-",
            "marker": "o",
        },
    )

    # marker 위치
    marker_interval = max(
        1,
        len(trajectory) // 10,
    )

    ax.plot(
        x,
        trajectory,
        label=algorithm,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markevery=marker_interval,
        markersize=7,
        linewidth=2.3,
        alpha=0.95,
    )


# 목표 ON ratio
ax.axhline(
    y=TARGET_ON_RATIO,
    linestyle="--",
    linewidth=1.8,
    color="black",
    alpha=0.8,
    label=r"Target $\rho=0.6$",
)


# ============================================================
# 8. 축 설정
# ============================================================
ax.set_xlabel("Time step")
ax.set_ylabel("Mean BS ON-ratio")

ax.set_xlim(
    0,
    MAX_STEPS,
)

ax.set_ylim(
    0.0,
    1.05,
)

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

ax.set_yticks(
    np.arange(0.0, 1.01, 0.2)
)

ax.grid(
    True,
    linestyle="-",
    linewidth=0.7,
    alpha=0.3,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# 9. Legend
# ============================================================
ax.legend(
    loc="lower left",
    bbox_to_anchor=(0.02, 0.03),
    ncol=2,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
)


plt.tight_layout()


# ============================================================
# 10. 저장
# ============================================================
png_path = os.path.join(
    SAVE_DIR,
    "mean_bs_on_ratio_5algorithms.png",
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved PNG: {png_path}")