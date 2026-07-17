import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Multi-seed summary 파일 경로
# ============================================================
NPZ_FILES = {
    "DDPP": (
        "results/results_multi_seed/"
        "ddpp_5seeds_on_ratio.npz"
    ),
    "MaxSNR": (
        "results/results_multi_seed/"
        "maxsnr_5seeds_on_ratio.npz"
    ),
    "PF-HAPPO": (
        "results/results_multi_seed/"
        "pf_happo_5seeds_on_ratio.npz"
    ),
    "Jensen-HAPPO": (
        "results/results_multi_seed/"
        "jensen_happo_5seeds_on_ratio.npz"
    ),
    "HeLyMARL": (
        "results/results_multi_seed/"
        "helymarl_5seeds_on_ratio.npz"
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

# 평균 ± BAND_SCALE × 표준편차
BAND_SCALE = 1.0
BAND_ALPHA = 0.20


# ============================================================
# 3. 논문용 스타일
# ============================================================
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
# 4. Causal moving average
# ============================================================
def moving_average(x, window):
    """
    Causal moving average.

    출력 길이는 입력 길이와 동일합니다.
    """
    x = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)

    if x.size == 0:
        return x

    window = max(
        1,
        int(window),
    )

    cumulative_sum = np.cumsum(
        np.insert(x, 0, 0.0)
    )

    result = np.empty_like(
        x,
        dtype=float,
    )

    for t in range(len(x)):
        start = max(
            0,
            t - window + 1,
        )

        count = t - start + 1

        result[t] = (
            cumulative_sum[t + 1]
            - cumulative_sum[start]
        ) / count

    return result


# ============================================================
# 5. Multi-seed trajectory 불러오기
# ============================================================
def load_multi_seed_trajectory(
    npz_path,
    max_steps,
    smooth_window,
):
    """
    반환:
        trajectory_mean: [T]
        trajectory_var : [T]
        trajectory_std : [T]
        eval_seeds      : [S]

    우선 on_ratio_trajectory_per_seed를 사용해
    각 seed를 smoothing한 뒤 통계를 계산합니다.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"File not found: {npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:

        eval_seeds = np.asarray(
            data.get(
                "eval_seeds",
                [],
            ),
            dtype=int,
        ).reshape(-1)

        # ----------------------------------------------------
        # 권장 방식:
        # seed별 raw trajectory -> 각각 smoothing -> 통계 계산
        # ----------------------------------------------------
        if "on_ratio_trajectory_per_seed" in data.files:
            trajectory_per_seed = np.asarray(
                data["on_ratio_trajectory_per_seed"],
                dtype=float,
            )

            trajectory_per_seed = np.squeeze(
                trajectory_per_seed
            )

            if trajectory_per_seed.ndim == 1:
                trajectory_per_seed = (
                    trajectory_per_seed[None, :]
                )

            if trajectory_per_seed.ndim != 2:
                raise ValueError(
                    "'on_ratio_trajectory_per_seed' must "
                    f"have shape [S, T], but got "
                    f"{trajectory_per_seed.shape}"
                )

            T = min(
                max_steps,
                trajectory_per_seed.shape[1],
            )

            trajectory_per_seed = (
                trajectory_per_seed[:, :T]
            )

            smoothed_per_seed = np.stack(
                [
                    moving_average(
                        trajectory,
                        smooth_window,
                    )
                    for trajectory
                    in trajectory_per_seed
                ],
                axis=0,
            )

            trajectory_mean = np.nanmean(
                smoothed_per_seed,
                axis=0,
            )

            ddof = (
                1
                if smoothed_per_seed.shape[0] > 1
                else 0
            )

            trajectory_var = np.nanvar(
                smoothed_per_seed,
                axis=0,
                ddof=ddof,
            )

            trajectory_std = np.nanstd(
                smoothed_per_seed,
                axis=0,
                ddof=ddof,
            )

            return {
                "mean": trajectory_mean,
                "var": trajectory_var,
                "std": trajectory_std,
                "per_seed": smoothed_per_seed,
                "eval_seeds": eval_seeds,
            }

        # ----------------------------------------------------
        # Fallback:
        # summary mean/std만 저장된 경우
        # ----------------------------------------------------
        required_keys = [
            "on_ratio_trajectory_mean",
            "on_ratio_trajectory_std",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in data.files
        ]

        if missing_keys:
            raise KeyError(
                f"Missing keys: {missing_keys}\n"
                f"Available keys: {data.files}"
            )

        trajectory_mean = np.asarray(
            data["on_ratio_trajectory_mean"],
            dtype=float,
        ).reshape(-1)

        trajectory_std = np.asarray(
            data["on_ratio_trajectory_std"],
            dtype=float,
        ).reshape(-1)

        if "on_ratio_trajectory_var" in data.files:
            trajectory_var = np.asarray(
                data["on_ratio_trajectory_var"],
                dtype=float,
            ).reshape(-1)
        else:
            trajectory_var = (
                trajectory_std ** 2
            )

        T = min(
            max_steps,
            len(trajectory_mean),
            len(trajectory_std),
        )

        trajectory_mean = moving_average(
            trajectory_mean[:T],
            smooth_window,
        )

        trajectory_std = moving_average(
            trajectory_std[:T],
            smooth_window,
        )

        trajectory_var = moving_average(
            trajectory_var[:T],
            smooth_window,
        )

        return {
            "mean": trajectory_mean,
            "var": trajectory_var,
            "std": trajectory_std,
            "per_seed": None,
            "eval_seeds": eval_seeds,
        }


# ============================================================
# 6. 알고리즘별 결과 로드
# ============================================================
trajectory_results = {}

for algorithm, npz_path in NPZ_FILES.items():
    try:
        result = load_multi_seed_trajectory(
            npz_path=npz_path,
            max_steps=MAX_STEPS,
            smooth_window=SMOOTH_WINDOW,
        )

        trajectory_results[algorithm] = result

        number_of_seeds = (
            result["per_seed"].shape[0]
            if result["per_seed"] is not None
            else len(result["eval_seeds"])
        )

        overall_mean = float(
            np.mean(result["mean"])
        )

        mean_std = float(
            np.mean(result["std"])
        )

        print(
            f"[{algorithm}] "
            f"seeds={number_of_seeds}, "
            f"trajectory length={len(result['mean'])}, "
            f"overall ON mean={overall_mean:.4f}, "
            f"mean temporal std={mean_std:.4f}"
        )

    except Exception as error:
        print(
            f"[WARNING] {algorithm}: {error}"
        )


if not trajectory_results:
    raise RuntimeError(
        "Plot에 사용할 multi-seed trajectory가 없습니다."
    )


# ============================================================
# 7. Line style
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
# 8. Figure 생성
# ============================================================
fig, ax = plt.subplots(
    figsize=(9.2, 5.4)
)

for algorithm, result in trajectory_results.items():
    trajectory_mean = result["mean"]
    trajectory_std = result["std"]

    x = np.arange(
        1,
        len(trajectory_mean) + 1,
    )

    style = line_styles.get(
        algorithm,
        {
            "linestyle": "-",
            "marker": "o",
        },
    )

    marker_interval = max(
        1,
        len(trajectory_mean) // 10,
    )

    # 평균 trajectory
    line, = ax.plot(
        x,
        trajectory_mean,
        label=algorithm,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markevery=marker_interval,
        markersize=7,
        linewidth=2.3,
        alpha=0.95,
        zorder=3,
    )

    line_color = line.get_color()

    # 평균 ± 표준편차
    lower_bound = (
        trajectory_mean
        - BAND_SCALE * trajectory_std
    )

    upper_bound = (
        trajectory_mean
        + BAND_SCALE * trajectory_std
    )

    # ON-ratio 범위 제한
    lower_bound = np.clip(
        lower_bound,
        0.0,
        1.0,
    )

    upper_bound = np.clip(
        upper_bound,
        0.0,
        1.0,
    )

    ax.fill_between(
        x,
        lower_bound,
        upper_bound,
        color=line_color,
        alpha=BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )


# ============================================================
# 9. 목표 ON ratio
# ============================================================
ax.axhline(
    y=TARGET_ON_RATIO,
    linestyle="--",
    linewidth=1.8,
    color="black",
    alpha=0.8,
    label=rf"Target $\rho={TARGET_ON_RATIO:g}$",
    zorder=2,
)


# ============================================================
# 10. 축 설정
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
    np.arange(
        0.0,
        1.01,
        0.2,
    )
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
# 11. Legend
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
# 12. 저장
# ============================================================
png_path = os.path.join(
    SAVE_DIR,
    "mean_bs_on_ratio_5algorithms_5seeds.png",
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved PNG: {png_path}")