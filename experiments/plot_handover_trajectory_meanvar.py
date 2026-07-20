import os
import numpy as np
import matplotlib.pyplot as plt


NPZ_FILES = {
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
        "pf_happo_5seeds_evaluation_summary.npz"
    ),
    "Jensen-HAPPO": (
        "results/results_multi_seed/"
        "jensen_happo_5seeds_evaluation_summary.npz"
    ),
    "HeLyMARL": (
        "results/results_multi_seed/"
        "helymarl_5seeds_evaluation_summary.npz"
    ),
}


# ============================================================
# 2. Plot 설정
# ============================================================
SAVE_DIR = "eval_compare_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_STEPS = 10000
TARGET_KAPPA = 0.03

# cumulative ratio 자체가 이미 smoothing 효과가 있으므로
# 추가 moving average는 기본적으로 사용하지 않음
SMOOTH_WINDOW = 1

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

    if window == 1:
        return x.copy()

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
# 5. Slot HO indicator -> cumulative HO ratio
# ============================================================
def compute_cumulative_handover_ratio(
    handover_indicator,
):
    """
    handover_indicator의 가능한 입력 형태:

        [T]
            각 step에서 사용자 평균 handover indicator

        [T, U]
            step별, 사용자별 handover indicator

        [U, T]
            사용자별, step별 handover indicator

    반환:
        cumulative_ratio: [T]

    정의:
        cumulative_ratio[t]
        = sum_{tau=0}^{t} sum_u h_u(tau)
          / ((t+1) * U)
    """
    handover_indicator = np.asarray(
        handover_indicator,
        dtype=float,
    )

    handover_indicator = np.squeeze(
        handover_indicator
    )

    # --------------------------------------------------------
    # 이미 step별 mean handover indicator인 경우: [T]
    # --------------------------------------------------------
    if handover_indicator.ndim == 1:
        step_handover_ratio = handover_indicator

    # --------------------------------------------------------
    # 사용자별 handover indicator가 포함된 경우: [T,U] or [U,T]
    # --------------------------------------------------------
    elif handover_indicator.ndim == 2:

        # 일반적으로 evaluation step 수가 사용자 수보다 훨씬 큼
        # 긴 축을 time 축으로 판단
        if (
            handover_indicator.shape[0]
            >= handover_indicator.shape[1]
        ):
            # [T, U]
            step_handover_ratio = np.nanmean(
                handover_indicator,
                axis=1,
            )
        else:
            # [U, T]
            step_handover_ratio = np.nanmean(
                handover_indicator,
                axis=0,
            )

    else:
        raise ValueError(
            "handover indicator must have shape "
            f"[T], [T,U], or [U,T], but got "
            f"{handover_indicator.shape}"
        )

    step_handover_ratio = np.nan_to_num(
        step_handover_ratio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cumulative_count = np.cumsum(
        step_handover_ratio
    )

    denominator = np.arange(
        1,
        len(step_handover_ratio) + 1,
        dtype=float,
    )

    cumulative_ratio = (
        cumulative_count / denominator
    )

    return cumulative_ratio


# ============================================================
# 6. 저장된 배열을 [S,T] 형태로 정리
# ============================================================
def ensure_seed_time_shape(
    array,
    eval_seeds=None,
):
    """
    입력 배열을 가능한 한 [S,T] 형태로 변환합니다.

    지원:
        [T]
        [S,T]
        [T,S]

    eval_seeds 길이가 있으면 seed 축 판별에 사용합니다.
    """
    array = np.asarray(
        array,
        dtype=float,
    )

    array = np.squeeze(
        array
    )

    if array.ndim == 1:
        return array[None, :]

    if array.ndim != 2:
        raise ValueError(
            "trajectory array must have shape "
            f"[T], [S,T], or [T,S], but got "
            f"{array.shape}"
        )

    number_of_eval_seeds = (
        len(eval_seeds)
        if eval_seeds is not None
        else 0
    )

    if number_of_eval_seeds > 0:
        if array.shape[0] == number_of_eval_seeds:
            return array

        if array.shape[1] == number_of_eval_seeds:
            return array.T

    # 일반적으로 seed 수보다 trajectory 길이가 훨씬 큼
    if array.shape[0] <= array.shape[1]:
        return array

    return array.T


# ============================================================
# 7. Multi-seed cumulative HO trajectory 불러오기
# ============================================================
def load_multi_seed_handover_trajectory(
    npz_path,
    max_steps,
    smooth_window=1,
):
    """
    반환:
        mean       : [T]
        var        : [T]
        std        : [T]
        per_seed   : [S,T]
        eval_seeds : [S]

    권장 저장 key:
        handover_cumulative_ratio_per_seed : [S,T]

    아래 key들도 자동으로 탐색:
        handover_ratio_trajectory_per_seed
        handover_trajectory_per_seed
        handover_indicator_per_seed
        handover_flags_per_seed
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"File not found: {npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:

        print(
            f"\n[{os.path.basename(npz_path)}]"
        )
        print(
            f"Available keys: {data.files}"
        )

        eval_seeds = np.asarray(
            data.get(
                "eval_seeds",
                [],
            ),
            dtype=int,
        ).reshape(-1)

        # ====================================================
        # Case 1.
        # seed별 cumulative handover ratio가 이미 저장된 경우
        # ====================================================
        cumulative_keys = [
            "handover_cumulative_ratio_per_seed",
            "cumulative_handover_ratio_per_seed",
            "handover_ratio_trajectory_per_seed",
            "handover_trajectory_per_seed",
        ]

        for key in cumulative_keys:
            if key not in data.files:
                continue

            trajectory_per_seed = (
                ensure_seed_time_shape(
                    data[key],
                    eval_seeds=eval_seeds,
                )
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

            return calculate_seed_statistics(
                smoothed_per_seed,
                eval_seeds,
                loaded_key=key,
            )

        # ====================================================
        # Case 2.
        # seed별 slot handover indicator가 저장된 경우
        #
        # 예상 shape:
        #   [S,T]
        #   [S,T,U]
        #   [S,U,T]
        # ====================================================
        indicator_keys = [
            "handover_indicator_per_seed",
            "handover_flags_per_seed",
            "handover_per_seed",
            "handover_event_per_seed",
        ]

        for key in indicator_keys:
            if key not in data.files:
                continue

            raw_indicator = np.asarray(
                data[key],
                dtype=float,
            )

            raw_indicator = np.squeeze(
                raw_indicator
            )

            # 한 개 seed만 저장된 경우
            if raw_indicator.ndim in [1, 2]:
                raw_indicator = (
                    raw_indicator[None, ...]
                )

            if raw_indicator.ndim not in [2, 3]:
                raise ValueError(
                    f"'{key}' must have shape [S,T], "
                    f"[S,T,U], or [S,U,T], but got "
                    f"{raw_indicator.shape}"
                )

            # seed 축이 첫 번째가 아닐 가능성 처리
            if len(eval_seeds) > 0:
                number_of_seeds = len(eval_seeds)

                if (
                    raw_indicator.shape[0]
                    != number_of_seeds
                ):
                    seed_axis_candidates = [
                        axis
                        for axis, size
                        in enumerate(
                            raw_indicator.shape
                        )
                        if size == number_of_seeds
                    ]

                    if seed_axis_candidates:
                        raw_indicator = np.moveaxis(
                            raw_indicator,
                            seed_axis_candidates[0],
                            0,
                        )

            cumulative_per_seed = []

            for seed_indicator in raw_indicator:
                cumulative_ratio = (
                    compute_cumulative_handover_ratio(
                        seed_indicator
                    )
                )

                cumulative_per_seed.append(
                    cumulative_ratio[:max_steps]
                )

            min_length = min(
                len(x)
                for x in cumulative_per_seed
            )

            cumulative_per_seed = np.stack(
                [
                    moving_average(
                        x[:min_length],
                        smooth_window,
                    )
                    for x in cumulative_per_seed
                ],
                axis=0,
            )

            return calculate_seed_statistics(
                cumulative_per_seed,
                eval_seeds,
                loaded_key=key,
            )

        # ====================================================
        # Case 3.
        # mean/std summary만 저장된 경우
        # ====================================================
        mean_keys = [
            "handover_cumulative_ratio_mean",
            "handover_ratio_trajectory_mean",
            "handover_trajectory_mean",
        ]

        std_keys = [
            "handover_cumulative_ratio_std",
            "handover_ratio_trajectory_std",
            "handover_trajectory_std",
        ]

        selected_mean_key = next(
            (
                key
                for key in mean_keys
                if key in data.files
            ),
            None,
        )

        selected_std_key = next(
            (
                key
                for key in std_keys
                if key in data.files
            ),
            None,
        )

        if (
            selected_mean_key is None
            or selected_std_key is None
        ):
            raise KeyError(
                "Handover trajectory key를 찾지 못했습니다.\n"
                f"Available keys: {data.files}\n\n"
                "권장 key:\n"
                "  handover_cumulative_ratio_per_seed [S,T]\n"
                "또는\n"
                "  handover_indicator_per_seed [S,T,U]"
            )

        trajectory_mean = np.asarray(
            data[selected_mean_key],
            dtype=float,
        ).reshape(-1)

        trajectory_std = np.asarray(
            data[selected_std_key],
            dtype=float,
        ).reshape(-1)

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

        return {
            "mean": trajectory_mean,
            "var": trajectory_std ** 2,
            "std": trajectory_std,
            "per_seed": None,
            "eval_seeds": eval_seeds,
            "loaded_key": (
                f"{selected_mean_key}, "
                f"{selected_std_key}"
            ),
        }


# ============================================================
# 8. Seed 통계 계산
# ============================================================
def calculate_seed_statistics(
    trajectory_per_seed,
    eval_seeds,
    loaded_key,
):
    trajectory_per_seed = np.asarray(
        trajectory_per_seed,
        dtype=float,
    )

    trajectory_mean = np.nanmean(
        trajectory_per_seed,
        axis=0,
    )

    ddof = (
        1
        if trajectory_per_seed.shape[0] > 1
        else 0
    )

    trajectory_var = np.nanvar(
        trajectory_per_seed,
        axis=0,
        ddof=ddof,
    )

    trajectory_std = np.nanstd(
        trajectory_per_seed,
        axis=0,
        ddof=ddof,
    )

    return {
        "mean": trajectory_mean,
        "var": trajectory_var,
        "std": trajectory_std,
        "per_seed": trajectory_per_seed,
        "eval_seeds": eval_seeds,
        "loaded_key": loaded_key,
    }


# ============================================================
# 9. 알고리즘별 결과 로드
# ============================================================
trajectory_results = {}

for algorithm, npz_path in NPZ_FILES.items():
    try:
        result = load_multi_seed_handover_trajectory(
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

        final_mean = float(
            result["mean"][-1]
        )

        final_std = float(
            result["std"][-1]
        )

        print(
            f"[{algorithm}] "
            f"key={result['loaded_key']}, "
            f"seeds={number_of_seeds}, "
            f"trajectory length={len(result['mean'])}, "
            f"final cumulative HO ratio="
            f"{final_mean:.4f} ± {final_std:.4f}"
        )

    except Exception as error:
        print(
            f"[WARNING] {algorithm}: {error}"
        )


if not trajectory_results:
    raise RuntimeError(
        "Plot에 사용할 multi-seed handover trajectory가 없습니다."
    )


# ============================================================
# 10. Line style
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
# 11. Figure 생성
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

    # 평균 cumulative handover ratio
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

    lower_bound = np.clip(
        lower_bound,
        0.0,
        None,
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
# 12. Handover budget κ
# ============================================================
ax.axhline(
    y=TARGET_KAPPA,
    linestyle="--",
    linewidth=1.8,
    color="black",
    alpha=0.8,
    label=rf"Target $\kappa={TARGET_KAPPA:g}$",
    zorder=2,
)


# ============================================================
# 13. 축 설정
# ============================================================
ax.set_xlabel("Time step")
ax.set_ylabel("Mean cumulative handover ratio")

ax.set_xlim(
    0,
    MAX_STEPS,
)

# 실제 결과에 맞춰 자동 범위 설정
all_upper_values = [
    np.nanmax(
        result["mean"]
        + BAND_SCALE * result["std"]
    )
    for result in trajectory_results.values()
]

maximum_value = max(
    max(all_upper_values),
    TARGET_KAPPA,
)

# 예: 최대값이 0.05이면 약 0.06까지 표시
y_upper = max(
    0.04,
    maximum_value * 1.12,
)

ax.set_ylim(
    0.0,
    y_upper,
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

ax.grid(
    True,
    linestyle="-",
    linewidth=0.7,
    alpha=0.3,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# 14. Legend
# ============================================================
ax.legend(
    loc="upper right",
    ncol=2,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
)


plt.tight_layout()


# ============================================================
# 15. 저장
# ============================================================
png_path = os.path.join(
    SAVE_DIR,
    "mean_cumulative_handover_ratio_"
    "5algorithms_kappa003_5seeds.png",
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved PNG: {png_path}")