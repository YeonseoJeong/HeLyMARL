import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 결과 파일 경로
# ============================================================
RESULT_DIR = "results/results_V_sweep"

EVAL_PATTERN = os.path.join(
    RESULT_DIR,
    "HeLyMARL_eval_hard_V_*.npz",
)


# ============================================================
# 2. Plot 설정
# ============================================================
SAVE_DIR = "eval_V_sweep_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_STEPS = 10000
SMOOTH_WINDOW = 100
TARGET_ON_RATIO = 0.6


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

    출력 길이는 입력과 동일합니다.
    각 시점 t에서 과거 window개 슬롯만 이용합니다.
    """
    x = np.asarray(x, dtype=float).reshape(-1)

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
            cumulative_sum[t + 1]
            - cumulative_sum[start]
        ) / count

    return result


# ============================================================
# 5. 파일명에서 V 추출
# ============================================================
def extract_eval_v(npz_path):
    """
    예:
        HeLyMARL_eval_hard_V_5.0.npz  -> 5.0
        HeLyMARL_eval_hard_V_50.npz   -> 50.0
    """
    file_name = os.path.basename(npz_path)

    match = re.search(
        r"HeLyMARL_eval_hard_V_"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\.npz$",
        file_name,
    )

    if match is None:
        return None

    return float(match.group(1))


# ============================================================
# 6. power_mat 불러오기
# ============================================================
def load_bs_on_matrix(npz_path):
    """
    반환:
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

            power_mat = np.squeeze(power_mat)

            if power_mat.ndim != 2:
                raise ValueError(
                    f"'power_mat' must be 2-D, "
                    f"but got shape {power_mat.shape}"
                )

            # [T, B]라면 [B, T]로 transpose
            if (
                power_mat.shape[0] > power_mat.shape[1]
                and power_mat.shape[1] <= 20
            ):
                power_mat = power_mat.T

            return (
                power_mat > 0.0
            ).astype(np.float32)

        # ----------------------------------------------------
        # Case 2: bs_on_mat
        # ----------------------------------------------------
        if "bs_on_mat" in data.files:
            bs_on_mat = np.asarray(
                data["bs_on_mat"],
                dtype=float,
            )

            bs_on_mat = np.squeeze(bs_on_mat)

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
        # Case 3: power_bs1, power_bs2, ...
        # ----------------------------------------------------
        power_keys = sorted(
            [
                key
                for key in data.files
                if key.startswith("power_bs")
            ],
            key=lambda key: int(
                re.search(r"\d+", key).group()
            ),
        )

        if power_keys:
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
# 7. V별 ON-ratio trajectory 계산
# ============================================================
eval_files = glob.glob(EVAL_PATTERN)

if not eval_files:
    raise FileNotFoundError(
        "V-sweep evaluation 파일을 찾을 수 없습니다.\n"
        f"검색 경로: {EVAL_PATTERN}"
    )


v_file_pairs = []

for path in eval_files:
    eval_v = extract_eval_v(path)

    if eval_v is None:
        print(
            f"[WARNING] V 값을 읽지 못해 제외: {path}"
        )
        continue

    v_file_pairs.append(
        (eval_v, path)
    )


v_file_pairs.sort(
    key=lambda item: item[0]
)


trajectories = {}

for eval_v, path in v_file_pairs:
    try:
        bs_on_mat = load_bs_on_matrix(path)

        T = min(
            MAX_STEPS,
            bs_on_mat.shape[1],
        )

        bs_on_mat = bs_on_mat[:, :T]

        # 각 슬롯에서 전체 BS 중 켜진 비율
        mean_on_per_slot = np.mean(
            bs_on_mat,
            axis=0,
        )

        # ON/OFF 동작을 보기 위한 causal smoothing
        smoothed_on_ratio = moving_average(
            mean_on_per_slot,
            SMOOTH_WINDOW,
        )

        trajectories[eval_v] = smoothed_on_ratio

        print(
            f"[V={eval_v:g}] "
            f"power shape={bs_on_mat.shape}, "
            f"overall ON mean={np.mean(mean_on_per_slot):.4f}, "
            f"final 100-slot mean="
            f"{np.mean(mean_on_per_slot[-100:]):.4f}"
        )

    except Exception as error:
        print(
            f"[WARNING] V={eval_v:g}: {error}"
        )


if not trajectories:
    raise RuntimeError(
        "Plot에 사용할 ON-ratio trajectory가 없습니다."
    )


# ============================================================
# 8. V별 line style
# ============================================================
linestyles = [
    "-",
    "--",
    "-.",
    ":",
]

markers = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
]


# ============================================================
# 9. Figure 생성
# ============================================================
fig, ax = plt.subplots(
    figsize=(9.2, 5.4)
)

for index, (eval_v, trajectory) in enumerate(
    trajectories.items()
):
    x = np.arange(
        1,
        len(trajectory) + 1,
    )

    marker_interval = max(
        1,
        len(trajectory) // 10,
    )

    ax.plot(
        x,
        trajectory,
        label=rf"$V={eval_v:g}$",
        linestyle=linestyles[
            index % len(linestyles)
        ],
        marker=markers[
            index % len(markers)
        ],
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
    label=rf"Target $\rho={TARGET_ON_RATIO:g}$",
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
    loc="best",
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
    "HeLyMARL_V_sweep_mean_bs_on_ratio.png",
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved PNG: {png_path}")