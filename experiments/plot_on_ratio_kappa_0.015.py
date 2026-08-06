import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Result paths and seed settings
# ============================================================
# DDPP and MaxSNR are read from their existing multi-seed
# summary files.
SUMMARY_NPZ_FILES = {
    "DDPP": (
        "results/results_baselines/"
        "ddpp_5seeds_evaluation_summary.npz"
    ),
    "MaxSNR": (
        "results/results_baselines/"
        "maxsnr_5seeds_evaluation_summary.npz"
    ),
}

# PF-HAPPO and Jensen-HAPPO are read directly from the raw
# train-seed/eval-seed files, in the same manner as HeLyMARL.
#
# Expected folder structures:
#
# results/results_baselines/pf/
#   kappa_0.015_seed_0/eval_seed_2000.npz
#   ...
#
# results/results_baselines/jensen/
#   kappa_0.015_seed_0/eval_seed_2000.npz
#   ...
BASELINE_RAW_CONFIGS = {
    "PF-HAPPO": {
        "result_root": "results/results_baselines/pf",
        "folder_prefix": None,
    },
    "Jensen-HAPPO": {
        "result_root": "results/results_baselines/jensen",
        "folder_prefix": None,
    },
}

# HeLyMARL:
#
# results/results_kappa/
#   HAPPO_kappa_0.015_seed_0/eval_seed_2000.npz
#   ...
HELYMARL_RAW_ROOT = "results/results_kappa"
HELYMARL_FOLDER_PREFIX = "HAPPO"

KAPPA = 0.015
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEEDS = (2000, 2001, 2002, 2003, 2004)


# ============================================================
# 2. Plot settings
# ============================================================
SAVE_DIR = "eval_compare_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_STEPS = 10000
SMOOTH_WINDOW = 100
TARGET_ON_RATIO = 0.6

# Mean ± BAND_SCALE × standard deviation
BAND_SCALE = 1.0
BAND_ALPHA = 0.05


# Paper style
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
# 3. Causal moving average
# ============================================================
def moving_average(x, window):
    """
    Causal moving average with the same output length.
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    if x.size == 0:
        return x

    window = max(1, int(window))

    if window == 1:
        return x.copy()

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
# 4. Statistics helper
# ============================================================
def calculate_seed_statistics(
    trajectory_per_seed,
    seed_ids,
    loaded_key,
):
    """
    trajectory_per_seed: [S, T]
    """
    trajectory_per_seed = np.asarray(
        trajectory_per_seed,
        dtype=float,
    )

    if trajectory_per_seed.ndim != 2:
        raise ValueError(
            "trajectory_per_seed must have shape [S,T], "
            f"but got {trajectory_per_seed.shape}"
        )

    mean = np.nanmean(
        trajectory_per_seed,
        axis=0,
    )

    ddof = 1 if trajectory_per_seed.shape[0] > 1 else 0

    std = np.nanstd(
        trajectory_per_seed,
        axis=0,
        ddof=ddof,
    )

    var = np.nanvar(
        trajectory_per_seed,
        axis=0,
        ddof=ddof,
    )

    return {
        "mean": mean,
        "std": std,
        "var": var,
        "per_seed": trajectory_per_seed,
        "seed_ids": np.asarray(seed_ids, dtype=int),
        "loaded_key": loaded_key,
    }


# ============================================================
# 5. Load an ON/OFF matrix from one raw evaluation file
# ============================================================
def load_bs_on_matrix(npz_path):
    """
    Returns
    -------
    bs_on_mat : ndarray, shape [B, T]

    Supported keys:
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

            # Convert [T,B] to [B,T].
            if (
                power_mat.shape[0] > power_mat.shape[1]
                and power_mat.shape[1] <= 20
            ):
                power_mat = power_mat.T

            return (
                power_mat > 0.0
            ).astype(np.float32), "power_mat"

        # ----------------------------------------------------
        # Case 2: binary ON matrix
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
            ).astype(np.float32), "bs_on_mat"

        # ----------------------------------------------------
        # Case 3: power_bs1, power_bs2, ...
        # ----------------------------------------------------
        power_keys = sorted(
            key
            for key in data.files
            if key.startswith("power_bs")
        )

        if power_keys:
            power_list = [
                np.asarray(
                    data[key],
                    dtype=float,
                ).reshape(-1)
                for key in power_keys
            ]

            common_length = min(
                len(power)
                for power in power_list
            )

            power_mat = np.stack(
                [
                    power[:common_length]
                    for power in power_list
                ],
                axis=0,
            )

            return (
                power_mat > 0.0
            ).astype(np.float32), ", ".join(power_keys)

        raise KeyError(
            "No usable ON/OFF information found.\n"
            f"File: {npz_path}\n"
            f"Available keys: {data.files}"
        )


def load_on_trajectory_from_eval(
    npz_path,
    max_steps,
    smooth_window,
):
    """
    Loads one raw evaluation file and returns the mean BS
    ON-ratio trajectory [T].
    """
    bs_on_mat, loaded_key = load_bs_on_matrix(npz_path)

    T = min(
        int(max_steps),
        int(bs_on_mat.shape[1]),
    )

    bs_on_mat = bs_on_mat[:, :T]

    mean_on_per_slot = np.nanmean(
        bs_on_mat,
        axis=0,
    )

    mean_on_per_slot = np.nan_to_num(
        mean_on_per_slot,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    trajectory = moving_average(
        mean_on_per_slot,
        smooth_window,
    )

    return trajectory, loaded_key


# ============================================================
# 6. Load PF/Jensen/HeLyMARL raw files
# ============================================================
def make_raw_run_dir(
    result_root,
    folder_prefix,
    kappa,
    train_seed,
):
    kappa_tag = f"{kappa:.3f}"

    if folder_prefix is None:
        folder_name = (
            f"kappa_{kappa_tag}_seed_{train_seed}"
        )
    else:
        folder_name = (
            f"{folder_prefix}_kappa_{kappa_tag}_seed_{train_seed}"
        )

    return os.path.join(
        result_root,
        folder_name,
    )


def load_raw_multiseed_on_trajectory(
    result_root,
    folder_prefix,
    kappa,
    train_seeds,
    eval_seeds,
    max_steps,
    smooth_window,
    method_name,
):
    """
    Aggregation:
        1) Average the five eval seeds within each train seed.
        2) Compute mean/std across the three train seeds.
    """
    trajectory_per_train_seed = []
    successful_train_seeds = []
    loaded_keys = set()

    for train_seed in train_seeds:
        run_dir = make_raw_run_dir(
            result_root=result_root,
            folder_prefix=folder_prefix,
            kappa=kappa,
            train_seed=train_seed,
        )

        eval_trajectories = []
        successful_eval_seeds = []

        for eval_seed in eval_seeds:
            eval_path = os.path.join(
                run_dir,
                f"eval_seed_{eval_seed}.npz",
            )

            if not os.path.exists(eval_path):
                print(
                    f"[Warning] {method_name} file not found: "
                    f"{eval_path}"
                )
                continue

            trajectory, loaded_key = (
                load_on_trajectory_from_eval(
                    npz_path=eval_path,
                    max_steps=max_steps,
                    smooth_window=smooth_window,
                )
            )

            eval_trajectories.append(trajectory)
            successful_eval_seeds.append(eval_seed)
            loaded_keys.add(loaded_key)

        if not eval_trajectories:
            print(
                f"[Warning] No valid {method_name} files "
                f"for train_seed={train_seed}"
            )
            continue

        common_length = min(
            len(x)
            for x in eval_trajectories
        )

        eval_matrix = np.stack(
            [
                x[:common_length]
                for x in eval_trajectories
            ],
            axis=0,
        )

        # First average eval seeds within this train seed.
        train_seed_mean = np.nanmean(
            eval_matrix,
            axis=0,
        )

        trajectory_per_train_seed.append(
            train_seed_mean
        )
        successful_train_seeds.append(
            train_seed
        )

        print(
            f"[{method_name} kappa={kappa:.3f}] "
            f"train_seed={train_seed}, "
            f"eval_seeds={successful_eval_seeds}, "
            f"overall ON={np.mean(train_seed_mean):.4f}"
        )

    if not trajectory_per_train_seed:
        raise RuntimeError(
            f"No raw results found for {method_name}: "
            f"{result_root}"
        )

    common_length = min(
        len(x)
        for x in trajectory_per_train_seed
    )

    trajectory_per_train_seed = np.stack(
        [
            x[:common_length]
            for x in trajectory_per_train_seed
        ],
        axis=0,
    )

    return calculate_seed_statistics(
        trajectory_per_seed=trajectory_per_train_seed,
        seed_ids=successful_train_seeds,
        loaded_key=(
            f"raw eval files; keys={sorted(loaded_keys)}; "
            "eval-seed mean per train seed"
        ),
    )


# ============================================================
# 7. Load DDPP/MaxSNR summary files
# ============================================================
def ensure_seed_time_shape(
    array,
    seed_ids=None,
):
    """
    Converts [T], [S,T], or [T,S] into [S,T].
    """
    array = np.asarray(
        array,
        dtype=float,
    )

    array = np.squeeze(array)

    if array.ndim == 1:
        return array[None, :]

    if array.ndim != 2:
        raise ValueError(
            "Expected [T], [S,T], or [T,S], "
            f"but got {array.shape}"
        )

    number_of_seeds = (
        len(seed_ids)
        if seed_ids is not None
        else 0
    )

    if number_of_seeds > 0:
        if array.shape[0] == number_of_seeds:
            return array
        if array.shape[1] == number_of_seeds:
            return array.T

    # The time dimension is normally much longer.
    if array.shape[0] <= array.shape[1]:
        return array

    return array.T


def load_summary_on_trajectory(
    npz_path,
    max_steps,
    smooth_window,
):
    """
    Loads DDPP/MaxSNR multi-seed summary files.

    Preferred per-seed keys:
        bs_on_ratio_trajectory_per_seed
        on_ratio_trajectory_per_seed
        bs_on_trajectory_per_seed
        mean_bs_on_ratio_per_seed

    Also supports stored mean/std trajectory pairs.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"File not found: {npz_path}"
        )

    with np.load(npz_path, allow_pickle=True) as data:
        print(f"\n[{os.path.basename(npz_path)}]")
        print(f"Available keys: {data.files}")

        seed_ids = np.asarray(
            data.get("eval_seeds", []),
            dtype=int,
        ).reshape(-1)

        per_seed_keys = [
            "bs_on_ratio_trajectory_per_seed",
            "on_ratio_trajectory_per_seed",
            "bs_on_trajectory_per_seed",
            "mean_bs_on_ratio_per_seed",
            "bs_on_ratio_per_seed",
        ]

        for key in per_seed_keys:
            if key not in data.files:
                continue

            per_seed = ensure_seed_time_shape(
                data[key],
                seed_ids=seed_ids,
            )

            T = min(
                int(max_steps),
                int(per_seed.shape[1]),
            )

            per_seed = per_seed[:, :T]

            per_seed = np.stack(
                [
                    moving_average(x, smooth_window)
                    for x in per_seed
                ],
                axis=0,
            )

            return calculate_seed_statistics(
                trajectory_per_seed=per_seed,
                seed_ids=(
                    seed_ids
                    if seed_ids.size > 0
                    else np.arange(per_seed.shape[0])
                ),
                loaded_key=key,
            )

        mean_keys = [
            "bs_on_ratio_trajectory_mean",
            "on_ratio_trajectory_mean",
            "bs_on_trajectory_mean",
            "mean_bs_on_ratio_mean",
        ]

        std_keys = [
            "bs_on_ratio_trajectory_std",
            "on_ratio_trajectory_std",
            "bs_on_trajectory_std",
            "mean_bs_on_ratio_std",
        ]

        mean_key = next(
            (
                key
                for key in mean_keys
                if key in data.files
            ),
            None,
        )

        std_key = next(
            (
                key
                for key in std_keys
                if key in data.files
            ),
            None,
        )

        if mean_key is None or std_key is None:
            raise KeyError(
                "Could not find an ON-ratio trajectory key.\n"
                f"Available keys: {data.files}\n"
                "Recommended key:\n"
                "  bs_on_ratio_trajectory_per_seed [S,T]"
            )

        mean = np.asarray(
            data[mean_key],
            dtype=float,
        ).reshape(-1)

        std = np.asarray(
            data[std_key],
            dtype=float,
        ).reshape(-1)

        T = min(
            int(max_steps),
            len(mean),
            len(std),
        )

        mean = moving_average(
            mean[:T],
            smooth_window,
        )

        # Approximate smoothing of a saved std trajectory.
        std = moving_average(
            std[:T],
            smooth_window,
        )

        return {
            "mean": mean,
            "std": std,
            "var": std ** 2,
            "per_seed": None,
            "seed_ids": seed_ids,
            "loaded_key": f"{mean_key}, {std_key}",
        }


# ============================================================
# 8. Load all methods
# ============================================================
trajectory_results = {}

# DDPP and MaxSNR
for algorithm, npz_path in SUMMARY_NPZ_FILES.items():
    try:
        result = load_summary_on_trajectory(
            npz_path=npz_path,
            max_steps=MAX_STEPS,
            smooth_window=SMOOTH_WINDOW,
        )

        trajectory_results[algorithm] = result

        print(
            f"[{algorithm}] "
            f"key={result['loaded_key']}, "
            f"trajectory length={len(result['mean'])}, "
            f"overall ON={np.mean(result['mean']):.4f}"
        )

    except Exception as error:
        print(f"[WARNING] {algorithm}: {error}")


# PF-HAPPO and Jensen-HAPPO
for algorithm, config in BASELINE_RAW_CONFIGS.items():
    try:
        result = load_raw_multiseed_on_trajectory(
            result_root=config["result_root"],
            folder_prefix=config["folder_prefix"],
            kappa=KAPPA,
            train_seeds=TRAIN_SEEDS,
            eval_seeds=EVAL_SEEDS,
            max_steps=MAX_STEPS,
            smooth_window=SMOOTH_WINDOW,
            method_name=algorithm,
        )

        trajectory_results[algorithm] = result

        print(
            f"[{algorithm}] "
            f"key={result['loaded_key']}, "
            f"train seeds={result['per_seed'].shape[0]}, "
            f"trajectory length={len(result['mean'])}, "
            f"overall ON={np.mean(result['mean']):.4f}"
        )

    except Exception as error:
        print(f"[WARNING] {algorithm}: {error}")


# HeLyMARL
try:
    result = load_raw_multiseed_on_trajectory(
        result_root=HELYMARL_RAW_ROOT,
        folder_prefix=HELYMARL_FOLDER_PREFIX,
        kappa=KAPPA,
        train_seeds=TRAIN_SEEDS,
        eval_seeds=EVAL_SEEDS,
        max_steps=MAX_STEPS,
        smooth_window=SMOOTH_WINDOW,
        method_name="HeLyMARL",
    )

    trajectory_results["HeLyMARL"] = result

    print(
        f"[HeLyMARL] "
        f"key={result['loaded_key']}, "
        f"train seeds={result['per_seed'].shape[0]}, "
        f"trajectory length={len(result['mean'])}, "
        f"overall ON={np.mean(result['mean']):.4f}"
    )

except Exception as error:
    print(f"[WARNING] HeLyMARL: {error}")


if not trajectory_results:
    raise RuntimeError(
        "No ON-ratio trajectories were loaded."
    )


# ============================================================
# 9. Plot styles
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
# 10. Figure
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

    lower_bound = np.clip(
        trajectory_mean
        - BAND_SCALE * trajectory_std,
        0.0,
        1.0,
    )

    upper_bound = np.clip(
        trajectory_mean
        + BAND_SCALE * trajectory_std,
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


# Target ON ratio
ax.axhline(
    y=TARGET_ON_RATIO,
    linestyle="--",
    linewidth=1.8,
    color="black",
    alpha=0.8,
    label=rf"Target $\eta={TARGET_ON_RATIO:g}$",
    zorder=2,
)


# ============================================================
# 11. Axes
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
# 12. Legend
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
# 13. Save
# ============================================================
png_path = os.path.join(
    SAVE_DIR,
    "mean_bs_on_ratio_5algorithms_kappa0015.png",
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved PNG: {png_path}")
