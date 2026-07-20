import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter


# ============================================================
# Plot settings
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

ALGORITHM_COLORS = {
    "DDPP": "#1f77b4",
    "MaxSNR": "#ff7f0e",
    "PF-HAPPO": "#2ca02c",
    "Jensen-HAPPO": "#d62728",
    "HeLyMARL": "#9467bd",
}

def load_policy_improvement_objective(npz_path):
    """
    Training NPZ에서 학습 전 policy와
    episode 내부 checkpoint policy의 evaluation objective를 불러온다.

    Required keys
    -------------
    policy_eval_steps
    policy_eval_objective_mean

    Optional keys
    -------------
    policy_eval_objective_std
    policy_eval_episodes
    policy_eval_updates
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"NPZ file not found: {npz_path}"
        )

    with np.load(npz_path, allow_pickle=True) as data:
        print(f"\n[{os.path.basename(npz_path)}]")
        print("Available keys:")
        print(data.files)

        required_keys = [
            "policy_eval_steps",
            "policy_eval_objective_mean",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in data
        ]

        if missing_keys:
            raise KeyError(
                f"Missing keys in {npz_path}: {missing_keys}\n"
                "새 policy improvement trainer로 학습했는지 확인해야 합니다."
            )

        train_steps = np.asarray(
            data["policy_eval_steps"],
            dtype=np.int64,
        ).reshape(-1)

        objective_mean = np.asarray(
            data["policy_eval_objective_mean"],
            dtype=np.float64,
        ).reshape(-1)

        # ----------------------------------------------------
        # Gbps-based objective -> bps-based objective
        # 필요한 경우 아래 주석 해제
        # ----------------------------------------------------
        # objective_mean = (
        #     objective_mean
        #     + 20 * np.log(1e9)
        # )

        if "policy_eval_objective_std" in data:
            objective_std = np.asarray(
                data["policy_eval_objective_std"],
                dtype=np.float64,
            ).reshape(-1)
        else:
            objective_std = np.zeros_like(
                objective_mean,
                dtype=np.float64,
            )

        if "policy_eval_episodes" in data:
            episodes = np.asarray(
                data["policy_eval_episodes"],
                dtype=np.int32,
            ).reshape(-1)
        else:
            episodes = np.full(
                len(train_steps),
                -1,
                dtype=np.int32,
            )

        if "policy_eval_updates" in data:
            updates = np.asarray(
                data["policy_eval_updates"],
                dtype=np.int32,
            ).reshape(-1)
        else:
            updates = np.full(
                len(train_steps),
                -1,
                dtype=np.int32,
            )

    lengths = [
        len(train_steps),
        len(objective_mean),
        len(objective_std),
        len(episodes),
        len(updates),
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            "policy evaluation 관련 배열의 길이가 서로 다릅니다."
        )

    valid_mask = (
        np.isfinite(train_steps)
        & np.isfinite(objective_mean)
        & np.isfinite(objective_std)
    )

    train_steps = train_steps[valid_mask]
    objective_mean = objective_mean[valid_mask]
    objective_std = objective_std[valid_mask]
    episodes = episodes[valid_mask]
    updates = updates[valid_mask]

    # 혹시 저장 순서가 뒤섞였을 경우 step 순으로 정렬
    order = np.argsort(train_steps)

    return (
        train_steps[order],
        objective_mean[order],
        objective_std[order],
        episodes[order],
        updates[order],
    )


def format_k_tick(value, position):
    """
    x축 값이 이미 1000으로 나누어진 경우,
    0, 20k, 40k, ... 형식으로 표시한다.
    """
    if np.isclose(value, 0.0):
        return "0"

    if np.isclose(value, round(value)):
        return f"{int(round(value))}k"

    return f"{value:g}k"


def plot_policy_improvement_objective(
    npz_paths,
    save_path,
    show_std=True,
    steps_per_episode=10000,
    x_scale=1000.0,
    marker_every=2,
    show_episode_boundaries=True,
):
    """
    checkpoint별 evaluation objective를
    전체 training environment step에 따라 표시한다.

    Parameters
    ----------
    npz_paths : dict
        {
            "PF-HAPPO": "path/to/file.npz",
            "Jensen-HAPPO": "path/to/file.npz",
            "HeLyMARL": "path/to/file.npz",
        }

    save_path : str
        저장할 그림 경로

    show_std : bool
        evaluation seed 간 objective std 음영 표시 여부

    steps_per_episode : int
        episode 경계 표시용 step 수

    x_scale : float
        x축 scaling 값.
        1000이면 10000 step이 x축에서 10으로 표시됨.

    marker_every : int
        몇 개 checkpoint마다 marker를 표시할지 결정

    show_episode_boundaries : bool
        episode 경계 수직선 표시 여부
    """
    output_dir = (
        os.path.dirname(save_path)
        if os.path.dirname(save_path)
        else "."
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 데이터 먼저 모두 로드
    # --------------------------------------------------------
    loaded_results = {}
    all_train_steps = []

    for algorithm, npz_path in npz_paths.items():
        try:
            (
                train_steps,
                objective_mean,
                objective_std,
                episodes,
                updates,
            ) = load_policy_improvement_objective(npz_path)

        except (FileNotFoundError, KeyError, ValueError) as error:
            print(f"[Warning] {algorithm}: {error}")
            continue

        if len(train_steps) == 0:
            print(
                f"[Warning] Empty objective data: {algorithm}"
            )
            continue

        x = (
            train_steps.astype(np.float64)
            / x_scale
        )

        loaded_results[algorithm] = {
            "train_steps": train_steps,
            "objective_mean": objective_mean,
            "objective_std": objective_std,
            "episodes": episodes,
            "updates": updates,
            "x": x,
        }

        all_train_steps.extend(
            train_steps.tolist()
        )

    if not loaded_results:
        raise RuntimeError(
            "그릴 수 있는 policy improvement objective 데이터가 없습니다."
        )

    max_train_step = int(
        np.max(all_train_steps)
    )

    max_x = (
        max_train_step
        / x_scale
    )

    # --------------------------------------------------------
    # Figure 생성
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(7.8, 5.3)
    )

    line_styles = {
        "PF-HAPPO": "-.",
        "Jensen-HAPPO": "--",
        "HeLyMARL": "-",
    }

    markers = {
        "PF-HAPPO": "^",
        "Jensen-HAPPO": "D",
        "HeLyMARL": "o",
    }

    for algorithm, result in loaded_results.items():
        train_steps = result["train_steps"]
        objective_mean = result["objective_mean"]
        objective_std = result["objective_std"]
        episodes = result["episodes"]
        updates = result["updates"]
        x = result["x"]

        is_helymarl = (
            algorithm == "HeLyMARL"
        )

        linewidth = (
            2.8
            if is_helymarl
            else 2.0
        )

        markersize = (
            5.0
            if is_helymarl
            else 4.2
        )

        zorder = (
            4
            if is_helymarl
            else 3
        )

        linestyle = line_styles.get(
            algorithm,
            "-",
        )

        marker = markers.get(
            algorithm,
            "o",
        )

        line_color = ALGORITHM_COLORS.get(
            algorithm,
            "black",
        )

        ax.plot(
            x,
            objective_mean,
            color=line_color,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_every,
            markersize=markersize,
            linewidth=linewidth,
            label=algorithm,
            zorder=zorder,
        )

        if (
            show_std
            and len(objective_std) > 0
            and np.any(objective_std > 0)
        ):
            lower = (
                objective_mean
                - objective_std
            )

            upper = (
                objective_mean
                + objective_std
            )

            ax.fill_between(
                x,
                lower,
                upper,
                color=line_color,
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )

        # ----------------------------------------------------
        # 저장된 수치 출력
        # ----------------------------------------------------
        print(f"\n{algorithm}")
        print("=" * 85)

        for step, ep, update, mean, std in zip(
            train_steps,
            episodes,
            updates,
            objective_mean,
            objective_std,
        ):
            print(
                f"Step {int(step):7d} | "
                f"Episode {int(ep):3d} | "
                f"Update {int(update):4d} | "
                f"Objective = {mean:.6f} | "
                f"Std = {std:.6f}"
            )

    # --------------------------------------------------------
    # Episode boundary 표시
    # --------------------------------------------------------
    if show_episode_boundaries:
        episode_boundaries = np.arange(
            steps_per_episode,
            max_train_step + 1,
            steps_per_episode,
        )

        for boundary in episode_boundaries:
            ax.axvline(
                boundary / x_scale,
                color="gray",
                linestyle=":",
                linewidth=0.8,
                alpha=0.30,
                zorder=0,
            )

    # --------------------------------------------------------
    # 축 설정
    # --------------------------------------------------------
    ax.set_xlim(
        left=0.0,
        right=max_x,
    )

    ax.set_xlabel(
        "Training Environment Steps"
    )

    ax.set_ylabel(
        r"Evaluation Objective Eq. (7)"
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(format_k_tick)
    )

    ax.grid(
        True,
        alpha=0.25,
        linestyle="--",
        linewidth=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        length=4.5,
        width=1.1,
    )

    ax.legend(
        loc="lower right",
        frameon=False,
        handlelength=2.8,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\n✅ Saved policy improvement plot:\n"
        f"   PNG: {save_path}"
    )


if __name__ == "__main__":

    NPZ_PATHS = {
        "PF-HAPPO": (
            "results/policy_improvement/pf/"
            "ConstrainedHAPPO_pf_policy_improvement_"
            "kappa_0.03.npz"
        ),

        "Jensen-HAPPO": (
            "results/policy_improvement/jensen/"
            "ConstrainedHAPPO_jensen_policy_improvement_"
            "kappa_0.03.npz"
        ),

        "HeLyMARL": (
            "results/policy_improvement/HeLyMARL/"
            "HeLyMARL_policy_improvement_lambda_0.0.npz"
        ),
    }

    SAVE_PATH = (
        "eval_compare_plots/"
        "evaluation_objective_learning_curve.png"
    )

    plot_policy_improvement_objective(
        npz_paths=NPZ_PATHS,
        save_path=SAVE_PATH,
        show_std=True,
        steps_per_episode=10000,
        x_scale=1000.0,

        # 2개 checkpoint마다 marker 표시
        marker_every=2,

        # 10k마다 episode 경계선 표시
        show_episode_boundaries=True,
    )