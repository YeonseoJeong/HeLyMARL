import os
import matplotlib
import matplotlib.font_manager as fm

fm._load_fontmanager(try_read_cache=False)

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 14,
    "ps.fonttype": 42,
})


# ============================================================
# Training constraint-gap plot
# ============================================================
def plot_train_constraint_gap(
    result_dir="results/results_kappa",
    save_dir="eval_compare_plots",
    kappas=(0.01, 0.02, 0.03),
    seeds=(0, 1, 2, 3, 4),
):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    colors = ["C0", "C2", "C3"]
    markers = ["o", "^", "s"]
    line_styles = ["-", "--", "-."]

    found_train = False
    max_episode = 0

    for i, kappa in enumerate(kappas):
        kappa_tag = f"{kappa:.2f}"

        gap_per_seed = []
        successful_seeds = []

        # ====================================================
        # 같은 kappa에 대해 seed 5개 불러오기
        # ====================================================
        for seed in seeds:
            train_path = os.path.join(
                result_dir,
                (
                    f"HeLyMARL_train_rewards_"
                    f"kappa_{kappa_tag}_"
                    f"seed_{seed}.npz"
                ),
            )

            if not os.path.exists(train_path):
                print(
                    f"[Warning] Train file not found: "
                    f"{train_path}"
                )
                continue

            with np.load(
                train_path,
                allow_pickle=True,
            ) as data:

                # trainer에서 gap을 직접 저장한 경우
                if "episode_handover_gap" in data.files:
                    episode_gap = np.asarray(
                        data["episode_handover_gap"],
                        dtype=np.float64,
                    ).reshape(-1)

                # episode HO ratio만 있으면 gap 계산
                elif "episode_handover_ratio" in data.files:
                    episode_ratio = np.asarray(
                        data["episode_handover_ratio"],
                        dtype=np.float64,
                    ).reshape(-1)

                    if "kappa" in data.files:
                        saved_kappa = float(
                            np.asarray(
                                data["kappa"]
                            ).reshape(-1)[0]
                        )
                    else:
                        saved_kappa = float(kappa)

                    episode_gap = (
                        episode_ratio
                        - saved_kappa
                    )

                else:
                    print(
                        "[Warning] Neither "
                        "'episode_handover_gap' nor "
                        "'episode_handover_ratio' found in "
                        f"{train_path}"
                    )
                    continue

            if episode_gap.size == 0:
                print(
                    f"[Warning] Empty gap array: "
                    f"{train_path}"
                )
                continue

            gap_per_seed.append(episode_gap)
            successful_seeds.append(seed)

        if len(gap_per_seed) == 0:
            print(
                f"[Warning] No valid training seeds "
                f"for kappa={kappa_tag}"
            )
            continue

        # ====================================================
        # 모든 seed의 episode 길이 통일
        # ====================================================
        common_length = min(
            len(gap)
            for gap in gap_per_seed
        )

        if any(
            len(gap) != common_length
            for gap in gap_per_seed
        ):
            print(
                f"[Warning] Episode lengths differ for "
                f"kappa={kappa_tag}. "
                f"Truncated to {common_length} episodes."
            )

        gap_mat = np.stack(
            [
                gap[:common_length]
                for gap in gap_per_seed
            ],
            axis=0,
        )

        # gap_mat shape:
        # [number of seeds, number of episodes]
        ddof = (
            1
            if len(successful_seeds) > 1
            else 0
        )

        gap_mean = np.mean(
            gap_mat,
            axis=0,
        )

        gap_std = np.std(
            gap_mat,
            axis=0,
            ddof=ddof,
        )

        episodes = np.arange(
            1,
            common_length + 1,
        )

        max_episode = max(
            max_episode,
            common_length,
        )

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = line_styles[
            i % len(line_styles)
        ]

        # ====================================================
        # Seed 평균 gap
        # ====================================================
        ax.plot(
            episodes,
            gap_mean,
            linewidth=2.5,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            markevery=1,
            color=color,
            label=rf"$\kappa={kappa:.2f}$",
            zorder=3,
        )

        # ====================================================
        # 평균 ± 표준편차
        # ====================================================
        ax.fill_between(
            episodes,
            gap_mean - gap_std,
            gap_mean + gap_std,
            color=color,
            alpha=0.05,
            linewidth=0,
            zorder=2,
        )

        print(
            f"\n[Kappa={kappa:.2f}]"
        )
        print(
            f"  Successful seeds: "
            f"{successful_seeds}"
        )
        print(
            f"  Final gap: "
            f"{gap_mean[-1]:+.6f} "
            f"± {gap_std[-1]:.6f}"
        )

        found_train = True

    # ========================================================
    # Gap의 constraint boundary
    #
    # gap = HO ratio - kappa
    # gap <= 0이면 constraint 만족
    # ========================================================
    ax.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Constraint boundary",
        zorder=1,
    )

    ax.set_xlabel(
        "Training Episode"
    )

    ax.set_ylabel(
        r"Handover Constraint Gap "
        r"$\bar{h}^{(k)}-\kappa$"
    )

    if found_train:
        ax.set_xlim(
            1,
            max_episode,
        )

        ax.set_xticks(
            np.arange(
                1,
                max_episode + 1,
                1,
            )
        )

        ax.legend(
            loc="best",
            fontsize=14,
            markerscale=0.85,
            handlelength=2.0,
            labelspacing=0.4,
            borderpad=0.5,
            frameon=True,
        )

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    png_path = os.path.join(
        save_dir,
        "train_handover_constraint_gap_5seeds.png",
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\n✅ Saved: {png_path}")

def plot_train_episode_handover_ratio(
    result_dir="results/results_kappa",
    save_dir="eval_compare_plots",
    kappas=(0.01, 0.02, 0.03),
    seeds=(0, 1, 2, 3, 4),
):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    colors = ["C0", "C2", "C3"]
    markers = ["o", "^", "s"]
    line_styles = ["-", "--", "-."]

    found_train = False
    max_episode = 0

    for i, kappa in enumerate(kappas):
        kappa_tag = f"{kappa:.2f}"

        episode_ratio_per_seed = []
        successful_seeds = []

        # ====================================================
        # 해당 kappa의 seed 5개 파일 로드
        # ====================================================
        for seed in seeds:
            train_path = os.path.join(
                result_dir,
                (
                    f"HeLyMARL_train_rewards_"
                    f"kappa_{kappa_tag}_"
                    f"seed_{seed}.npz"
                ),
            )

            if not os.path.exists(train_path):
                print(
                    f"[Warning] Train file not found: "
                    f"{train_path}"
                )
                continue

            with np.load(
                train_path,
                allow_pickle=True,
            ) as data:
                if (
                    "episode_handover_ratio"
                    not in data.files
                ):
                    print(
                        "[Warning] "
                        "'episode_handover_ratio' "
                        f"not found in {train_path}"
                    )
                    continue

                episode_ratio = np.asarray(
                    data["episode_handover_ratio"],
                    dtype=np.float64,
                ).reshape(-1)

            if episode_ratio.size == 0:
                print(
                    f"[Warning] Empty episode ratio: "
                    f"{train_path}"
                )
                continue

            episode_ratio_per_seed.append(
                episode_ratio
            )
            successful_seeds.append(seed)

        if len(episode_ratio_per_seed) == 0:
            print(
                f"[Warning] No valid seeds for "
                f"kappa={kappa_tag}"
            )
            continue

        # ====================================================
        # 모든 seed 길이를 공통 episode 수로 통일
        # ====================================================
        common_length = min(
            len(x)
            for x in episode_ratio_per_seed
        )

        if any(
            len(x) != common_length
            for x in episode_ratio_per_seed
        ):
            print(
                f"[Warning] Different episode lengths for "
                f"kappa={kappa_tag}; "
                f"truncated to {common_length}"
            )

        ratio_mat = np.stack(
            [
                x[:common_length]
                for x in episode_ratio_per_seed
            ],
            axis=0,
        )

        # shape: [num_seeds, num_episodes]
        ddof = (
            1
            if len(successful_seeds) > 1
            else 0
        )

        ratio_mean = np.mean(
            ratio_mat,
            axis=0,
        )

        ratio_std = np.std(
            ratio_mat,
            axis=0,
            ddof=ddof,
        )

        ratio_variance = np.var(
            ratio_mat,
            axis=0,
            ddof=ddof,
        )

        episodes = np.arange(
            1,
            common_length + 1,
        )

        max_episode = max(
            max_episode,
            common_length,
        )

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = line_styles[
            i % len(line_styles)
        ]

        # ====================================================
        # Seed 평균
        # ====================================================
        ax.plot(
            episodes,
            ratio_mean,
            linewidth=2.5,
            linestyle=linestyle,
            marker=marker,
            markersize=7,
            markevery=1,
            color=color,
            label=rf"$\kappa={kappa:.2f}$",
            zorder=3,
        )

        # ====================================================
        # 평균 ± 표준편차 음영
        # ====================================================
        ax.fill_between(
            episodes,
            ratio_mean - ratio_std,
            ratio_mean + ratio_std,
            color=color,
            alpha=0.05,
            linewidth=0,
            zorder=2,
        )

        # ====================================================
        # 해당 kappa constraint
        # ====================================================
        ax.axhline(
            y=kappa,
            color=color,
            linestyle=":",
            linewidth=1.8,
            alpha=0.9,
            zorder=1,
        )

        print(
            f"\n[Kappa={kappa:.2f}]"
        )
        print(
            f"  Successful seeds: "
            f"{successful_seeds}"
        )
        print(
            f"  Final ratio: "
            f"{ratio_mean[-1]:.6f} "
            f"± {ratio_std[-1]:.6f}"
        )
        print(
            f"  Final variance: "
            f"{ratio_variance[-1]:.8f}"
        )
        print(
            f"  Final gap: "
            f"{ratio_mean[-1] - kappa:+.6f}"
        )

        found_train = True

    ax.set_xlabel(
        "Training Episode"
    )

    ax.set_ylabel(
        "Episode-Average Handover Ratio"
    )

    if found_train:
        ax.set_xlim(
            1,
            max_episode,
        )

        ax.set_xticks(
            np.arange(
                1,
                max_episode + 1,
                1,
            )
        )

        ax.legend(
            loc="upper right",
            fontsize=14,
            markerscale=0.85,
            handlelength=2.0,
            labelspacing=0.4,
            borderpad=0.5,
            frameon=True,
        )

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    png_path = os.path.join(
        save_dir,
        "train_episode_handover_ratio_kappa_5seeds.png",
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\n✅ Saved: {png_path}")


# ============================================================
# Final hard-evaluation USER handover ratio bar plot
#
# Bar height : mean across users
# Error bar  : minimum ~ maximum across users
# ============================================================
def plot_eval_bar(
    result_dir="results/results_kappa",
    save_dir="eval_compare_plots",
    kappas=(0.01, 0.02, 0.03),
):
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 6.0)
    )

    colors = [
        "C0",
        "C2",
        "C3",
    ]

    eval_means = []
    eval_mins = []
    eval_maxs = []
    eval_budgets = []
    eval_labels = []

    for kappa in kappas:
        kappa_tag = f"{kappa:.2f}"

        eval_path = os.path.join(
            result_dir,
            (
                f"HeLyMARL_eval_hard_"
                f"kappa_{kappa_tag}.npz"
            ),
        )

        if not os.path.exists(eval_path):
            print(
                f"[Warning] Eval file not found: "
                f"{eval_path}"
            )
            continue

        with np.load(
            eval_path,
            allow_pickle=True,
        ) as data:

            # ------------------------------------------------
            # Evaluation horizon T 확인
            # ------------------------------------------------
            if "slot_rates" in data.files:
                slot_rates = np.asarray(
                    data["slot_rates"]
                )
                eval_steps = int(
                    slot_rates.shape[0]
                )

            elif "handover_ratio" in data.files:
                # handover_ratio가 slot별 기록이면 길이가 T
                eval_steps = int(
                    np.asarray(
                        data["handover_ratio"]
                    ).reshape(-1).size
                )

            else:
                raise KeyError(
                    f"Cannot determine evaluation horizon "
                    f"from {eval_path}. "
                    "Save 'slot_rates' or eval_steps."
                )

            # ------------------------------------------------
            # 사용자별 handover 횟수 불러오기
            #
            # handover_count가 반드시 shape=(n_users,)여야 함.
            # scalar 또는 shape=(T,)라면 사용자별 결과가 아님.
            # ------------------------------------------------
            if "handover_count" not in data.files:
                raise KeyError(
                    f"'handover_count' not found in "
                    f"{eval_path}. "
                    "Save the final handover count of each "
                    "user as shape=(n_users,)."
                )

            # 사용자별 HO ratio를 직접 저장한 경우
            if "handover_ratio_per_user" in data.files:
                user_ho_ratio = np.asarray(
                    data["handover_ratio_per_user"],
                    dtype=np.float64,
                ).reshape(-1)

            # 사용자별 HO count만 저장한 경우
            elif "handover_count_per_user" in data.files:
                handover_count_per_user = np.asarray(
                    data["handover_count_per_user"],
                    dtype=np.float64,
                ).reshape(-1)

                denominator = max(
                    eval_steps - 1,
                    1,
                )

                user_ho_ratio = (
                    handover_count_per_user
                    / denominator
                )

            else:
                raise KeyError(
                    f"Per-user handover result not found in {eval_path}. "
                    f"Available keys: {data.files}"
                )

            user_ho_ratio = user_ho_ratio[
                np.isfinite(user_ho_ratio)
            ]

            mean_val = float(np.mean(user_ho_ratio))
            min_val = float(np.min(user_ho_ratio))
            max_val = float(np.max(user_ho_ratio))

            if (
                "handover_budget_ratio"
                in data.files
            ):
                budget = float(
                    np.asarray(
                        data[
                            "handover_budget_ratio"
                        ]
                    ).reshape(-1)[0]
                )

            elif "kappa" in data.files:
                budget = float(
                    np.asarray(
                        data["kappa"]
                    ).reshape(-1)[0]
                )

            else:
                budget = float(kappa)

        eval_means.append(mean_val)
        eval_mins.append(min_val)
        eval_maxs.append(max_val)
        eval_budgets.append(budget)

        eval_labels.append(
            f"$\kappa={budget:.2f}$"
        )

        print(
            f"[kappa={budget:.2f}]"
        )
        print(
            f"  Number of users: "
            f"{user_ho_ratio.size}"
        )
        print(
            f"  User HO ratio min : "
            f"{min_val:.6f}"
        )
        print(
            f"  User HO ratio mean: "
            f"{mean_val:.6f}"
        )
        print(
            f"  User HO ratio max : "
            f"{max_val:.6f}"
        )

    if len(eval_means) == 0:
        plt.close()
        print(
            "[Warning] No evaluation files found."
        )
        return

    eval_means = np.asarray(
        eval_means,
        dtype=np.float64,
    )
    eval_mins = np.asarray(
        eval_mins,
        dtype=np.float64,
    )
    eval_maxs = np.asarray(
        eval_maxs,
        dtype=np.float64,
    )
    eval_budgets = np.asarray(
        eval_budgets,
        dtype=np.float64,
    )

    x_bar = np.arange(
        len(eval_means)
    )

    bar_colors = [
        colors[
            i % len(colors)
        ]
        for i in range(
            len(eval_means)
        )
    ]

    bars = ax.bar(
        x_bar,
        eval_means,
        color=bar_colors,
        alpha=0.85,
        width=0.6,
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )

    # 사용자별 최솟값~최댓값을 비대칭 error bar로 표시
    yerr = np.vstack(
        [
            eval_means - eval_mins,
            eval_maxs - eval_means,
        ]
    )

    ax.errorbar(
        x_bar,
        eval_means,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.8,
        capsize=7,
        capthick=1.8,
        zorder=4,
        label="User Min–Max",
    )

    # 각 kappa의 constraint 표시
    ax.scatter(
        x_bar,
        eval_budgets,
        marker="_",
        s=500,
        color="black",
        linewidths=2.0,
        label="HO constraint",
        zorder=5,
    )

    ax.set_xticks(
        x_bar
    )

    ax.set_xticklabels(
        eval_labels
    )

    ax.set_ylabel(
        "Per-User Handover Ratio"
    )

    max_y = max(
        float(np.max(eval_maxs)),
        float(np.max(eval_budgets)),
    )

    ax.set_ylim(
        0,
        max_y * 1.25,
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
        zorder=0,
    )

    ax.legend(
        loc="best",
    )

    text_offset = max_y * 0.03

    for bar, mean_val in zip(
        bars,
        eval_means,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            mean_val + text_offset,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=17,
            zorder=6,
        )

    plt.tight_layout()

    png_path = os.path.join(
        save_dir,
        "eval_user_handover_bar_kappa_minmax.png",
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"✅ Saved: {png_path}"
    )
if __name__ == "__main__":
    kappas = (
        0.01,
        0.02,
        0.03,
    )

    seeds = (
        0,
        1,
        2,
        3,
        4,
    )
    
    plot_train_episode_handover_ratio(
        result_dir="results/results_kappa",
        save_dir="eval_compare_plots",
        kappas=kappas,
        seeds=seeds,
    )

    plot_train_constraint_gap(
        result_dir="results/results_kappa",
        save_dir="eval_compare_plots",
        kappas=kappas,
        seeds=seeds,
    )

    plot_eval_bar(
        result_dir="results/results_kappa",
        save_dir="eval_compare_plots",
        kappas=kappas,
    )