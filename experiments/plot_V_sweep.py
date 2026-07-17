import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================
RESULT_DIR = "results/results_V_sweep"

TRAIN_V = 5.0

SAVE_FIG = True
SHOW_FIG = True


# ============================================================
# Utility functions
# ============================================================
def get_mean_value(data, candidate_keys):
    """
    candidate_keys 중 실제로 존재하는 첫 번째 key의 평균값을 반환합니다.
    """
    for key in candidate_keys:
        if key not in data:
            continue

        values = np.asarray(data[key], dtype=float)

        if values.size == 0:
            continue

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            continue

        return float(np.mean(finite_values))

    return np.nan


def get_throughput(data):
    return get_mean_value(
        data,
        [
            "episode_throughput_mean",
            "throughput",
            "mean_throughput",
        ],
    )


def get_fairness(data):
    return get_mean_value(
        data,
        [
            "fairness",
            "mean_fairness",
            "jfi",
        ],
    )


def get_on_ratio(data):
    return get_mean_value(
        data,
        [
            "bs_on_ratio_mean",
            "mean_on_ratio",
            "on_ratio",
        ],
    )


def get_handover_ratio(data):
    return get_mean_value(
        data,
        [
            "handover_ratio_mean",
            "mean_handover_ratio",
            "handover_ratio",
        ],
    )


def extract_eval_v(file_path):
    """
    파일명에서 evaluation V 값을 추출합니다.

    예:
    HeLyMARL_eval_hard_V_5.0.npz
    -> 5.0

    HeLyMARL_eval_hard_V_50.npz
    -> 50.0
    """
    file_name = os.path.basename(file_path)

    match = re.search(
        r"HeLyMARL_eval_hard_V_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.npz$",
        file_name,
    )

    if match is None:
        return None

    return float(match.group(1))


def load_v_sweep_results(result_dir):
    """
    다음 형식의 모든 평가 파일을 불러옵니다.

    results/results_V_sweep/
        HeLyMARL_eval_hard_V_5.0.npz
        HeLyMARL_eval_hard_V_10.0.npz
        HeLyMARL_eval_hard_V_20.0.npz
        ...
    """
    pattern = os.path.join(
        result_dir,
        "HeLyMARL_eval_hard_V_*.npz",
    )

    eval_files = glob.glob(pattern)

    if not eval_files:
        raise FileNotFoundError(
            "V-sweep evaluation 파일을 찾을 수 없습니다.\n"
            f"검색 경로: {pattern}"
        )

    results = []

    for file_path in eval_files:
        eval_v = extract_eval_v(file_path)

        if eval_v is None:
            print(
                "[Warning] 파일명에서 V 값을 읽지 못해 제외합니다: "
                f"{file_path}"
            )
            continue

        with np.load(file_path, allow_pickle=True) as data:
            result = {
                "eval_v": eval_v,
                "throughput": get_throughput(data),
                "fairness": get_fairness(data),
                "on_ratio": get_on_ratio(data),
                "handover_ratio": get_handover_ratio(data),
                "file_path": file_path,
            }

        results.append(result)

    if not results:
        raise RuntimeError(
            "유효한 V-sweep evaluation 결과가 없습니다."
        )

    results.sort(key=lambda result: result["eval_v"])

    return results


# ============================================================
# Print summary
# ============================================================
def print_summary(results, train_v):
    print("\n" + "=" * 92)
    print(
        f"HeLyMARL V sensitivity summary | "
        f"Model trained with V = {train_v:g}"
    )
    print("=" * 92)

    print(
        f"{'Eval V':>10} | "
        f"{'Throughput':>12} | "
        f"{'JFI':>10} | "
        f"{'ON ratio':>12} | "
        f"{'HO ratio':>12}"
    )

    print("-" * 92)

    for result in results:
        print(
            f"{result['eval_v']:>10.2f} | "
            f"{result['throughput']:>12.4f} | "
            f"{result['fairness']:>10.4f} | "
            f"{result['on_ratio']:>12.4f} | "
            f"{result['handover_ratio']:>12.4f}"
        )

    print("=" * 92)


# ============================================================
# Save integrated summary
# ============================================================
def save_summary_npz(results, result_dir, train_v):
    summary_path = os.path.join(
        result_dir,
        f"HeLyMARL_V_sweep_summary_trainV_{train_v:g}.npz",
    )

    np.savez(
        summary_path,
        train_V=np.asarray(train_v, dtype=float),

        eval_V=np.asarray(
            [result["eval_v"] for result in results],
            dtype=float,
        ),

        throughput=np.asarray(
            [result["throughput"] for result in results],
            dtype=float,
        ),

        fairness=np.asarray(
            [result["fairness"] for result in results],
            dtype=float,
        ),

        on_ratio=np.asarray(
            [result["on_ratio"] for result in results],
            dtype=float,
        ),

        handover_ratio=np.asarray(
            [result["handover_ratio"] for result in results],
            dtype=float,
        ),
    )

    print(f"Summary npz saved to: {summary_path}")

    return summary_path


# ============================================================
# Plot 1: Throughput and fairness versus evaluation V
# ============================================================
def plot_metrics_vs_v(results, result_dir, train_v):
    eval_v = np.asarray(
        [result["eval_v"] for result in results],
        dtype=float,
    )

    throughput = np.asarray(
        [result["throughput"] for result in results],
        dtype=float,
    )

    fairness = np.asarray(
        [result["fairness"] for result in results],
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
    )

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------
    axes[0].plot(
        eval_v,
        throughput,
        marker="o",
        linewidth=2.0,
        markersize=7,
    )

    for x_value, y_value in zip(eval_v, throughput):
        if np.isfinite(y_value):
            axes[0].annotate(
                f"{y_value:.4f}",
                xy=(x_value, y_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

    axes[0].set_xlabel("Evaluation V", fontsize=12)
    axes[0].set_ylabel("Mean Throughput", fontsize=12)
    axes[0].set_title(
        "Throughput versus Evaluation V",
        fontsize=13,
    )
    axes[0].grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Fairness
    # --------------------------------------------------------
    axes[1].plot(
        eval_v,
        fairness,
        marker="s",
        linewidth=2.0,
        markersize=7,
    )

    for x_value, y_value in zip(eval_v, fairness):
        if np.isfinite(y_value):
            axes[1].annotate(
                f"{y_value:.4f}",
                xy=(x_value, y_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

    axes[1].set_xlabel("Evaluation V", fontsize=12)
    axes[1].set_ylabel("Jain's Fairness Index", fontsize=12)
    axes[1].set_title(
        "Fairness versus Evaluation V",
        fontsize=13,
    )
    axes[1].grid(True, alpha=0.3)

    finite_fairness = fairness[np.isfinite(fairness)]

    if finite_fairness.size > 0:
        fairness_min = np.min(finite_fairness)
        fairness_max = np.max(finite_fairness)

        fairness_margin = max(
            0.02,
            (fairness_max - fairness_min) * 0.25,
        )

        axes[1].set_ylim(
            max(0.0, fairness_min - fairness_margin),
            min(1.0, fairness_max + fairness_margin),
        )

    fig.suptitle(
        f"HeLyMARL V Sensitivity: Model Trained with V={train_v:g}",
        fontsize=14,
    )

    fig.tight_layout()

    save_path = os.path.join(
        result_dir,
        f"HeLyMARL_metrics_vs_V_trainV_{train_v:g}.png",
    )

    if SAVE_FIG:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Metric figure saved to: {save_path}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Plot 2: Throughput-Fairness tradeoff
# ============================================================
def plot_throughput_fairness_tradeoff(
    results,
    result_dir,
    train_v,
):
    throughput = np.asarray(
        [result["throughput"] for result in results],
        dtype=float,
    )

    fairness = np.asarray(
        [result["fairness"] for result in results],
        dtype=float,
    )

    eval_v = np.asarray(
        [result["eval_v"] for result in results],
        dtype=float,
    )

    valid_mask = (
        np.isfinite(throughput)
        & np.isfinite(fairness)
        & np.isfinite(eval_v)
    )

    throughput = throughput[valid_mask]
    fairness = fairness[valid_mask]
    eval_v = eval_v[valid_mask]

    if throughput.size == 0:
        print(
            "[Warning] Throughput-Fairness 그래프에 사용할 "
            "유효한 값이 없습니다."
        )
        return

    fig, ax = plt.subplots(figsize=(7.4, 6.0))

    ax.plot(
        throughput,
        fairness,
        marker="o",
        linewidth=2.0,
        markersize=8,
    )

    for x_value, y_value, v_value in zip(
        throughput,
        fairness,
        eval_v,
    ):
        ax.annotate(
            f"V={v_value:g}",
            xy=(x_value, y_value),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel("Mean Throughput", fontsize=12)
    ax.set_ylabel("Jain's Fairness Index", fontsize=12)

    ax.set_title(
        (
            "Throughput–Fairness Tradeoff\n"
            f"Model Trained with V={train_v:g}"
        ),
        fontsize=13,
    )

    ax.grid(True, alpha=0.3)

    finite_fairness = fairness[np.isfinite(fairness)]

    if finite_fairness.size > 0:
        fairness_min = np.min(finite_fairness)
        fairness_max = np.max(finite_fairness)

        fairness_margin = max(
            0.02,
            (fairness_max - fairness_min) * 0.25,
        )

        ax.set_ylim(
            max(0.0, fairness_min - fairness_margin),
            min(1.0, fairness_max + fairness_margin),
        )

    fig.tight_layout()

    save_path = os.path.join(
        result_dir,
        (
            "HeLyMARL_throughput_fairness_"
            f"tradeoff_trainV_{train_v:g}.png"
        ),
    )

    if SAVE_FIG:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Tradeoff figure saved to: {save_path}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train_npz_path = os.path.join(
        RESULT_DIR,
        f"HeLyMARL_train_rewards_V_{TRAIN_V}.npz",
    )

    model_path = os.path.join(
        RESULT_DIR,
        f"HeLyMARL_model_V_{TRAIN_V}.pt",
    )

    print(f"Training result path: {train_npz_path}")
    print(f"Model path:           {model_path}")

    results = load_v_sweep_results(
        result_dir=RESULT_DIR,
    )

    print_summary(
        results=results,
        train_v=TRAIN_V,
    )

    save_summary_npz(
        results=results,
        result_dir=RESULT_DIR,
        train_v=TRAIN_V,
    )

    plot_metrics_vs_v(
        results=results,
        result_dir=RESULT_DIR,
        train_v=TRAIN_V,
    )

    plot_throughput_fairness_tradeoff(
        results=results,
        result_dir=RESULT_DIR,
        train_v=TRAIN_V,
    )

    print("\n✅ V-sweep plotting completed!\n")