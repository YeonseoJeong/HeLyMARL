import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=1000):
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return x

    if window <= 1:
        return x

    if len(x) < window:
        window = len(x)

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def extract_kappa_from_filename(path):
    """
    Example:
    LyMARL_train_rewards_kappa_0.01.npz
    LyMARL_eval_hard_kappa_0.01.npz
    """
    fname = os.path.basename(path)

    m = re.search(r"kappa_([0-9]+(?:\.[0-9]+)?)(?=\.npz|_|$)", fname)
    if m is None:
        return None

    # 혹시 .npz의 마지막 점까지 잡히는 경우 방지
    kappa_str = m.group(1).replace(".npz", "")
    return float(kappa_str)


def load_npz_by_kappa(result_dir, lambda_E=15.0):
    train_pattern = os.path.join(
        result_dir,
        f"LyMARL_train_rewards_kappa_*.npz"
    )

    eval_pattern = os.path.join(
        result_dir,
        f"LyMARL_eval_hard_kappa_*.npz"
    )

    train_files = glob.glob(train_pattern)
    eval_files = glob.glob(eval_pattern)

    train_dict = {}
    eval_dict = {}

    for path in train_files:
        kappa = extract_kappa_from_filename(path)
        if kappa is not None:
            train_dict[kappa] = path

    for path in eval_files:
        kappa = extract_kappa_from_filename(path)
        if kappa is not None:
            eval_dict[kappa] = path

    kappas = sorted(set(train_dict.keys()) | set(eval_dict.keys()))

    return kappas, train_dict, eval_dict


def plot_train_eval_handover_by_kappa(
    result_dir="results_kappa",
    lambda_E=15.0,
    window=1000,
    save_dir=None,
):
    if save_dir is None:
        save_dir = os.path.join(result_dir, "plots")

    os.makedirs(save_dir, exist_ok=True)

    kappas, train_dict, eval_dict = load_npz_by_kappa(
        result_dir=result_dir,
        lambda_E=lambda_E
    )

    if len(kappas) == 0:
        print("[Warning] No kappa files found.")
        return

    for kappa in kappas:
        plt.figure(figsize=(8, 5))

        budget = kappa
        plotted = False

        # --------------------------------------------------
        # Train curve
        # --------------------------------------------------
        if kappa in train_dict:
            train_data = np.load(train_dict[kappa], allow_pickle=True)

            train_ho = train_data["handover_ratio"].astype(np.float32)
            train_ho_ma = moving_average(train_ho, window=window)

            if "handover_budget_ratio" in train_data:
                budget = float(train_data["handover_budget_ratio"][0])

            x_train = np.arange(len(train_ho_ma))
            plt.plot(
                x_train,
                train_ho_ma,
                label=f"Train (MA{window})",
                linewidth=2.0
            )
            plotted = True

        else:
            print(f"[Warning] Missing train file for kappa={kappa}")

        # --------------------------------------------------
        # Eval curve
        # --------------------------------------------------
        if kappa in eval_dict:
            eval_data = np.load(eval_dict[kappa], allow_pickle=True)

            eval_ho = eval_data["handover_ratio"].astype(np.float32)
            eval_ho_ma = moving_average(eval_ho, window=window)

            if "handover_budget_ratio" in eval_data:
                budget = float(eval_data["handover_budget_ratio"][0])

            x_eval = np.arange(len(eval_ho_ma))
            plt.plot(
                x_eval,
                eval_ho_ma,
                label=f"Eval Hard (MA{window})",
                linewidth=2.0
            )
            plotted = True

        else:
            print(f"[Warning] Missing eval file for kappa={kappa}")

        if not plotted:
            plt.close()
            continue

        # --------------------------------------------------
        # Kappa budget line
        # --------------------------------------------------
        plt.axhline(
            y=budget,
            linestyle="--",
            linewidth=1.5,
            label=fr"Budget $\kappa$={budget}"
        )

        plt.xlabel("Step")
        plt.ylabel("Handover Ratio")
        plt.title(fr"Training/Evaluation Handover Ratio ($\kappa$={kappa}, $\lambda_E$={lambda_E})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"handover_ratio_train_eval_lambda_{lambda_E}_kappa_{kappa}.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    plot_train_eval_handover_by_kappa(
        result_dir="results_kappa",
        lambda_E=15.0,
        window=1000,
        save_dir="results_kappa/plots"
    )