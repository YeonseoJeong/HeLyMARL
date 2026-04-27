import os
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_get(data, key, default=None):
    if key in data.files:
        return data[key]
    return default


def parse_lambda(path):
    name = os.path.basename(path)
    m = re.search(r"lambda_([0-9.]+)", name)
    if m is None:
        return None
    return float(m.group(1).rstrip("."))

def infer_method(path):
    name = os.path.basename(path).lower()

    if "ddpp" in name:
        return "DDPP"

    if "soft" in name:
        return "LyMARL-Soft"

    if "hard" in name:
        return "LyMARL-Hard"

    if "lymarl" in name or "happo" in name:
        return "LyMARL-Hard"

    return "Unknown"


def summarize_npz(path, last_window=10000):
    data = np.load(path, allow_pickle=True)

    tag = safe_get(data, "tag", "Unknown")
    if isinstance(tag, np.ndarray):
        tag = str(tag.item())
    else:
        tag = str(tag)

    throughput = safe_get(data, "throughput", np.array([]))
    fairness = safe_get(data, "fairness", np.array([]))
    handover_ratio = safe_get(data, "handover_ratio", np.array([]))
    power_mat = safe_get(data, "power_mat", np.zeros((0, 0), dtype=np.float32))

    if throughput.size > 0:
        throughput_mean = float(np.mean(throughput[-last_window:]))
    else:
        throughput_mean = np.nan

    if fairness.size > 0:
        fairness_mean = float(np.nanmean(fairness[-last_window:]))
    else:
        fairness_mean = np.nan

    if handover_ratio.size > 0:
        handover_mean = float(np.mean(handover_ratio[-last_window:]))
    else:
        handover_mean = np.nan

    if power_mat.size > 0:
        recent_power = power_mat[:, -last_window:]
        on_ratio_mean = float(np.mean(recent_power > 0.0))
    else:
        bs_on_ratio_mean = safe_get(data, "bs_on_ratio_mean", np.array([np.nan]))
        on_ratio_mean = float(bs_on_ratio_mean[0]) if len(bs_on_ratio_mean) > 0 else np.nan

    return {
        "tag": tag,
        "throughput": throughput_mean,
        "fairness": fairness_mean,
        "on_ratio": on_ratio_mean,
        "handover_ratio": handover_mean,
    }


def plot_bar(labels, values, ylabel, title, save_path, hline=None, hline_label=None):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))

    colors = []
    for label in labels:
        if "DDPP" in label.upper():
            colors.append("tab:pink")
        elif "HAPPO" in label.upper():
            colors.append("tab:blue")
        else:
            colors.append("tab:gray")

    plt.bar(x, values, color=colors)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)

    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=2, label=hline_label)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_grouped_bar_by_lambda(
    summaries, metric_key, ylabel, title, save_path,
    hline=None, hline_label=None
):
    grouped = {}

    methods = ["DDPP", "LyMARL-Hard", "LyMARL-Soft"]
    colors = {
        "DDPP": "tab:orange",
        "LyMARL-Hard": "tab:blue",
        "LyMARL-Soft": "tab:green",
    }

    for s in summaries:
        lam = s.get("lambda", None)
        method = s.get("method", None)

        if lam is None or method is None:
            continue

        if method not in methods:
            continue

        if lam not in grouped:
            grouped[lam] = {}

        grouped[lam][method] = s[metric_key]

    lambdas = sorted(grouped.keys())

    x = np.arange(len(lambdas))
    width = 0.25

    plt.figure(figsize=(10, 5))

    offsets = {
        "DDPP": -width,
        "LyMARL-Hard": 0.0,
        "LyMARL-Soft": width,
    }

    for method in methods:
        vals = [grouped[lam].get(method, np.nan) for lam in lambdas]
        plt.bar(
            x + offsets[method],
            vals,
            width,
            label=method,
            color=colors[method],
        )

    plt.xticks(x, [f"λ={lam:g}" for lam in lambdas], rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)

    if hline is not None:
        plt.axhline(
            hline,
            linestyle="--",
            linewidth=2,
            color="black",
            label=hline_label,
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_metric_comparison(summaries, plot_dir):
    plot_grouped_bar_by_lambda(
        summaries,
        metric_key="throughput",
        ylabel="Throughput [Gbps]",
        title="Throughput Comparison by λ",
        save_path=os.path.join(plot_dir, "compare_throughput_grouped.png")
    )

    plot_grouped_bar_by_lambda(
        summaries,
        metric_key="fairness",
        ylabel="Jain's Fairness Index",
        title="Fairness Comparison by λ",
        save_path=os.path.join(plot_dir, "compare_fairness_grouped.png")
    )

    plot_grouped_bar_by_lambda(
        summaries,
        metric_key="on_ratio",
        ylabel="BS ON Ratio",
        title="BS ON Ratio Comparison by λ",
        save_path=os.path.join(plot_dir, "compare_on_ratio_grouped.png"),
        hline=0.6,
        hline_label="Energy Budget"
    )

    plot_grouped_bar_by_lambda(
        summaries,
        metric_key="handover_ratio",
        ylabel="Handover Ratio",
        title="Handover Ratio Comparison by λ",
        save_path=os.path.join(plot_dir, "compare_handover_ratio_grouped.png"),
        hline=0.1,
        hline_label="Handover Budget"
    )


def print_summary_table(summaries):
    method_order = ["DDPP", "LyMARL-Hard", "LyMARL-Soft"]

    grouped = {}
    for s in summaries:
        lam = s.get("lambda", None)
        method = s.get("method", "Unknown")

        if lam is None:
            continue

        if lam not in grouped:
            grouped[lam] = {}

        grouped[lam][method] = s

    print("\n" + "=" * 92)
    print("Performance Comparison by lambda_E")
    print("=" * 92)

    for lam in sorted(grouped.keys()):
        print(f"\nλ_E = {lam:g}")
        print("-" * 92)
        print(
            f"{'Method':<18} | "
            f"{'Throughput':>12} | "
            f"{'Fairness':>10} | "
            f"{'ON Ratio':>10} | "
            f"{'HO Ratio':>10}"
        )
        print("-" * 92)

        for method in method_order:
            if method not in grouped[lam]:
                continue

            s = grouped[lam][method]

            print(
                f"{method:<18} | "
                f"{s['throughput']:>12.4f} | "
                f"{s['fairness']:>10.4f} | "
                f"{s['on_ratio']:>10.4f} | "
                f"{s['handover_ratio']:>10.4f}"
            )

    print("\n" + "=" * 92 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpp_npz", type=str, default="results_compare/DDPP_eval_lambda_*.npz")
    parser.add_argument("--happo_dir", type=str, default="results_lambda")
    parser.add_argument("--plot_dir", type=str, default="results_compare/plots")
    parser.add_argument("--last_window", type=int, default=10000)
    parser.add_argument("--happo_lambda", type=float, default=None)
    args = parser.parse_args()

    ensure_dir(args.plot_dir)

    files = []

    ddpp_files = glob.glob(args.ddpp_npz)
    ddpp_files = sorted(
        ddpp_files,
        key=lambda p: parse_lambda(p) if parse_lambda(p) is not None else 1e9
    )
    if len(ddpp_files) > 0:
        files.extend(ddpp_files)
    else:
        print(f"[Warning] DDPP npz not found: {args.ddpp_npz}")

    happo_files = glob.glob(os.path.join(args.happo_dir, "**", "*eval*lambda_*.npz"), recursive=True)

    if args.happo_lambda is not None:
        selected = []
        for f in happo_files:
            lam = parse_lambda(f)
            if lam is not None and abs(lam - args.happo_lambda) < 1e-8:
                selected.append(f)
        happo_files = selected

    happo_files = sorted(happo_files, key=lambda p: parse_lambda(p) if parse_lambda(p) is not None else 1e9)

    files.extend(happo_files)

    if len(files) == 0:
        print("[Error] No npz files found.")
        return

    summaries = []
    for f in files:
        s = summarize_npz(f, last_window=args.last_window)

        lam = parse_lambda(f)
        method = infer_method(f)

        s["lambda"] = lam
        s["method"] = method

        if lam is not None:
            s["tag"] = f"{method} λ={lam:g}"
        else:
            s["tag"] = method

        summaries.append(s)
    print_summary_table(summaries)
    plot_metric_comparison(summaries, args.plot_dir)

    print(f"✅ Saved comparison plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()