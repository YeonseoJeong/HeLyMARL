import numpy as np

summary_files = {
    "PF-HAPPO": (
        "results/results_multi_seed/"
        "pf_happo_3train_5eval_summary.npz"
    ),
    "Jensen-HAPPO": (
        "results/results_multi_seed/"
        "jensen_happo_3train_5eval_summary.npz"
    ),
    "HeLyMARL": (
        "results/results_multi_seed/"
        "helymarl_3train_5eval_summary.npz"
    ),
}

for algorithm, path in summary_files.items():
    with np.load(path, allow_pickle=True) as data:
        train_seeds = np.asarray(
            data["train_seeds"]
        ).reshape(-1)

        throughput_values = np.asarray(
            data["throughput_per_train_seed"],
            dtype=float,
        ).reshape(-1)

        fairness_values = np.asarray(
            data["fairness_per_train_seed"],
            dtype=float,
        ).reshape(-1)

        print("\n" + "=" * 70)
        print(algorithm)
        print("=" * 70)

        for train_seed, throughput, fairness in zip(
            train_seeds,
            throughput_values,
            fairness_values,
        ):
            print(
                f"train_seed={train_seed} | "
                f"Throughput={throughput:.4f} | "
                f"Block-JFI={fairness:.4f}"
            )

        print("-" * 70)
        print(
            f"Final | "
            f"Throughput="
            f"{float(data['throughput_mean']):.4f} "
            f"+/- {float(data['throughput_std']):.4f} | "
            f"Block-JFI="
            f"{float(data['fairness_mean']):.4f} "
            f"+/- {float(data['fairness_std']):.4f}"
        )

# import numpy as np

# path = (
#     "results/results_multi_seed/evaluations/"
#     "helymarl/kappa_0.03_seed_0/eval_seed_2000.npz"
# )

# with np.load(path, allow_pickle=True) as data:
#     print(data.files)