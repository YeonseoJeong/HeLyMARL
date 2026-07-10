import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ============================================================
# 1. npz 파일 경로만 여기에 맞게 수정
# ============================================================
NPZ_FILES = {
    "DDPP": "results/results_compare/DDPP_eval_lambda_0.npz",
    "MaxSNR": "results/results_compare/MaxSNR_eval_lambda_0.0.npz",
    "PF-HAPPO": "results/results_baselines/ConstrainedHAPPO_pf_eval_hard_kappa_0.03.npz",
    "Jensen-HAPPO": "results/results_baselines/ConstrainedHAPPO_jensen_eval_hard_kappa_0.03.npz",
    "HeLyMARL": "results/results_kappa/HeLyMARL_eval_hard_kappa_0.03.npz",
}

# ============================================================
# 2. 저장 폴더
# ============================================================
SAVE_DIR = "eval_compare_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 3. metric key mapping
# ============================================================
METRIC_KEYS = {
    "Throughput Mean": "throughput",
    "Fairness": "fairness",
    "ON Mean": "bs_on_ratio_mean",
    "HO Mean": "handover_ratio_mean",
}

def to_scalar_mean(x):
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.nan
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))

def jain_fairness(rates, eps=1e-12):
    rates = np.asarray(rates, dtype=float)

    if rates.size == 0:
        return np.nan

    if rates.ndim == 1:
        user_rates = rates
    else:
        # 보통 slot_rates가 [T, U] 형태라고 가정
        # 각 user 평균 rate 계산
        if rates.shape[0] >= rates.shape[1]:
            user_rates = np.nanmean(rates, axis=0)
        else:
            user_rates = np.nanmean(rates, axis=1)

    user_rates = np.nan_to_num(user_rates, nan=0.0, posinf=0.0, neginf=0.0)

    numerator = np.sum(user_rates) ** 2
    denominator = len(user_rates) * np.sum(user_rates ** 2) + eps
    return float(numerator / denominator)

def get_fairness(data, alg_name=None):

    if "fairness" in data.files:
        val = to_scalar_mean(data["fairness"])
        if not np.isnan(val):
            return val

    if "episode_fairness_last" in data.files:
        val = to_scalar_mean(data["episode_fairness_last"])
        if not np.isnan(val):
            return val

    if "avg_user_rates" in data.files:
        return jain_fairness(data["avg_user_rates"])

    if "slot_rates" in data.files:
        return jain_fairness(data["slot_rates"])

    return np.nan

# ============================================================
# 4. npz 읽고 dataframe 생성
# ============================================================
rows = []

for alg_name, path in NPZ_FILES.items():
    row = {"Algorithm": alg_name}

    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        for metric_name in METRIC_KEYS:
            row[metric_name] = np.nan
        rows.append(row)
        continue

    data = np.load(path, allow_pickle=True)

    for metric_name, key in METRIC_KEYS.items():
        if metric_name == "Fairness":
            row[metric_name] = get_fairness(data, alg_name=alg_name)
        else:
            if key in data.files:
                row[metric_name] = to_scalar_mean(data[key])
            else:
                row[metric_name] = np.nan
                print(f"[WARNING] {alg_name}: key '{key}' not found")

    rows.append(row)

df = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("EVALUATION COMPARISON")
print("=" * 80)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("=" * 80)

# CSV 저장
csv_path = os.path.join(SAVE_DIR, "eval_comparison_4metrics.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved CSV to: {csv_path}")

# ============================================================
# 5. 4개 subplot line plot만 저장
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

algorithms = df["Algorithm"].tolist()
x = np.arange(len(algorithms))

for ax, metric_name in zip(axes, METRIC_KEYS.keys()):
    y = df[metric_name].values.astype(float)

    ax.plot(x, y, marker='o', linewidth=2)
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15)
    ax.grid(True, alpha=0.3)

    # 값 표시하고 싶으면 주석 해제
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:.4f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()

save_path = os.path.join(SAVE_DIR, "all_4metrics_line_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved combined line plot: {save_path}")