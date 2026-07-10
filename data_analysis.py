import pandas as pd
import numpy as np
from scipy import stats

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH = r"D:\data\EuroSAT_MS\multiseed_analysis\per_seed_results.csv"
OUTPUT_PATH = "final_stats.csv"
CONF_LEVEL = 0.95

# ─────────────────────────────────────────────
# STEP 1: LOAD + CONVERT (WIDE → LONG)
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

df_long = df.melt(
    id_vars=["seed", "model"],
    var_name="metric",
    value_name="value"
)

# Clean metric names (optional but nice)
df_long["metric"] = df_long["metric"].str.replace("clean_acc", "clean")

# ─────────────────────────────────────────────
# STEP 2: FUNCTIONS
# ─────────────────────────────────────────────
def compute_ci(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if n < 2:
        return mean, (mean, mean)

    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_crit * (std / np.sqrt(n))

    return mean, (mean - margin, mean + margin)


def cohens_d_paired(x, y):
    x = np.array(x)
    y = np.array(y)
    diff = x - y
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)


def interpret_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

# ─────────────────────────────────────────────
# STEP 3: COMPUTE STATS
# ─────────────────────────────────────────────
results = []

metrics = df_long["metric"].unique()

for metric in metrics:
    df_m = df_long[df_long["metric"] == metric]

    resnet_vals = df_m[df_m["model"] == "ResNet34"]["value"].values
    vit_vals    = df_m[df_m["model"] == "ViT-S/16"]["value"].values

    # Ensure pairing by sorting seeds
    resnet_vals = np.array(resnet_vals)
    vit_vals    = np.array(vit_vals)

    if len(resnet_vals) < 2 or len(vit_vals) < 2:
        continue

    # CI
    r_mean, r_ci = compute_ci(resnet_vals, CONF_LEVEL)
    v_mean, v_ci = compute_ci(vit_vals, CONF_LEVEL)

    # Cohen's d
    d = cohens_d_paired(resnet_vals, vit_vals)
    d_label = interpret_d(d)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(resnet_vals, vit_vals)

    results.append({
        "Metric": metric,

        "ResNet Mean (%)": r_mean * 100,
        "ResNet CI Low (%)": r_ci[0] * 100,
        "ResNet CI High (%)": r_ci[1] * 100,

        "ViT Mean (%)": v_mean * 100,
        "ViT CI Low (%)": v_ci[0] * 100,
        "ViT CI High (%)": v_ci[1] * 100,

        "Cohen_d": d,
        "Effect Size": d_label,

        "p_value": p_val
    })

# ─────────────────────────────────────────────
# STEP 4: SAVE + PRINT
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="p_value")

print("\n=== FINAL STATISTICAL RESULTS ===\n")
print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH}")