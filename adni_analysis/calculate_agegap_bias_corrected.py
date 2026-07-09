"""
calculate_agegap_bias_corrected.py

Age-bias-corrected version of calculate_agegap_crude.py.

Brain-age models systematically over-predict age in younger individuals and
under-predict it in older ones, so the raw brain-age gap (BAG = predicted age -
chronological age) is confounded by chronological age itself. Following

    de Lange A-MG & Cole JH, "Mind the gap: performance metric evaluation in
    brain-age prediction," Human Brain Mapping, 2022,

we remove this age dependence by fitting BAG ~ age on a reference cohort
(here, the training-set cognitively-normal controls) and subtracting the fitted
trend from every subject's BAG. All downstream thresholds and group
comparisons use the corrected BAG. Estimating the correction on the training
controls (never the held-out validation set) avoids circularity.
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib
matplotlib.use("Agg")   # headless: save figures to disk instead of plt.show()
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import mean_absolute_error, r2_score

PRED_TRAIN = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_train.csv"
PRED_VAL = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv"
META = "adni_brainrotnet_metadata.csv"


# ---------------------------------------------------------------------------
# Age-bias correction (de Lange & Cole, Hum Brain Mapp 2022)
# ---------------------------------------------------------------------------
def fit_bias_correction(ref_df):
    """Fit raw BAG ~ age on a reference cohort; return (slope, intercept)."""
    raw_bag = (ref_df["Predicted_Age"] - ref_df["Age"]).to_numpy()
    X = sm.add_constant(ref_df["Age"].to_numpy())
    res = sm.OLS(raw_bag, X).fit()
    intercept, slope = res.params[0], res.params[1]
    return slope, intercept


def apply_bias_correction(dfres, slope, intercept):
    """Return a copy with raw BAG and age-bias-corrected BAG (as 'AgeGap')."""
    d = dfres.copy()
    d["BAG_raw"] = d["Predicted_Age"] - d["Age"]
    d["AgeGap"] = d["BAG_raw"] - (slope * d["Age"] + intercept)  # corrected BAG
    return d


def compute_ci(a, b, c, d):
    """Return (OR, RR) with 95% CIs from a 2x2 count table."""
    or_val = (a / b) / (c / d)
    se_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_or = np.exp(np.log(or_val) + np.array([-1, 1]) * 1.96 * se_or)
    p1, p2 = a / (a + b), c / (c + d)
    rr_val = p1 / p2
    se_rr = np.sqrt((1/a) - (1/(a + b)) + (1/c) - (1/(c + d)))
    ci_rr = np.exp(np.log(rr_val) + np.array([-1, 1]) * 1.96 * se_rr)
    return or_val, ci_or, rr_val, ci_rr


# ---------------------------------------------------------------------------
# Load predictions + metadata
# ---------------------------------------------------------------------------
meta = pd.read_csv(META)

dfres_train = pd.read_csv(PRED_TRAIN, index_col=0).reset_index(drop=True)
dfres_val = pd.read_csv(PRED_VAL, index_col=0).reset_index(drop=True)

# Attach diagnostic Group to both sets (needed to pick training controls).
dfres_train = dfres_train.merge(meta[["ImageID", "Group"]], on="ImageID", how="inner")
dfres_val = dfres_val.merge(meta[["ImageID", "Group"]], on="ImageID", how="inner")

# Validation-set accuracy (unaffected by BAG correction; reported for the paper).
print("=== Validation-set accuracy ===")
print(f"n = {len(dfres_val)}")
print(f"MAE = {mean_absolute_error(dfres_val.Age, dfres_val.Predicted_Age):.3f}")
print(f"R2  = {r2_score(dfres_val.Age, dfres_val.Predicted_Age):.3f}")

# ---------------------------------------------------------------------------
# Fit the age-bias correction on TRAINING cognitively-normal (CN) controls
# ---------------------------------------------------------------------------
ref = dfres_train[dfres_train["Group"] == "CN"]
slope, intercept = fit_bias_correction(ref)
print("\n=== Age-bias correction (de Lange & Cole 2022) ===")
print(f"Reference cohort: {len(ref)} training CN controls")
print(f"Fitted  BAG_raw ~ age:  slope = {slope:.4f}, intercept = {intercept:.4f}")

# Apply correction everywhere; 'AgeGap' now holds the CORRECTED BAG.
dfres_train = apply_bias_correction(dfres_train, slope, intercept)
dfres_val = apply_bias_correction(dfres_val, slope, intercept)

# Sanity check: corrected BAG should be (near) uncorrelated with age.
r_raw = np.corrcoef(dfres_val["BAG_raw"], dfres_val["Age"])[0, 1]
r_cor = np.corrcoef(dfres_val["AgeGap"], dfres_val["Age"])[0, 1]
print(f"Val corr(BAG, age): raw = {r_raw:+.3f}  ->  corrected = {r_cor:+.3f}")

# From here on df_merged carries the corrected BAG in 'AgeGap'.
df_merged = dfres_val.copy()

# ---------------------------------------------------------------------------
# Threshold at +/- 1 SD of the CORRECTED BAG
# ---------------------------------------------------------------------------
std_threshold = df_merged["AgeGap"].std()
df_merged["HighAgeGap"] = df_merged["AgeGap"] > std_threshold
df_merged["LowAgeGap"] = df_merged["AgeGap"] < -std_threshold
print(f"\nCorrected-BAG SD = {std_threshold:.3f}  "
      f"(high: {int(df_merged['HighAgeGap'].sum())}, low: {int(df_merged['LowAgeGap'].sum())})")


def report_2x2(contingency, label):
    a = contingency.loc[True, True]
    b = contingency.loc[True, False]
    c = contingency.loc[False, True]
    d = contingency.loc[False, False]
    OR, ciOR, RR, ciRR = compute_ci(a, b, c, d)
    _, pchi, _, _ = chi2_contingency(contingency)
    _, pf = fisher_exact(contingency)
    print(f"\n[{label}] (corrected BAG)")
    print(contingency)
    print(f"  Odds Ratio:     {OR:.3f} (95% CI: {ciOR[0]:.3f}-{ciOR[1]:.3f})")
    print(f"  Relative Risk:  {RR:.3f} (95% CI: {ciRR[0]:.3f}-{ciRR[1]:.3f})")
    print(f"  Chi-square p:   {pchi:.3e}")
    print(f"  Fisher exact p: {pf:.3e}")


# ---------------------------------------------------------------------------
# Comparison 1: High corrected-BAG (> +1 SD) vs the rest
# ---------------------------------------------------------------------------
df_merged["IsAD"] = df_merged["Group"] == "AD"
df_merged["IsADorMCI"] = df_merged["Group"].isin(["AD", "MCI"])

report_2x2(pd.crosstab(df_merged["HighAgeGap"], df_merged["IsAD"]),
           "AD: High(>+1SD) vs rest")
report_2x2(pd.crosstab(df_merged["HighAgeGap"], df_merged["IsADorMCI"]),
           "AD/MCI: High(>+1SD) vs rest")

# ---------------------------------------------------------------------------
# Comparison 2: High (> +1 SD) vs Low (< -1 SD) corrected-BAG
# ---------------------------------------------------------------------------
df_extremes = df_merged[df_merged["HighAgeGap"] | df_merged["LowAgeGap"]].copy()
df_extremes["IsAD"] = df_extremes["Group"] == "AD"
df_extremes["IsADorMCI"] = df_extremes["Group"].isin(["AD", "MCI"])

report_2x2(pd.crosstab(df_extremes["HighAgeGap"], df_extremes["IsAD"]),
           "AD: High(>+1SD) vs Low(<-1SD)")
report_2x2(pd.crosstab(df_extremes["HighAgeGap"], df_extremes["IsADorMCI"]),
           "AD/MCI: High(>+1SD) vs Low(<-1SD)")

# ---------------------------------------------------------------------------
# Diagnostic scatter: corrected BAG vs age by group (saved, not shown)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for grp, color in [("AD", "darkred"), ("MCI", "orange")]:
    s = df_merged[df_merged["Group"] == grp]
    ax.scatter(s["Age"], s["AgeGap"], color=color, label=grp, alpha=0.8, s=25)
s = df_merged[df_merged["Group"] == "CN"]
ax.scatter(s["Age"], s["AgeGap"], facecolors=(0.8, 0.8, 0.8),
           edgecolors=(0.1, 0.1, 0.1), alpha=0.25, s=25, label="CN")
ax.axhline(std_threshold, color="black", linestyle="--", linewidth=1.2, label="+1 SD")
ax.axhline(-std_threshold, color="black", linestyle="--", linewidth=1.2, label="-1 SD")
ax.set_xlabel("Chronological age")
ax.set_ylabel("Age-bias-corrected BAG")
ax.set_title("Corrected brain-age gap vs age by cognitive status")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("model_dumps/mix/plots/adni_corrected_bag_vs_age.png", dpi=150)
plt.close()
print("\nSaved plot: model_dumps/mix/plots/adni_corrected_bag_vs_age.png")


# ===========================================================================
# Crude-style plot set (same figures as calculate_agegap_crude.py) but driven
# by the AGE-BIAS-CORRECTED BAG. Saved to disk (headless) rather than shown.
# ===========================================================================
PLOTS = "model_dumps/mix/plots"
os.makedirs(PLOTS, exist_ok=True)


# (1)-(2) Age vs Predicted-age scatter, coloured by corrected BAG, with MAE/R2.
def scatter_pred(data, title, fname, xlim):
    mae = mean_absolute_error(data["Age"], data["Predicted_Age"])
    r2 = r2_score(data["Age"], data["Predicted_Age"])
    lim = float(np.ceil(np.nanpercentile(np.abs(data["AgeGap"]), 98))) or 1.0
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=data, x="Age", y="Predicted_Age", hue="AgeGap",
                    palette="coolwarm", hue_norm=(-lim, lim), s=25)
    plt.plot(xlim, xlim, ls="--", color="gray", lw=1)          # identity line
    plt.xlim(*xlim)
    plt.title(f"{title}\nMAE: {mae:.2f}, R2: {r2:.2f}   (hue = corrected BAG)")
    plt.xlabel("Chronological age"); plt.ylabel("Predicted age")
    plt.legend(title="Corrected BAG", loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{PLOTS}/{fname}", dpi=150); plt.close()
    print(f"Saved plot: {PLOTS}/{fname}")


scatter_pred(dfres_train, "Corrected age-gap predictions (Train Set)",
             "adni_corrected_scatter_train.png", (40, 100))
scatter_pred(dfres_val, "Corrected age-gap predictions (Validation Set)",
             "adni_corrected_scatter_val.png", (50, 100))


# (3) Proportion of AD/MCI vs CN: High corrected-BAG (> +1 SD) vs Low/Normal.
def prop_melt(contingency):
    p = contingency.div(contingency.sum(axis=1), axis=0).reset_index()
    return p.melt(id_vars=p.columns[0], var_name="IsADorMCI", value_name="Proportion")


cont_high = pd.crosstab(df_merged["HighAgeGap"], df_merged["IsADorMCI"])
prop_high = prop_melt(cont_high)
prop_high["HighAgeGap"] = prop_high["HighAgeGap"].map({True: "High BAG (> +1 SD)", False: "Low/Normal BAG"})
prop_high["IsADorMCI"] = prop_high["IsADorMCI"].map({True: "AD/MCI", False: "CN"})
plt.figure(figsize=(6, 5))
sns.barplot(data=prop_high, x="HighAgeGap", y="Proportion", hue="IsADorMCI",
            palette=["#b3cde0", "#005b96"])
plt.title("Proportion of AD/MCI vs CN by corrected Age-Gap group", fontsize=13, weight="bold")
plt.xlabel(""); plt.ylabel("Proportion"); plt.ylim(0, 1); plt.legend(title="Diagnosis", loc="upper right")
plt.tight_layout(); plt.savefig(f"{PLOTS}/adni_corrected_prop_high_vs_rest.png", dpi=150); plt.close()
print(f"Saved plot: {PLOTS}/adni_corrected_prop_high_vs_rest.png")


# (4) Proportion of AD/MCI vs CN: extreme High (> +1 SD) vs Low (< -1 SD).
extreme_df = df_merged[df_merged["HighAgeGap"] | df_merged["LowAgeGap"]].copy()
extreme_df["ExtremeGroup"] = np.where(extreme_df["HighAgeGap"], "High BAG (> +1 SD)", "Low BAG (< -1 SD)")
cont_ext = pd.crosstab(extreme_df["ExtremeGroup"], extreme_df["IsADorMCI"])
prop_ext = prop_melt(cont_ext)
prop_ext["IsADorMCI"] = prop_ext["IsADorMCI"].map({True: "AD/MCI", False: "CN"})
plt.figure(figsize=(6, 5))
sns.barplot(data=prop_ext, x="ExtremeGroup", y="Proportion", hue="IsADorMCI",
            palette=["#b3cde0", "#005b96"])
plt.title("Proportion of AD/MCI vs CN by extreme corrected Age-Gap", fontsize=13, weight="bold")
plt.xlabel(""); plt.ylabel("Proportion"); plt.ylim(0, 1); plt.legend(title="Diagnosis", loc="upper right")
plt.tight_layout(); plt.savefig(f"{PLOTS}/adni_corrected_prop_extreme.png", dpi=150); plt.close()
print(f"Saved plot: {PLOTS}/adni_corrected_prop_extreme.png")


# (5) Grouped bars: proportion AD and AD/MCI in Low (< -1 SD) vs High (> +1 SD).
high_group = df_merged[df_merged["AgeGap"] > std_threshold]
low_group = df_merged[df_merged["AgeGap"] < -std_threshold]
high_props = [(high_group["Group"] == "AD").mean(), high_group["Group"].isin(["AD", "MCI"]).mean()]
low_props = [(low_group["Group"] == "AD").mean(), low_group["Group"].isin(["AD", "MCI"]).mean()]
groups = ["AD", "AD/MCI"]; x = range(len(groups)); width = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
b1 = ax.bar([i - width / 2 for i in x], low_props, width, label="Corrected BAG < -1 SD", color="steelblue")
b2 = ax.bar([i + width / 2 for i in x], high_props, width, label="Corrected BAG > +1 SD", color="salmon")
ax.set_ylabel("Proportion of subjects"); ax.set_ylim(0, 1)
ax.set_title("Proportion of AD and AD/MCI by corrected Age-Gap group")
ax.set_xticks(list(x)); ax.set_xticklabels(groups); ax.legend()
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(f"{PLOTS}/adni_corrected_prop_grouped_bar.png", dpi=150); plt.close()
print(f"Saved plot: {PLOTS}/adni_corrected_prop_grouped_bar.png")
