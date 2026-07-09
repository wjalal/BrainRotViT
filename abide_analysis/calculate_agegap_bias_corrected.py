"""
calculate_agegap_bias_corrected.py  (ABIDE / ASD)

Age-bias-corrected version of calculate_agegap_crude.py, following
    de Lange A-MG & Cole JH, "Mind the gap: performance metric evaluation in
    brain-age prediction," Human Brain Mapping, 2022.

Brain-age models over-predict age in the young and under-predict in the old, so
the raw brain-age gap (BAG = predicted - chronological age) is confounded by
chronological age. We fit BAG ~ age on a reference cohort -- the TRAINING-set
control subjects -- and subtract the fitted trend from every subject's BAG. All
downstream thresholds / group comparisons then use the corrected BAG. Estimating
the correction only on training controls (never on ASD or on validation data)
avoids circularity.

Outcome of interest: Autism Spectrum Disorder (ASD) vs Control. Association is
evaluated on the full merged sample (train+val, 1053), matching the crude ABIDE
analysis; the correction itself is fit only on training controls.
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import mean_absolute_error, r2_score

PRED_TRAIN = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_train.csv"
PRED_VAL = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv"
META = "abide_brainrotnet_metadata.csv"
PLOTS = "model_dumps/mix/plots"
os.makedirs(PLOTS, exist_ok=True)


def fit_bias_correction(ref_df):
    """Fit raw BAG ~ age on a reference cohort; return (slope, intercept)."""
    raw_bag = (ref_df["Predicted_Age"] - ref_df["Age"]).to_numpy()
    X = sm.add_constant(ref_df["Age"].to_numpy())
    res = sm.OLS(raw_bag, X).fit()
    return res.params[1], res.params[0]        # slope, intercept


def apply_bias_correction(d, slope, intercept):
    d = d.copy()
    d["BAG_raw"] = d["Predicted_Age"] - d["Age"]
    d["AgeGap"] = d["BAG_raw"] - (slope * d["Age"] + intercept)   # corrected BAG
    return d


def compute_ci(a, b, c, d):
    or_val = (a / b) / (c / d)
    se_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_or = np.exp(np.log(or_val) + np.array([-1, 1]) * 1.96 * se_or)
    p1, p2 = a / (a + b), c / (c + d)
    rr_val = p1 / p2
    se_rr = np.sqrt((1/a) - (1/(a + b)) + (1/c) - (1/(c + d)))
    ci_rr = np.exp(np.log(rr_val) + np.array([-1, 1]) * 1.96 * se_rr)
    return or_val, ci_or, rr_val, ci_rr


# ---------------------------------------------------------------------------
# Load predictions + metadata (Diag, Sex, SITE_ID)
# ---------------------------------------------------------------------------
meta = pd.read_csv(META)
dfres_train = pd.read_csv(PRED_TRAIN, index_col=0).reset_index(drop=True)
dfres_val = pd.read_csv(PRED_VAL, index_col=0).reset_index(drop=True)
cols = ["ImageID", "Diag", "SITE_ID"]   # Sex already present in the prediction CSV
dfres_train = dfres_train.merge(meta[cols], on="ImageID", how="inner")
dfres_val = dfres_val.merge(meta[cols], on="ImageID", how="inner")

print("=== Validation-set accuracy (unaffected by correction) ===")
print(f"n = {len(dfres_val)}   "
      f"MAE = {mean_absolute_error(dfres_val.Age, dfres_val.Predicted_Age):.3f}   "
      f"R2 = {r2_score(dfres_val.Age, dfres_val.Predicted_Age):.3f}")

# ---------------------------------------------------------------------------
# Fit age-bias correction on TRAINING controls
# ---------------------------------------------------------------------------
ref = dfres_train[dfres_train["Diag"] == "Control"]
slope, intercept = fit_bias_correction(ref)
print("\n=== Age-bias correction (de Lange & Cole 2022) ===")
print(f"Reference: {len(ref)} training controls; BAG_raw ~ age "
      f"slope = {slope:.4f}, intercept = {intercept:.4f}")

dfres_train = apply_bias_correction(dfres_train, slope, intercept)
dfres_val = apply_bias_correction(dfres_val, slope, intercept)

# Association evaluated on the full merged sample (matches the crude ABIDE run).
df_merged = pd.concat([dfres_train, dfres_val], ignore_index=True)
r_raw = np.corrcoef(df_merged["BAG_raw"], df_merged["Age"])[0, 1]
r_cor = np.corrcoef(df_merged["AgeGap"], df_merged["Age"])[0, 1]
print(f"corr(BAG, age): raw = {r_raw:+.3f}  ->  corrected = {r_cor:+.3f}")

# ---------------------------------------------------------------------------
# Threshold at +1 SD of the CORRECTED BAG; ASD vs Control
# ---------------------------------------------------------------------------
std_threshold = df_merged["AgeGap"].std()
df_merged["HighAgeGap"] = df_merged["AgeGap"] > std_threshold
df_merged["LowAgeGap"] = df_merged["AgeGap"] < -std_threshold
df_merged["IsASD"] = df_merged["Diag"] == "ASD"
print(f"\nCorrected-BAG SD = {std_threshold:.3f}  "
      f"(high: {int(df_merged['HighAgeGap'].sum())}, low: {int(df_merged['LowAgeGap'].sum())})")


def report_2x2(contingency, label):
    a = contingency.loc[True, True]; b = contingency.loc[True, False]
    c = contingency.loc[False, True]; d = contingency.loc[False, False]
    OR, ciOR, RR, ciRR = compute_ci(a, b, c, d)
    _, pchi, _, _ = chi2_contingency(contingency)
    _, pf = fisher_exact(contingency)
    print(f"\n[{label}] (corrected BAG)")
    print(contingency)
    print(f"  Odds Ratio:     {OR:.4f} (95% CI: {ciOR[0]:.4f}-{ciOR[1]:.4f})")
    print(f"  Relative Risk:  {RR:.4f} (95% CI: {ciRR[0]:.4f}-{ciRR[1]:.4f})")
    print(f"  Chi-square p:   {pchi:.4f}   Fisher exact p: {pf:.4f}")


# High corrected-BAG (> +1 SD): ASD vs Control  (symmetric OR == 'ASD x more likely high BAG')
report_2x2(pd.crosstab(df_merged["HighAgeGap"], df_merged["IsASD"]),
           "ASD: High(>+1SD) vs rest")
# Extreme: High (> +1 SD) vs Low (< -1 SD)
ext = df_merged[df_merged["HighAgeGap"] | df_merged["LowAgeGap"]]
report_2x2(pd.crosstab(ext["HighAgeGap"], ext["IsASD"]),
           "ASD: High(>+1SD) vs Low(<-1SD)")

# ---------------------------------------------------------------------------
# Plots (crude-style, using corrected BAG) -- saved, not shown
# ---------------------------------------------------------------------------
# (a) corrected BAG vs age by diagnosis
fig, ax = plt.subplots(figsize=(10, 6))
s = df_merged[df_merged["Diag"] == "ASD"]
ax.scatter(s["Age"], s["AgeGap"], color="darkred", label="ASD", alpha=0.7, s=22)
s = df_merged[df_merged["Diag"] == "Control"]
ax.scatter(s["Age"], s["AgeGap"], facecolors=(0.8, 0.8, 0.8), edgecolors=(0.1, 0.1, 0.1),
           alpha=0.25, s=22, label="Control")
ax.axhline(std_threshold, color="black", ls="--", lw=1.2, label="+1 SD")
ax.axhline(-std_threshold, color="black", ls="--", lw=1.2, label="-1 SD")
ax.set_xlabel("Chronological age"); ax.set_ylabel("Age-bias-corrected BAG")
ax.set_title("ABIDE: corrected brain-age gap vs age by diagnosis")
ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(f"{PLOTS}/abide_corrected_bag_vs_age.png", dpi=150); plt.close()

# (b) Age vs predicted scatter, coloured by corrected BAG
lim = float(np.ceil(np.nanpercentile(np.abs(df_merged["AgeGap"]), 98))) or 1.0
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df_merged, x="Age", y="Predicted_Age", hue="AgeGap",
                palette="coolwarm", hue_norm=(-lim, lim), s=22)
plt.plot((0, 60), (0, 60), ls="--", color="gray", lw=1); plt.xlim(0, 60)
plt.title("ABIDE age-gap predictions (hue = corrected BAG)")
plt.xlabel("Chronological age"); plt.ylabel("Predicted age")
plt.tight_layout(); plt.savefig(f"{PLOTS}/abide_corrected_scatter.png", dpi=150); plt.close()

# (c) proportion of ASD vs Control by corrected Age-Gap group
cont = pd.crosstab(df_merged["HighAgeGap"], df_merged["IsASD"])
prop = cont.div(cont.sum(axis=1), axis=0).reset_index().melt(
    id_vars="HighAgeGap", var_name="IsASD", value_name="Proportion")
prop["HighAgeGap"] = prop["HighAgeGap"].map({True: "High BAG (> +1 SD)", False: "Low/Normal BAG"})
prop["IsASD"] = prop["IsASD"].map({True: "ASD", False: "Control"})
plt.figure(figsize=(6, 5))
sns.barplot(data=prop, x="HighAgeGap", y="Proportion", hue="IsASD", palette=["#b3cde0", "#005b96"])
plt.title("ABIDE: proportion ASD vs Control by corrected Age-Gap group", fontsize=12, weight="bold")
plt.xlabel(""); plt.ylabel("Proportion"); plt.ylim(0, 1); plt.legend(title="Diagnosis")
plt.tight_layout(); plt.savefig(f"{PLOTS}/abide_corrected_prop.png", dpi=150); plt.close()

print(f"\nSaved plots -> {PLOTS}/abide_corrected_bag_vs_age.png, "
      f"abide_corrected_scatter.png, abide_corrected_prop.png")
