"""
calculate_agegap_bias_corrected_adjusted.py  (ABIDE / ASD)

Age-bias-corrected AND covariate-adjusted version, mirroring the ADNI analysis.

(a) Age-bias correction (de Lange & Cole 2022): fit BAG ~ age on the training-set
    controls and subtract, giving a corrected BAG (near-)independent of age.

(b) Covariate-adjusted association (HEADLINE): a multivariable logistic
    regression of diagnosis (ASD vs Control) on the corrected BAG (in SD units),
    adjusting for age, sex, and acquisition site (SITE_ID). We report adjusted
    odds ratios (per +1 SD and per +1 year of corrected BAG) with 95% CIs. The
    simple +/-1 SD threshold odds ratios are retained as secondary descriptives.
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import mean_absolute_error, r2_score

PRED_TRAIN = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_train.csv"
PRED_VAL = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv"
META = "abide_brainrotnet_metadata.csv"
PLOTS = "model_dumps/mix/plots"
MIN_SITE_N = 10          # sites with fewer subjects pooled into OTHER
os.makedirs(PLOTS, exist_ok=True)


def fit_bias_correction(ref_df):
    raw_bag = (ref_df["Predicted_Age"] - ref_df["Age"]).to_numpy()
    X = sm.add_constant(ref_df["Age"].to_numpy())
    res = sm.OLS(raw_bag, X).fit()
    return res.params[1], res.params[0]


def apply_bias_correction(d, slope, intercept):
    d = d.copy()
    d["BAG_raw"] = d["Predicted_Age"] - d["Age"]
    d["AgeGap"] = d["BAG_raw"] - (slope * d["Age"] + intercept)
    return d


def compute_ci(a, b, c, d):
    or_val = (a / b) / (c / d)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci = np.exp(np.log(or_val) + np.array([-1, 1]) * 1.96 * se)
    return or_val, ci


# ---------------------------------------------------------------------------
# Load + correct
# ---------------------------------------------------------------------------
meta = pd.read_csv(META)
cols = ["ImageID", "Diag", "SITE_ID"]   # Sex already present in the prediction CSV
dfres_train = pd.read_csv(PRED_TRAIN, index_col=0).reset_index(drop=True).merge(meta[cols], on="ImageID", how="inner")
dfres_val = pd.read_csv(PRED_VAL, index_col=0).reset_index(drop=True).merge(meta[cols], on="ImageID", how="inner")

print("=== Validation-set accuracy ===")
print(f"n = {len(dfres_val)}   MAE = {mean_absolute_error(dfres_val.Age, dfres_val.Predicted_Age):.3f}   "
      f"R2 = {r2_score(dfres_val.Age, dfres_val.Predicted_Age):.3f}")

ref = dfres_train[dfres_train["Diag"] == "Control"]
slope, intercept = fit_bias_correction(ref)
print(f"\n=== Age-bias correction (de Lange & Cole 2022) ===")
print(f"Reference: {len(ref)} training controls; slope = {slope:.4f}, intercept = {intercept:.4f}")

dfres_train = apply_bias_correction(dfres_train, slope, intercept)
dfres_val = apply_bias_correction(dfres_val, slope, intercept)
df = pd.concat([dfres_train, dfres_val], ignore_index=True)
r_raw = np.corrcoef(df["BAG_raw"], df["Age"])[0, 1]
r_cor = np.corrcoef(df["AgeGap"], df["Age"])[0, 1]
print(f"corr(BAG, age): raw = {r_raw:+.3f}  ->  corrected = {r_cor:+.3f}")

df["IsASD"] = (df["Diag"] == "ASD").astype(int)
df["male"] = (df["Sex"] == "M").astype(int)
bag_sd = df["AgeGap"].std()
df["AgeGap_SD"] = df["AgeGap"] / bag_sd
site_counts = df["SITE_ID"].value_counts()
df["site_grp"] = np.where(df["SITE_ID"].isin(site_counts[site_counts >= MIN_SITE_N].index),
                          df["SITE_ID"], "OTHER")
print(f"Sites: {df['SITE_ID'].nunique()} raw -> {df['site_grp'].nunique()} after pooling (<{MIN_SITE_N} -> OTHER)")


# ===========================================================================
# HEADLINE: covariate-adjusted logistic regression   ASD ~ BAG + age + sex + site
# ===========================================================================
forest = []


def adjusted_or(adjust_site=True):
    terms = ["AgeGap_SD", "Age", "C(male)"]
    if adjust_site:
        terms.append("C(site_grp)")
    res = smf.logit("IsASD ~ " + " + ".join(terms), data=df).fit(disp=0, maxiter=200)
    coef = res.params["AgeGap_SD"]; ci = res.conf_int().loc["AgeGap_SD"]
    or_sd, ci_sd = np.exp(coef), np.exp(ci)
    or_yr, ci_yr = np.exp(coef / bag_sd), np.exp(ci / bag_sd)
    p = res.pvalues["AgeGap_SD"]
    tag = "age+sex+site" if adjust_site else "age+sex"
    print(f"\n[ASD] adjusted for {tag}  (n={int(res.nobs)})")
    print(f"  adjusted OR per +1 SD corrected BAG:  {or_sd:.4f} (95% CI {ci_sd[0]:.4f}-{ci_sd[1]:.4f}), p = {p:.4f}")
    print(f"  adjusted OR per +1 year corrected BAG: {or_yr:.4f} (95% CI {ci_yr[0]:.4f}-{ci_yr[1]:.4f})")
    forest.append({"label": f"ASD  (adj. {tag})", "or": or_sd,
                   "lo": float(ci_sd[0]), "hi": float(ci_sd[1]), "p": p})
    return res


print("\n" + "=" * 70)
print("PRIMARY ANALYSIS: covariate-adjusted logistic regression (ASD vs Control)")
print("=" * 70)
try:
    adjusted_or(adjust_site=True)
except Exception as e:
    print(f"full model (with site) failed ({type(e).__name__}); reporting age+sex only.")
adjusted_or(adjust_site=False)


# ===========================================================================
# SECONDARY (descriptive): +/-1 SD threshold odds ratio on corrected BAG
# ===========================================================================
print("\n" + "=" * 70)
print("SECONDARY: +1 SD threshold odds ratio on corrected BAG (ASD vs Control)")
print("=" * 70)
std_threshold = df["AgeGap"].std()
df["HighAgeGap"] = df["AgeGap"] > std_threshold
df["LowAgeGap"] = df["AgeGap"] < -std_threshold


def report_2x2(contingency, label):
    a = contingency.loc[True, True]; b = contingency.loc[True, False]
    c = contingency.loc[False, True]; d = contingency.loc[False, False]
    OR, ci = compute_ci(a, b, c, d)
    _, pchi, _, _ = chi2_contingency(contingency)
    _, pf = fisher_exact(contingency)
    print(f"\n[{label}]  OR {OR:.4f} (95% CI {ci[0]:.4f}-{ci[1]:.4f}); chi2 p {pchi:.4f}; fisher p {pf:.4f}")


report_2x2(pd.crosstab(df["HighAgeGap"], df["IsASD"].astype(bool)), "ASD: High(>+1SD) vs rest")
ext = df[df["HighAgeGap"] | df["LowAgeGap"]]
report_2x2(pd.crosstab(ext["HighAgeGap"], ext["IsASD"].astype(bool)), "ASD: High(>+1SD) vs Low(<-1SD)")


# ===========================================================================
# Forest plot of adjusted ORs (per +1 SD corrected BAG)
# ===========================================================================
seen, recs = set(), []
for r in forest:
    if r["label"] not in seen:
        seen.add(r["label"]); recs.append(r)
recs = recs[::-1]
fig, ax = plt.subplots(figsize=(8, 0.9 * len(recs) + 1.6))
ys = np.arange(len(recs))
for y, r in zip(ys, recs):
    color = "#005b96" if "site" in r["label"] else "#4C9F70"
    ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=2)
    ax.plot(r["or"], y, "o", color=color, ms=8)
    ax.annotate(f"{r['or']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}]  p={r['p']:.3f}",
                xy=(r["hi"], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5)
ax.axvline(1.0, color="black", ls="--", lw=1.2)
ax.set_yticks(ys); ax.set_yticklabels([r["label"] for r in recs], fontsize=9)
ax.set_xlabel("Adjusted odds ratio per +1 SD corrected BAG (95% CI)")
ax.set_title("ABIDE: covariate-adjusted association of corrected BAG with ASD", fontsize=12, weight="bold")
ax.set_xlim(0.8, max(r["hi"] for r in recs) * 1.35); ax.margins(y=0.25)
plt.tight_layout(); plt.savefig(f"{PLOTS}/abide_adjusted_or_forest.png", dpi=150); plt.close()
print(f"\nSaved plot: {PLOTS}/abide_adjusted_or_forest.png")
