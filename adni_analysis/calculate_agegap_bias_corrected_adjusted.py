"""
calculate_agegap_bias_corrected_adjusted.py

Age-bias-corrected AND covariate-adjusted version of calculate_agegap_crude.py.

Two methodological upgrades over the crude thresholded analysis:

(a) Age-bias correction (de Lange & Cole, "Mind the gap: performance metric
    evaluation in brain-age prediction," Hum Brain Mapp 2022): brain-age models
    over-predict the young and under-predict the old, so raw BAG is confounded
    by chronological age. We fit BAG ~ age on the training-set cognitively-
    normal (CN) controls and subtract the fitted trend, giving a corrected BAG
    that is (near-)independent of age.

(b) Covariate-adjusted association: instead of relying only on a raw odds ratio
    from a +1 SD threshold, the HEADLINE result is a multivariable logistic
    regression of diagnosis on corrected BAG, adjusting for age, sex, and
    acquisition site (encoded in the ADNI SubjectID, NNN_S_NNNN). We report
    adjusted odds ratios (per +1 SD and per +1 year of corrected BAG) with 95%
    CIs. The simple threshold-based odds ratios are retained as secondary
    descriptive statistics.
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")   # headless: save figures to disk
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import mean_absolute_error, r2_score

PRED_TRAIN = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_train.csv"
PRED_VAL = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv"
META = "adni_brainrotnet_metadata.csv"
MIN_SITE_N = 10  # sites with fewer val subjects are pooled into "OTHER" for stability


# ---------------------------------------------------------------------------
# Age-bias correction (de Lange & Cole 2022)
# ---------------------------------------------------------------------------
def fit_bias_correction(ref_df):
    raw_bag = (ref_df["Predicted_Age"] - ref_df["Age"]).to_numpy()
    X = sm.add_constant(ref_df["Age"].to_numpy())
    res = sm.OLS(raw_bag, X).fit()
    intercept, slope = res.params[0], res.params[1]
    return slope, intercept


def apply_bias_correction(dfres, slope, intercept):
    d = dfres.copy()
    d["BAG_raw"] = d["Predicted_Age"] - d["Age"]
    d["AgeGap"] = d["BAG_raw"] - (slope * d["Age"] + intercept)  # corrected BAG
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
# Load predictions + metadata (Group, Sex, SubjectID->site)
# ---------------------------------------------------------------------------
meta = pd.read_csv(META)
meta["site"] = meta["SubjectID"].astype(str).str.extract(r"^(\d+)_S_")[0]

# Sex and Age already come from the prediction CSVs; pull only the extras.
cols = ["ImageID", "SubjectID", "Group", "site"]
dfres_train = pd.read_csv(PRED_TRAIN, index_col=0).reset_index(drop=True).merge(
    meta[cols], on="ImageID", how="inner")
dfres_val = pd.read_csv(PRED_VAL, index_col=0).reset_index(drop=True).merge(
    meta[cols], on="ImageID", how="inner")

print("=== Validation-set accuracy ===")
print(f"n = {len(dfres_val)}   "
      f"MAE = {mean_absolute_error(dfres_val.Age, dfres_val.Predicted_Age):.3f}   "
      f"R2 = {r2_score(dfres_val.Age, dfres_val.Predicted_Age):.3f}")

# ---------------------------------------------------------------------------
# Age-bias correction fit on TRAINING CN controls
# ---------------------------------------------------------------------------
ref = dfres_train[dfres_train["Group"] == "CN"]
slope, intercept = fit_bias_correction(ref)
print("\n=== Age-bias correction (de Lange & Cole 2022) ===")
print(f"Reference: {len(ref)} training CN controls; "
      f"BAG_raw ~ age slope = {slope:.4f}, intercept = {intercept:.4f}")

dfres_val = apply_bias_correction(dfres_val, slope, intercept)
r_raw = np.corrcoef(dfres_val["BAG_raw"], dfres_val["Age"])[0, 1]
r_cor = np.corrcoef(dfres_val["AgeGap"], dfres_val["Age"])[0, 1]
print(f"Val corr(BAG, age): raw = {r_raw:+.3f}  ->  corrected = {r_cor:+.3f}")

df = dfres_val.copy()
df["IsAD"] = (df["Group"] == "AD").astype(int)
df["IsADorMCI"] = df["Group"].isin(["AD", "MCI"]).astype(int)
df["male"] = (df["Sex"] == "M").astype(int)
bag_sd = df["AgeGap"].std()
df["AgeGap_SD"] = df["AgeGap"] / bag_sd          # corrected BAG in SD units
# Pool rare sites so the categorical term is estimable.
site_counts = df["site"].value_counts()
df["site_grp"] = np.where(df["site"].isin(site_counts[site_counts >= MIN_SITE_N].index),
                          df["site"], "OTHER")
n_sites = df["site_grp"].nunique()
print(f"Sites: {df['site'].nunique()} raw -> {n_sites} after pooling (<{MIN_SITE_N} -> OTHER)")


# ===========================================================================
# HEADLINE: covariate-adjusted logistic regression
#   diagnosis ~ corrected_BAG + age + sex + site
# ===========================================================================
OUTCOME_LABEL = {"IsADorMCI": "AD/MCI", "IsAD": "AD"}
forest_records = []   # collected adjusted ORs (per +1 SD) for the forest plot


def adjusted_or(outcome, adjust_site=True):
    terms = ["AgeGap_SD", "Age", "C(male)"]
    if adjust_site:
        terms.append("C(site_grp)")
    formula = f"{outcome} ~ " + " + ".join(terms)
    res = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    coef = res.params["AgeGap_SD"]
    ci = res.conf_int().loc["AgeGap_SD"]
    or_sd = np.exp(coef)
    ci_sd = np.exp(ci)
    # per-year OR = OR per SD scaled to one year of corrected BAG
    or_yr = np.exp(coef / bag_sd)
    ci_yr = np.exp(ci / bag_sd)
    p = res.pvalues["AgeGap_SD"]
    tag = "age+sex+site" if adjust_site else "age+sex"
    print(f"\n[{outcome}] adjusted for {tag}  (n={int(res.nobs)})")
    print(f"  adjusted OR per +1 SD corrected BAG:  {or_sd:.3f} (95% CI {ci_sd[0]:.3f}-{ci_sd[1]:.3f}), p = {p:.3e}")
    print(f"  adjusted OR per +1 year corrected BAG: {or_yr:.3f} (95% CI {ci_yr[0]:.3f}-{ci_yr[1]:.3f})")
    forest_records.append({
        "label": f"{OUTCOME_LABEL.get(outcome, outcome)}  (adj. {tag})",
        "or": or_sd, "lo": float(ci_sd[0]), "hi": float(ci_sd[1]), "p": p, "n": int(res.nobs),
    })
    return res


print("\n" + "=" * 70)
print("PRIMARY ANALYSIS: covariate-adjusted logistic regression")
print("=" * 70)
for outcome in ["IsADorMCI", "IsAD"]:
    try:
        adjusted_or(outcome, adjust_site=True)
    except Exception as e:
        print(f"\n[{outcome}] full model (with site) failed ({type(e).__name__}); "
              f"reporting age+sex-adjusted instead.")
        adjusted_or(outcome, adjust_site=False)
    # Always also show the age+sex-only model as a robustness check.
    adjusted_or(outcome, adjust_site=False)


# ===========================================================================
# SECONDARY (descriptive): simple +/-1 SD threshold odds ratios on corrected BAG
# ===========================================================================
print("\n" + "=" * 70)
print("SECONDARY (descriptive): +/-1 SD threshold odds ratios on corrected BAG")
print("=" * 70)
std_threshold = df["AgeGap"].std()
df["HighAgeGap"] = df["AgeGap"] > std_threshold
df["LowAgeGap"] = df["AgeGap"] < -std_threshold


def report_2x2(contingency, label):
    a = contingency.loc[True, True]; b = contingency.loc[True, False]
    c = contingency.loc[False, True]; d = contingency.loc[False, False]
    OR, ciOR, RR, ciRR = compute_ci(a, b, c, d)
    _, pchi, _, _ = chi2_contingency(contingency)
    _, pf = fisher_exact(contingency)
    print(f"\n[{label}]")
    print(f"  OR {OR:.3f} (95% CI {ciOR[0]:.3f}-{ciOR[1]:.3f}); "
          f"RR {RR:.3f} (95% CI {ciRR[0]:.3f}-{ciRR[1]:.3f}); "
          f"chi2 p {pchi:.3e}; fisher p {pf:.3e}")


for outcome_col, name in [("IsAD", "AD"), ("IsADorMCI", "AD/MCI")]:
    report_2x2(pd.crosstab(df["HighAgeGap"], df[outcome_col].astype(bool)),
               f"{name}: High(>+1SD) vs rest")
ext = df[df["HighAgeGap"] | df["LowAgeGap"]]
for outcome_col, name in [("IsAD", "AD"), ("IsADorMCI", "AD/MCI")]:
    report_2x2(pd.crosstab(ext["HighAgeGap"], ext[outcome_col].astype(bool)),
               f"{name}: High(>+1SD) vs Low(<-1SD)")


# ===========================================================================
# Forest plot of the covariate-adjusted odds ratios (per +1 SD corrected BAG).
# This is the figure that reflects what THIS script adds over the crude/
# bias-corrected ones: the multivariable adjustment. (The threshold-based
# proportion/scatter plots would be identical to the bias-corrected script's,
# since they use the same corrected BAG, so they are not duplicated here.)
# ===========================================================================
PLOTS = "model_dumps/mix/plots"
os.makedirs(PLOTS, exist_ok=True)

# de-duplicate by label (the age+sex model may be reported twice) keeping first.
seen, records = set(), []
for r in forest_records:
    if r["label"] not in seen:
        seen.add(r["label"]); records.append(r)
records = records[::-1]   # so the first-reported row lands at the top

fig, ax = plt.subplots(figsize=(8, 0.9 * len(records) + 1.6))
ys = np.arange(len(records))
for y, r in zip(ys, records):
    color = "#005b96" if "site" in r["label"] else "#4C9F70"
    ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=2)
    ax.plot(r["or"], y, "o", color=color, ms=8)
    ax.annotate(f"{r['or']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}]  p={r['p']:.1e}",
                xy=(r["hi"], y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5)
ax.axvline(1.0, color="black", ls="--", lw=1.2)          # OR = 1 (no effect)
ax.set_yticks(ys); ax.set_yticklabels([r["label"] for r in records], fontsize=9)
ax.set_xlabel("Adjusted odds ratio per +1 SD corrected BAG (95% CI)")
ax.set_title("Covariate-adjusted association of corrected brain-age gap with diagnosis",
             fontsize=12, weight="bold")
xmax = max(r["hi"] for r in records)
ax.set_xlim(0.8, xmax * 1.35)
ax.margins(y=0.2)
plt.tight_layout(); plt.savefig(f"{PLOTS}/adni_adjusted_or_forest.png", dpi=150); plt.close()
print(f"\nSaved plot: {PLOTS}/adni_adjusted_or_forest.png")
