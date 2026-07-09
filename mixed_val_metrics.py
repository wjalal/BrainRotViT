"""
mixed_val_metrics.py

Reviewer response: Pearson r / R^2 over a wide chronological age range can mask
clinically meaningful error. This script reports age-range-independent accuracy
for the pooled mixed-cohort VALIDATION set:

  * MAE (with bootstrap 95% CI) stratified by age decade, age range, sex, and
    source dataset (each source dataset is a distinct acquisition/scanner cohort,
    so the by-dataset breakdown is the scanner/site analysis for the pooled
    model). A diagnosis breakdown is added where a clinical label exists in the
    pooled metadata (ABIDE); the remaining cohorts are unlabelled controls, and
    disorder-specific analyses are reported in the ADNI/ABIDE sections.
  * Calibration plot (predicted vs chronological age + binned calibration curve).
  * Residual distribution and residual-vs-age plot.
  * Bland-Altman analysis (bias + 95% limits of agreement).
  * Age-bias-corrected brain-age gap (de Lange & Cole, 2022): the BAG~age trend
    is fit on the TRAINING set and removed from the validation BAG.

All figures/tables are written under model_dumps/mix/ ; nothing is overwritten
in place. Everything is headless (Agg) -- no plt.show().
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

VAL = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv"
TRAIN = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_train.csv"
PLOTS = "model_dumps/mix/plots"
TABLE = "model_dumps/mix/mixed_val_stratified_mae.csv"
N_BOOT = 2000
RNG = np.random.RandomState(69420)
os.makedirs(PLOTS, exist_ok=True)

DATASET_NAME = {
    "adni": "ADNI", "ixi": "IXI", "abide": "ABIDE", "dlbs": "DLBS", "cobre": "COBRE",
    "fcon1000": "FCON1000", "corr": "CoRR", "oasis1": "OASIS-1", "camcan": "CamCAN",
    "nimh": "NIMH", "bold": "BOLD",
}


def dataset_of(fp):
    m = re.search(r'([A-Za-z0-9]+)_storage/', str(fp))
    return DATASET_NAME.get(m.group(1).lower(), m.group(1)) if m else "?"


def load(path):
    d = pd.read_csv(path)
    if d.columns[0].startswith("Unnamed") or d.columns[0] == "":
        d = d.drop(columns=d.columns[0])
    d["dataset"] = d["filepath"].apply(dataset_of)
    d["Predicted_Age"] = d["Predicted_Age"].astype(float)
    d["Age"] = d["Age"].astype(float)
    d["residual"] = d["Predicted_Age"] - d["Age"]        # raw BAG
    d["abs_err"] = d["residual"].abs()
    return d


val = load(VAL)
train = load(TRAIN) if os.path.exists(TRAIN) else None

# ---------------------------------------------------------------------------
# Age-bias correction (de Lange & Cole 2022): fit BAG~age on TRAIN, apply to VAL
# ---------------------------------------------------------------------------
if train is not None:
    X = sm.add_constant(train["Age"].to_numpy())
    ols = sm.OLS(train["residual"].to_numpy(), X).fit()
    intercept, slope = ols.params[0], ols.params[1]
    ref_desc = f"training set (n={len(train)})"
else:
    X = sm.add_constant(val["Age"].to_numpy())
    ols = sm.OLS(val["residual"].to_numpy(), X).fit()
    intercept, slope = ols.params[0], ols.params[1]
    ref_desc = f"validation set self-fit (n={len(val)}); TRAIN csv not found"
val["BAG_corrected"] = val["residual"] - (slope * val["Age"] + intercept)

# ---------------------------------------------------------------------------
# Age strata
# ---------------------------------------------------------------------------
# Age decades, with 90+ collapsed into a single "80+" group.
val["decade"] = np.where(val["Age"] >= 80, "80+",
                         (val["Age"] // 10 * 10).astype(int).astype(str) + "s")
range_edges = [0, 20, 40, 60, 80, 200]
range_lbl = ["<20", "20-40", "40-60", "60-80", "80+"]
val["age_range"] = pd.cut(val["Age"], bins=range_edges, labels=range_lbl, right=False)

# Diagnosis join for the two pooled cohorts that carry a clinical label:
# ABIDE (ASD/Control) and ADNI (CN/MCI/AD).
val["diagnosis"] = pd.Series(pd.NA, index=val.index, dtype="object")
val["diag_cohort"] = pd.Series(pd.NA, index=val.index, dtype="object")
_diag_sources = [
    ("ABIDE", "abide_storage/abide_brainrotnet_metadata.csv", "Diag"),
    ("ADNI", "adni_storage/adni_brainrotnet_metadata.csv", "Group"),
]
for cohort, path, col in _diag_sources:
    if not os.path.exists(path):
        continue
    meta = pd.read_csv(path)[["ImageID", col]].rename(columns={col: "_diag"})
    val = val.merge(meta, on="ImageID", how="left")
    m = (val["dataset"] == cohort) & val["_diag"].notna()
    val.loc[m, "diagnosis"] = val.loc[m, "_diag"]
    val.loc[m, "diag_cohort"] = cohort
    val = val.drop(columns=["_diag"])


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def boot_mae_ci(abs_err):
    a = np.asarray(abs_err, float)
    if len(a) < 2:
        return (a.mean() if len(a) else np.nan), np.nan, np.nan
    idx = RNG.randint(0, len(a), size=(N_BOOT, len(a)))
    boots = a[idx].mean(axis=1)
    return a.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def full_metrics(g):
    y, p = g["Age"].to_numpy(), g["Predicted_Age"].to_numpy()
    mae, lo, hi = boot_mae_ci(g["abs_err"].to_numpy())
    out = {
        "n": len(g), "MAE": mae, "MAE_lo": lo, "MAE_hi": hi,
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "bias": float(np.mean(p - y)),
        "Pearson": pearsonr(y, p)[0] if len(g) > 1 else np.nan,
        "Spearman": spearmanr(y, p).correlation if len(g) > 1 else np.nan,
    }
    return out


def stratum_table(kind, col):
    rows = []
    for name, g in val.groupby(col, observed=True):
        if len(g) == 0:
            continue
        rows.append({"stratum_type": kind, "stratum": str(name), **full_metrics(g)})
    return rows


# ---------------------------------------------------------------------------
# Overall + stratified tables
# ---------------------------------------------------------------------------
overall = full_metrics(val)
print("=" * 68)
print(f"POOLED MIXED VALIDATION SET   n={overall['n']}   "
      f"age {val.Age.min():.1f}-{val.Age.max():.1f}")
print(f"Overall  MAE {overall['MAE']:.3f} [{overall['MAE_lo']:.3f}, {overall['MAE_hi']:.3f}]  "
      f"RMSE {overall['RMSE']:.3f}  bias {overall['bias']:+.3f}  "
      f"Pearson {overall['Pearson']:.3f}  Spearman {overall['Spearman']:.3f}")
print(f"Age-bias correction fit on {ref_desc}: slope {slope:+.4f}, intercept {intercept:+.4f}")
r_raw = np.corrcoef(val["residual"], val["Age"])[0, 1]
r_cor = np.corrcoef(val["BAG_corrected"], val["Age"])[0, 1]
print(f"corr(BAG, age): raw {r_raw:+.3f} -> corrected {r_cor:+.3f}")

rows = [{"stratum_type": "overall", "stratum": "all", **overall}]
rows += stratum_table("age_decade", "decade")
rows += stratum_table("age_range", "age_range")
rows += stratum_table("sex", "Sex")
rows += stratum_table("dataset", "dataset")
for cohort in ["ABIDE", "ADNI"]:
    sub = val[(val["diag_cohort"] == cohort) & val["diagnosis"].notna()]
    for name, g in sub.groupby("diagnosis"):
        rows.append({"stratum_type": f"diagnosis({cohort})", "stratum": str(name), **full_metrics(g)})

table = pd.DataFrame(rows)
for c in ["MAE", "MAE_lo", "MAE_hi", "RMSE", "bias", "Pearson", "Spearman"]:
    table[c] = table[c].round(3)
table.to_csv(TABLE, index=False)
print(f"\nStratified table -> {TABLE}")
print(table.to_string(index=False))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _bar(ax, sub, title, order=None):
    sub = sub.copy()
    if order is not None:
        sub["stratum"] = pd.Categorical(sub["stratum"], categories=order, ordered=True)
        sub = sub.sort_values("stratum")
    x = np.arange(len(sub))
    yerr = np.vstack([sub["MAE"] - sub["MAE_lo"], sub["MAE_hi"] - sub["MAE"]])
    ax.bar(x, sub["MAE"], color="#4C72B0", edgecolor="black", lw=0.5)
    ax.errorbar(x, sub["MAE"], yerr=yerr, fmt="none", ecolor="black", capsize=3, lw=0.8)
    ax.axhline(overall["MAE"], color="crimson", ls="--", lw=1, label=f"overall {overall['MAE']:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(sub["stratum"], rotation=45, ha="right", fontsize=8)
    for xi, (m, n) in enumerate(zip(sub["MAE"], sub["n"])):
        ax.text(xi, m, f"{n}", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("MAE (years)"); ax.set_title(title, fontsize=10); ax.legend(fontsize=7)


# (1) Stratified MAE panel
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
_bar(ax[0, 0], table[table.stratum_type == "age_decade"], "(a) MAE by age decade",
     order=sorted(table[table.stratum_type == "age_decade"]["stratum"], key=lambda s: int(s[:-1])))
_bar(ax[0, 1], table[table.stratum_type == "age_range"], "(b) MAE by age range", order=range_lbl)
_bar(ax[1, 0], table[table.stratum_type == "sex"], "(c) MAE by sex")
_bar(ax[1, 1], table[table.stratum_type == "dataset"].sort_values("MAE"),
     "(d) MAE by dataset (scanner/site cohort)")
fig.suptitle("Pooled mixed-cohort validation MAE, stratified (bars: 95% bootstrap CI)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{PLOTS}/mixed_val_mae_stratified.png", dpi=150); plt.close(fig)

# (1b) Diagnosis MAE: ABIDE (ASD/Control) and ADNI (CN/MCI/AD)
_diag_order = {"ABIDE": ["Control", "ASD"], "ADNI": ["CN", "MCI", "AD"]}
present = [c for c in ["ABIDE", "ADNI"]
           if (table["stratum_type"] == f"diagnosis({c})").any()]
if present:
    figd, axd = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False)
    for j, cohort in enumerate(present):
        _bar(axd[0, j], table[table.stratum_type == f"diagnosis({cohort})"],
             f"({chr(97+j)}) MAE by diagnosis — {cohort}", order=_diag_order.get(cohort))
    figd.suptitle("Validation MAE by diagnosis (95% bootstrap CI)", fontsize=13)
    figd.tight_layout(rect=[0, 0, 1, 0.95])
    figd.savefig(f"{PLOTS}/mixed_val_mae_diagnosis.png", dpi=150); plt.close(figd)

# (2) Calibration: predicted vs chronological age + binned calibration curve
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].scatter(val["Age"], val["Predicted_Age"], s=10, alpha=0.35, color="#4C72B0", edgecolors="none")
lims = [val["Age"].min() - 2, val["Age"].max() + 2]
ax[0].plot(lims, lims, "k--", lw=1, label="identity")
b, a = np.polyfit(val["Age"], val["Predicted_Age"], 1)
ax[0].plot(np.array(lims), b * np.array(lims) + a, color="crimson", lw=1.6,
           label=f"fit (slope {b:.2f})")
ax[0].set_xlabel("Chronological age"); ax[0].set_ylabel("Predicted age")
ax[0].set_title("(a) Calibration: predicted vs chronological"); ax[0].legend(fontsize=8)
# binned calibration curve
bins = np.arange(np.floor(val.Age.min() / 5) * 5, val.Age.max() + 5, 5)
val["_ab"] = pd.cut(val["Age"], bins)
gb = val.groupby("_ab", observed=True)
centers = [iv.mid for iv in gb.groups.keys()]
mean_pred = gb["Predicted_Age"].mean().to_numpy()
sd_pred = gb["Predicted_Age"].std().to_numpy()
ax[1].plot(lims, lims, "k--", lw=1, label="identity")
ax[1].errorbar(centers, mean_pred, yerr=sd_pred, fmt="o-", color="#55A868", capsize=3,
               label="mean predicted ± SD")
ax[1].set_xlabel("Chronological age (5-yr bins)"); ax[1].set_ylabel("Predicted age")
ax[1].set_title("(b) Binned calibration curve"); ax[1].legend(fontsize=8)
val = val.drop(columns=["_ab"])
fig.tight_layout(); fig.savefig(f"{PLOTS}/mixed_val_calibration.png", dpi=150); plt.close(fig)

# (3) Residual distribution + residual vs age
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].hist(val["residual"], bins=50, color="#4C72B0", edgecolor="black")
ax[0].axvline(val["residual"].mean(), color="crimson", ls="--",
              label=f"mean {val['residual'].mean():+.2f}")
ax[0].axvline(0, color="black", lw=0.8)
ax[0].set_xlabel("Residual (predicted − chronological), yr"); ax[0].set_ylabel("count")
ax[0].set_title("(a) Residual distribution"); ax[0].legend(fontsize=8)
order = np.argsort(val["Age"].to_numpy())
xa = val["Age"].to_numpy()[order]; ra = val["residual"].to_numpy()[order]
sl, ic = np.polyfit(xa, ra, 1)
lo = lowess(ra, xa, frac=0.5, return_sorted=True)
ax[1].scatter(xa, ra, s=10, alpha=0.35, color="#55A868", edgecolors="none")
ax[1].plot(xa, sl * xa + ic, color="crimson", lw=1.6, label=f"linear (slope {sl:+.3f}/yr)")
ax[1].plot(lo[:, 0], lo[:, 1], color="navy", lw=1.6, label="LOWESS")
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_xlabel("Chronological age"); ax[1].set_ylabel("Residual (yr)")
ax[1].set_title(f"(b) Residual vs age  (corr {r_raw:+.2f})"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{PLOTS}/mixed_val_residuals.png", dpi=150); plt.close(fig)

# (4) Bland-Altman
mean_meas = (val["Predicted_Age"] + val["Age"]) / 2
diff = val["residual"]
bias = diff.mean(); sd = diff.std(ddof=1)
lo_loa, hi_loa = bias - 1.96 * sd, bias + 1.96 * sd
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.scatter(mean_meas, diff, s=12, alpha=0.35, color="#4C72B0", edgecolors="none")
ax.axhline(bias, color="crimson", lw=1.5, label=f"bias {bias:+.2f}")
ax.axhline(hi_loa, color="gray", ls="--", lw=1.2, label=f"+1.96 SD {hi_loa:+.2f}")
ax.axhline(lo_loa, color="gray", ls="--", lw=1.2, label=f"−1.96 SD {lo_loa:+.2f}")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("Mean of predicted & chronological age (yr)")
ax.set_ylabel("Predicted − chronological (yr)")
ax.set_title("Bland-Altman: brain-age prediction (pooled validation)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{PLOTS}/mixed_val_bland_altman.png", dpi=150); plt.close(fig)

# (5) Age-bias-corrected BAG: residual vs age, raw vs corrected
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for a_, y_, ttl, col in [(ax[0], val["residual"], f"(a) Raw BAG (corr {r_raw:+.2f})", "#55A868"),
                         (ax[1], val["BAG_corrected"], f"(b) Corrected BAG (corr {r_cor:+.2f})", "#C44E52")]:
    a_.scatter(val["Age"], y_, s=10, alpha=0.35, color=col, edgecolors="none")
    sll, icc = np.polyfit(val["Age"], y_, 1)
    a_.plot(np.array(lims), sll * np.array(lims) + icc, color="black", lw=1.4)
    a_.axhline(0, color="black", lw=0.7, ls=":")
    a_.set_xlabel("Chronological age"); a_.set_title(ttl)
ax[0].set_ylabel("Brain-age gap (yr)")
fig.suptitle("Age-bias correction (de Lange & Cole 2022): BAG vs age", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{PLOTS}/mixed_val_bag_correction.png", dpi=150); plt.close(fig)

print(f"\nBland-Altman: bias {bias:+.3f}, LoA [{lo_loa:+.3f}, {hi_loa:+.3f}]")
print("Saved figures -> " + ", ".join([
    "mixed_val_mae_stratified.png", "mixed_val_calibration.png", "mixed_val_residuals.png",
    "mixed_val_bland_altman.png", "mixed_val_bag_correction.png"]))
