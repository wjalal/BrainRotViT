"""
validation_metrics.py

Reviewer R5.8 (point d): global Pearson r / R2 are inflated by a wide age range
and do not reflect clinical precision. This script reports precision metrics that
do NOT depend on age spread, for the ADNI VALIDATION set:

  1. MAE (with bootstrap 95% CI) stratified by
        - age decade
        - sex
        - diagnostic group (CN / MCI / AD)
        - acquisition site  -- the within-ADNI analog of "by dataset"
                               (ADNI is a single dataset; site = scanner/source
                               of variance. Sites with <MIN_SITE_N val scans are
                               pooled as "OTHER").
     Saved as a supplementary table (CSV) + a Figure-5-style panel of bar charts.

  2. Bland-Altman plot: (predicted - chronological) vs mean of the two, with bias
     and 95% limits of agreement.

  3. Residual (predicted - chronological) vs chronological age, with linear and
     LOWESS trends and a zero reference -- exposes any age-dependent bias.

All figures are written to disk (headless Agg); nothing requires a display.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess

PRED_VAL = "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv"
META = "adni_brainrotnet_metadata.csv"
PLOT_DIR = "model_dumps/mix/plots"
TABLE_OUT = "model_dumps/mix/adni_val_mae_stratified.csv"
MIN_SITE_N = 20          # sites with fewer val scans are pooled into OTHER
N_BOOT = 2000
RNG = np.random.RandomState(69420)
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load predictions + attach Group and site
# ---------------------------------------------------------------------------
df = pd.read_csv(PRED_VAL)
meta = pd.read_csv(META)
meta["site"] = meta["SubjectID"].astype(str).str.extract(r"^(\d+)_S_")[0]
df = df.merge(meta[["ImageID", "Group", "site"]], on="ImageID", how="left")

df["Predicted_Age"] = df["Predicted_Age"].astype(float)
df["Age"] = df["Age"].astype(float)
df["residual"] = df["Predicted_Age"] - df["Age"]          # signed error (BAG)
df["abs_err"] = df["residual"].abs()
df["decade"] = (df["Age"] // 10 * 10).astype(int)
site_counts = df["site"].value_counts()
df["site_grp"] = np.where(df["site"].isin(site_counts[site_counts >= MIN_SITE_N].index),
                          df["site"], "OTHER")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def boot_mae_ci(abs_err, n_boot=N_BOOT):
    a = np.asarray(abs_err, float)
    if len(a) < 2:
        return a.mean() if len(a) else np.nan, np.nan, np.nan
    idx = RNG.randint(0, len(a), size=(n_boot, len(a)))
    boots = a[idx].mean(axis=1)
    return a.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def stratum_row(kind, name, g):
    mae, lo, hi = boot_mae_ci(g["abs_err"].values)
    return {
        "stratum_type": kind,
        "stratum": str(name),
        "n": len(g),
        "MAE": round(mae, 3),
        "MAE_CI_lo": round(lo, 3) if not np.isnan(lo) else np.nan,
        "MAE_CI_hi": round(hi, 3) if not np.isnan(hi) else np.nan,
        "RMSE": round(float(np.sqrt((g["residual"] ** 2).mean())), 3),
        "bias": round(float(g["residual"].mean()), 3),   # mean signed error
    }


# ---------------------------------------------------------------------------
# Overall reference metrics
# ---------------------------------------------------------------------------
overall_mae = mean_absolute_error(df["Age"], df["Predicted_Age"])
overall_r2 = r2_score(df["Age"], df["Predicted_Age"])
overall_r = pearsonr(df["Age"], df["Predicted_Age"])[0]
overall_rho = spearmanr(df["Age"], df["Predicted_Age"])[0]
print(f"=== ADNI validation set (n={len(df)}, age {df.Age.min():.1f}-{df.Age.max():.1f}) ===")
print(f"Overall  MAE {overall_mae:.3f}  RMSE {np.sqrt((df.residual**2).mean()):.3f}  "
      f"R2 {overall_r2:.3f}  Pearson {overall_r:.3f}  Spearman {overall_rho:.3f}")

# ---------------------------------------------------------------------------
# Stratified MAE table
# ---------------------------------------------------------------------------
rows = [stratum_row("overall", "all", df)]
for dec, g in df.groupby("decade"):
    rows.append(stratum_row("age_decade", f"{dec}s", g))
for sx, g in df.groupby("Sex"):
    rows.append(stratum_row("sex", sx, g))
for grp, g in df.groupby("Group"):
    rows.append(stratum_row("diagnosis", grp, g))
for st, g in df.groupby("site_grp"):
    rows.append(stratum_row("site", st, g))

table = pd.DataFrame(rows)
table.to_csv(TABLE_OUT, index=False)
print(f"\nStratified MAE table -> {TABLE_OUT}")
with pd.option_context("display.width", 120):
    print(table.to_string(index=False))


# ---------------------------------------------------------------------------
# Figure-5-style panel: MAE bar charts by decade / sex / diagnosis / site
# ---------------------------------------------------------------------------
def bar_panel(ax, sub, title, order=None):
    sub = sub.copy()
    if order is not None:
        sub["stratum"] = pd.Categorical(sub["stratum"], categories=order, ordered=True)
        sub = sub.sort_values("stratum")
    else:
        sub = sub.sort_values("MAE")
    x = np.arange(len(sub))
    yerr = np.vstack([sub["MAE"] - sub["MAE_CI_lo"], sub["MAE_CI_hi"] - sub["MAE"]])
    ax.bar(x, sub["MAE"], color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.errorbar(x, sub["MAE"], yerr=yerr, fmt="none", ecolor="black", capsize=3, lw=0.8)
    ax.axhline(overall_mae, color="crimson", ls="--", lw=1, label=f"overall {overall_mae:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["stratum"], rotation=45, ha="right", fontsize=8)
    for xi, (mae, n) in enumerate(zip(sub["MAE"], sub["n"])):
        ax.text(xi, mae, f"n={n}", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("MAE (years)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
bar_panel(axes[0, 0], table[table.stratum_type == "age_decade"], "(a) MAE by age decade",
          order=[f"{d}s" for d in sorted(df["decade"].unique())])
bar_panel(axes[0, 1], table[table.stratum_type == "sex"], "(b) MAE by sex")
bar_panel(axes[1, 0], table[table.stratum_type == "diagnosis"], "(c) MAE by diagnosis",
          order=["CN", "MCI", "AD"])
bar_panel(axes[1, 1], table[table.stratum_type == "site"], "(d) MAE by acquisition site")
fig.suptitle("ADNI validation MAE stratified (bars: 95% bootstrap CI)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{PLOT_DIR}/adni_val_mae_stratified.png", dpi=150)
plt.close(fig)
print(f"Panel figure -> {PLOT_DIR}/adni_val_mae_stratified.png")


# ---------------------------------------------------------------------------
# Bland-Altman: (predicted - chronological) vs mean of the two
# ---------------------------------------------------------------------------
mean_age = (df["Predicted_Age"] + df["Age"]) / 2.0
diff = df["residual"]
bias = diff.mean()
sd = diff.std(ddof=1)
loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(mean_age, diff, s=14, alpha=0.4, color="#4C72B0", edgecolors="none")
ax.axhline(bias, color="crimson", lw=1.5, label=f"bias {bias:+.2f}")
ax.axhline(loa_hi, color="gray", ls="--", lw=1.2, label=f"+1.96 SD {loa_hi:+.2f}")
ax.axhline(loa_lo, color="gray", ls="--", lw=1.2, label=f"-1.96 SD {loa_lo:+.2f}")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("Mean of predicted & chronological age (years)")
ax.set_ylabel("Predicted - chronological age (years)")
ax.set_title("Bland-Altman: brain-age prediction vs chronological age (ADNI val)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/adni_val_bland_altman.png", dpi=150)
plt.close(fig)
print(f"Bland-Altman -> {PLOT_DIR}/adni_val_bland_altman.png  "
      f"(bias {bias:+.3f}, LoA [{loa_lo:+.3f}, {loa_hi:+.3f}])")


# ---------------------------------------------------------------------------
# Residual vs chronological age (linear + LOWESS trend)
# ---------------------------------------------------------------------------
order = np.argsort(df["Age"].values)
age_s = df["Age"].values[order]
res_s = df["residual"].values[order]
slope, intercept = np.polyfit(age_s, res_s, 1)
low = lowess(res_s, age_s, frac=0.5, return_sorted=True)
r_res_age = pearsonr(df["Age"], df["residual"])[0]

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(age_s, res_s, s=14, alpha=0.4, color="#55A868", edgecolors="none")
ax.plot(age_s, slope * age_s + intercept, color="crimson", lw=1.8,
        label=f"linear fit (slope {slope:+.3f}/yr)")
ax.plot(low[:, 0], low[:, 1], color="navy", lw=1.8, label="LOWESS")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Chronological age (years)")
ax.set_ylabel("Residual: predicted - chronological (years)")
ax.set_title(f"Residual vs age (ADNI val)   corr(residual, age) = {r_res_age:+.3f}")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/adni_val_residual_vs_age.png", dpi=150)
plt.close(fig)
print(f"Residual-vs-age -> {PLOT_DIR}/adni_val_residual_vs_age.png  "
      f"(slope {slope:+.4f}/yr, corr {r_res_age:+.3f})")
print("\nDone.")
