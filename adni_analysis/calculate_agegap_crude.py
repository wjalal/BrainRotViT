import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import make_interp_spline, interp1d
import matplotlib.pyplot as plt

def calculate_lowess_yhat_and_agegap(dfres):
    dfres_agegap = dfres.copy()
    dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["Age"]
    # dfres_agegap["AgeGap"] = dfres_agegap["AgeGap"].abs()
    return dfres_agegap

# Function to calculate MAE and R², and annotate the plot
def plot_with_metrics(data, x_col, y_col, hue_col, title, x_lim):
    # Calculate MAE and R²
    mae = mean_absolute_error(data[x_col], data[y_col])
    r2 = r2_score(data[x_col], data[y_col])
    
    # Create scatterplot
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, palette='coolwarm', hue_norm=(-12, 12))
    plt.xlim(*x_lim)
    plt.title(f"{title}\nMAE: {mae:.2f}, R²: {r2:.2f}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

# For training set
dfres_train = pd.read_csv("model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_train.csv", sep=",", index_col=0).reset_index()
dfres_train = calculate_lowess_yhat_and_agegap(dfres_train)

# # Keep only the row with the smallest Age for each SubjectID
# dfres_train = dfres_train.loc[dfres_train.groupby('SubjectID')['Age'].idxmin()]
# dfres_train = dfres_train.reset_index(drop=True)

# Plot for training set
plot_with_metrics(dfres_train, x_col="Age", y_col="Predicted_Age", hue_col="AgeGap",
                  title="Age gap predictions (Train Set)", x_lim=(40, 100))

# For validation set
dfres_val = pd.read_csv("model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv", sep=",", index_col=0).reset_index()
dfres_val = calculate_lowess_yhat_and_agegap(dfres_val)

# # Keep only the row with the smallest Age for each SubjectID
# dfres_val = dfres_val.loc[dfres_val.groupby('SubjectID')['Age'].idxmin()]
# dfres_val = dfres_val.reset_index(drop=True)

# Plot for validation set
plot_with_metrics(dfres_val, x_col="Age", y_col="Predicted_Age", hue_col="AgeGap",
                  title="Age gap predictions (Validation Set)", x_lim=(50, 100))






# Load metadata and merge (same as before)
meta = pd.read_csv("adni_brainrotnet_metadata.csv")
df_merged = dfres_val.merge(meta[["ImageID", "Group"]], on="ImageID", how="inner")

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm

# -------------------------------
# Compute threshold = 1 standard deviation
# -------------------------------
std_threshold = df_merged["AgeGap"].std()
df_merged["HighAgeGap"] = df_merged["AgeGap"] > std_threshold

# -------------------------------
# Helper function for CI
# -------------------------------
def compute_ci(a, b, c, d):
    """Return (OR, RR) with 95% CIs."""
    # Odds Ratio
    or_val = (a / b) / (c / d)
    se_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_or = np.exp(np.log(or_val) + np.array([-1, 1]) * 1.96 * se_or)

    # Relative Risk
    p1 = a / (a + b)
    p2 = c / (c + d)
    rr_val = p1 / p2
    se_rr = np.sqrt((1/a) - (1/(a + b)) + (1/c) - (1/(c + d)))
    ci_rr = np.exp(np.log(rr_val) + np.array([-1, 1]) * 1.96 * se_rr)

    return or_val, ci_or, rr_val, ci_rr

# -------------------------------
# Case 1: Outcome = AD
# -------------------------------
df_merged["IsAD"] = df_merged["Group"] == "AD"
contingency_ad = pd.crosstab(df_merged["HighAgeGap"], df_merged["IsAD"])
print("Contingency table (AD):\n", contingency_ad)

a = contingency_ad.loc[True, True]
b = contingency_ad.loc[True, False]
c = contingency_ad.loc[False, True]
d = contingency_ad.loc[False, False]

odds_ratio_ad, ci_or_ad, relative_risk_ad, ci_rr_ad = compute_ci(a, b, c, d)
chi2_ad, p_ad, _, _ = chi2_contingency(contingency_ad)
_, fisher_p_ad = fisher_exact(contingency_ad)

print(f"\nAD Odds Ratio: {odds_ratio_ad:.3f} (95% CI: {ci_or_ad[0]:.3f}–{ci_or_ad[1]:.3f})")
print(f"AD Relative Risk: {relative_risk_ad:.3f} (95% CI: {ci_rr_ad[0]:.3f}–{ci_rr_ad[1]:.3f})")
print(f"AD Chi-square p-value: {p_ad:.6f}")
print(f"AD Fisher exact p-value: {fisher_p_ad:.6f}")

# -------------------------------
# Case 2: Outcome = AD or MCI
# -------------------------------
df_merged["IsADorMCI"] = df_merged["Group"].isin(["AD", "MCI"])
contingency_admci = pd.crosstab(df_merged["HighAgeGap"], df_merged["IsADorMCI"])
print("\nContingency table (AD or MCI):\n", contingency_admci)

a = contingency_admci.loc[True, True]
b = contingency_admci.loc[True, False]
c = contingency_admci.loc[False, True]
d = contingency_admci.loc[False, False]

odds_ratio_admci, ci_or_admci, relative_risk_admci, ci_rr_admci = compute_ci(a, b, c, d)
chi2_admci, p_admci, _, _ = chi2_contingency(contingency_admci)
_, fisher_p_admci = fisher_exact(contingency_admci)

print(f"\nAD/MCI Odds Ratio: {odds_ratio_admci:.3f} (95% CI: {ci_or_admci[0]:.3f}–{ci_or_admci[1]:.3f})")
print(f"AD/MCI Relative Risk: {relative_risk_admci:.3f} (95% CI: {ci_rr_admci[0]:.3f}–{ci_rr_admci[1]:.3f})")
print(f"AD/MCI Chi-square p-value: {p_admci:.6f}")
print(f"AD/MCI Fisher exact p-value: {fisher_p_admci:.6f}")



import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

# -------------------------------
# Define threshold boundaries
# -------------------------------
std_threshold = df_merged["AgeGap"].std()
df_merged["HighAgeGap"] = df_merged["AgeGap"] > std_threshold
df_merged["LowAgeGap"]  = df_merged["AgeGap"] < -std_threshold

# Keep only extreme groups
df_extremes = df_merged[df_merged["HighAgeGap"] | df_merged["LowAgeGap"]].copy()

# -------------------------------
# Helper: Compute OR, RR, and 95% CI
# -------------------------------
def compute_ci(a, b, c, d):
    """Return (OR, RR) with 95% CIs."""
    # Odds Ratio
    or_val = (a / b) / (c / d)
    se_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_or = np.exp(np.log(or_val) + np.array([-1, 1]) * 1.96 * se_or)

    # Relative Risk
    p1 = a / (a + b)
    p2 = c / (c + d)
    rr_val = p1 / p2
    se_rr = np.sqrt((1/a) - (1/(a + b)) + (1/c) - (1/(c + d)))
    ci_rr = np.exp(np.log(rr_val) + np.array([-1, 1]) * 1.96 * se_rr)

    return or_val, ci_or, rr_val, ci_rr


# -------------------------------
# Case 1: Outcome = AD
# -------------------------------
df_extremes["IsAD"] = df_extremes["Group"] == "AD"

contingency_ad = pd.crosstab(df_extremes["HighAgeGap"], df_extremes["IsAD"])
print("Contingency table (AD, High vs Low AgeGap):\n", contingency_ad)

a = contingency_ad.loc[True, True]   # HighAgeGap & AD
b = contingency_ad.loc[True, False]  # HighAgeGap & not AD
c = contingency_ad.loc[False, True]  # LowAgeGap & AD
d = contingency_ad.loc[False, False] # LowAgeGap & not AD

odds_ratio_ad, ci_or_ad, relative_risk_ad, ci_rr_ad = compute_ci(a, b, c, d)
chi2_ad, p_ad, _, _ = chi2_contingency(contingency_ad)
_, fisher_p_ad = fisher_exact(contingency_ad)

print(f"\nAD Odds Ratio (High vs Low AgeGap): {odds_ratio_ad:.3f} (95% CI: {ci_or_ad[0]:.3f}–{ci_or_ad[1]:.3f})")
print(f"AD Relative Risk (High vs Low AgeGap): {relative_risk_ad:.3f} (95% CI: {ci_rr_ad[0]:.3f}–{ci_rr_ad[1]:.3f})")
print(f"AD Chi-square p-value: {p_ad:.6f}")
print(f"AD Fisher exact p-value: {fisher_p_ad:.6f}")


# -------------------------------
# Case 2: Outcome = AD or MCI
# -------------------------------
df_extremes["IsADorMCI"] = df_extremes["Group"].isin(["AD", "MCI"])

contingency_admci = pd.crosstab(df_extremes["HighAgeGap"], df_extremes["IsADorMCI"])
print("\nContingency table (AD or MCI, High vs Low AgeGap):\n", contingency_admci)

a = contingency_admci.loc[True, True]
b = contingency_admci.loc[True, False]
c = contingency_admci.loc[False, True]
d = contingency_admci.loc[False, False]

odds_ratio_admci, ci_or_admci, relative_risk_admci, ci_rr_admci = compute_ci(a, b, c, d)
chi2_admci, p_admci, _, _ = chi2_contingency(contingency_admci)
_, fisher_p_admci = fisher_exact(contingency_admci)

print(f"\nAD/MCI Odds Ratio (High vs Low AgeGap): {odds_ratio_admci:.3f} (95% CI: {ci_or_admci[0]:.3f}–{ci_or_admci[1]:.3f})")
print(f"AD/MCI Relative Risk (High vs Low AgeGap): {relative_risk_admci:.3f} (95% CI: {ci_rr_admci[0]:.3f}–{ci_rr_admci[1]:.3f})")
print(f"AD/MCI Chi-square p-value: {p_admci:.6f}")
print(f"AD/MCI Fisher exact p-value: {fisher_p_admci:.6f}")



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

# -------------------------------
# Compute thresholds for AgeGap
# -------------------------------
std_threshold = df_merged["AgeGap"].std()
df_merged["HighAgeGap"] = df_merged["AgeGap"] > std_threshold
df_merged["LowAgeGap"] = df_merged["AgeGap"] < -std_threshold

# -------------------------------
# Case 1: AD/MCI vs CN for HighAgeGap (> +1 SD)
# -------------------------------
df_merged["IsADorMCI"] = df_merged["Group"].isin(["AD", "MCI"])
contingency_high = pd.crosstab(df_merged["HighAgeGap"], df_merged["IsADorMCI"])

# Convert to proportions for visualization
prop_high = contingency_high.div(contingency_high.sum(axis=1), axis=0).reset_index()
prop_high = prop_high.melt(id_vars="HighAgeGap", var_name="IsADorMCI", value_name="Proportion")

# Map labels
prop_high["HighAgeGap"] = prop_high["HighAgeGap"].map({True: "High Age-Gap (> +1 SD)", False: "Low/Normal Age-Gap"})
prop_high["IsADorMCI"] = prop_high["IsADorMCI"].map({True: "AD/MCI", False: "CN"})

# -------------------------------
# Case 2: AD/MCI vs CN for Extreme Comparison (> +1 SD vs < -1 SD)
# -------------------------------
extreme_df = df_merged[(df_merged["HighAgeGap"]) | (df_merged["LowAgeGap"])].copy()
extreme_df["ExtremeGroup"] = np.where(extreme_df["HighAgeGap"], "High Age-Gap (> +1 SD)", "Low Age-Gap (< -1 SD)")

contingency_extreme = pd.crosstab(extreme_df["ExtremeGroup"], extreme_df["IsADorMCI"])

# Convert to proportions
prop_extreme = contingency_extreme.div(contingency_extreme.sum(axis=1), axis=0).reset_index()
prop_extreme = prop_extreme.melt(id_vars="ExtremeGroup", var_name="IsADorMCI", value_name="Proportion")
prop_extreme["IsADorMCI"] = prop_extreme["IsADorMCI"].map({True: "AD/MCI", False: "CN"})

# -------------------------------
# Plot 1: High vs Low/Normal
# -------------------------------
plt.figure(figsize=(6, 5))
sns.barplot(
    data=prop_high,
    x="HighAgeGap",
    y="Proportion",
    hue="IsADorMCI",
    palette=["#b3cde0", "#005b96"]
)
plt.title("Proportion of AD/MCI vs CN by Age-Gap Group", fontsize=14, weight="bold")
plt.xlabel("")
plt.ylabel("Proportion", fontsize=12)
plt.legend(title="Diagnosis", loc="upper right")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# -------------------------------
# Plot 2: High (> +1 SD) vs Low (< -1 SD)
# -------------------------------
plt.figure(figsize=(6, 5))
sns.barplot(
    data=prop_extreme,
    x="ExtremeGroup",
    y="Proportion",
    hue="IsADorMCI",
    palette=["#b3cde0", "#005b96"]
)
plt.title("Proportion of AD/MCI vs CN by Extreme Age-Gap Groups", fontsize=14, weight="bold")
plt.xlabel("")
plt.ylabel("Proportion", fontsize=12)
plt.legend(title="Diagnosis", loc="upper right")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()





# Load metadata and merge
meta = pd.read_csv("adni_brainrotnet_metadata.csv")
df_merged = dfres_val.merge(meta[["ImageID", "Group"]], on="ImageID", how="inner")

# Compute std threshold
std_val = df_merged["AgeGap"].std()

# Create groups
high_group = df_merged[df_merged["AgeGap"] > std_val]
low_group  = df_merged[df_merged["AgeGap"] < -std_val]

print()
# -------------------------------
# Case 1: Proportion AD
# -------------------------------
prop_high_ad = (high_group["Group"] == "AD").mean()
prop_low_ad  = (low_group["Group"] == "AD").mean()
ratio_ad = prop_high_ad / prop_low_ad if prop_low_ad > 0 else float("inf")

print(f"Proportion AD in High AgeGap (>+1SD): {prop_high_ad:.3f}")
print(f"Proportion AD in Low AgeGap (<-1SD): {prop_low_ad:.3f}")
print(f"Relative likelihood (High vs Low): {ratio_ad:.2f}x")

# -------------------------------
# Case 2: Proportion AD or MCI
# -------------------------------
prop_high_admci = high_group["Group"].isin(["AD","MCI"]).mean()
prop_low_admci  = low_group["Group"].isin(["AD","MCI"]).mean()
ratio_admci = prop_high_admci / prop_low_admci if prop_low_admci > 0 else float("inf")

print(f"\nProportion AD/MCI in High AgeGap (>+1SD): {prop_high_admci:.3f}")
print(f"Proportion AD/MCI in Low AgeGap (<-1SD): {prop_low_admci:.3f}")
print(f"Relative likelihood (High vs Low): {ratio_admci:.2f}x")



import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Counts for AD only
n_high_ad = (high_group["Group"] == "AD").sum()
n_high_total = len(high_group)
n_low_ad = (low_group["Group"] == "AD").sum()
n_low_total = len(low_group)

# Proportion comparison (AD)
count = np.array([n_high_ad, n_low_ad])
nobs = np.array([n_high_total, n_low_total])

stat, pval = proportions_ztest(count, nobs)
print(f"P-value (difference in AD proportions, High vs Low AgeGap): {pval:.5f}")

# Counts for AD/MCI combined
n_high_admci = high_group["Group"].isin(["AD","MCI"]).sum()
n_low_admci = low_group["Group"].isin(["AD","MCI"]).sum()

count_admci = np.array([n_high_admci, n_low_admci])
nobs_admci = np.array([n_high_total, n_low_total])

stat_admci, pval_admci = proportions_ztest(count_admci, nobs_admci)
print(f"P-value (difference in AD/MCI proportions, High vs Low AgeGap): {pval_admci:.5f}")


import matplotlib.pyplot as plt
import pandas as pd

# (using df_merged, std_val, high_group, low_group from earlier code)

# Compute proportions
prop_high_ad = (high_group["Group"] == "AD").mean()
prop_low_ad  = (low_group["Group"] == "AD").mean()

prop_high_admci = high_group["Group"].isin(["AD","MCI"]).mean()
prop_low_admci  = low_group["Group"].isin(["AD","MCI"]).mean()

# Data for plotting
groups = ["AD", "AD/MCI"]
high_props = [prop_high_ad, prop_high_admci]
low_props  = [prop_low_ad, prop_low_admci]

x = range(len(groups))
width = 0.35

# Plot
fig, ax = plt.subplots(figsize=(7,5))

bars1 = ax.bar([i - width/2 for i in x], low_props, width, label="AgeGap < -1 SD", color="steelblue")
bars2 = ax.bar([i + width/2 for i in x], high_props, width, label="AgeGap > +1 SD", color="salmon")

# Labels and formatting
ax.set_ylabel("Proportion of Subjects")
ax.set_title("Proportion of AD and AD/MCI by AgeGap group")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

# Annotate bars with percentages
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,3),  # offset above bar
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt

# Compute std dev threshold
std_val = df_merged["AgeGap"].std()

# Simplify groups
df_merged["GroupSimplified"] = df_merged["Group"].replace({
    "AD": "AD",
    "MCI": "MCI",
    "CN": "CN"
})

fig, ax = plt.subplots(figsize=(10,6))

# AD subjects (red)
subset = df_merged[df_merged["GroupSimplified"] == "AD"]
ax.scatter(subset["AgeGap"], subset["Age"],
           color="darkred", label="AD", alpha=0.8, s=30)

# MCI subjects (orange)
subset = df_merged[df_merged["GroupSimplified"] == "MCI"]
ax.scatter(subset["AgeGap"], subset["Age"],
           color="orange", label="MCI", alpha=0.8, s=30)

# CN subjects (light gray fill + darker gray stroke)
subset = df_merged[df_merged["GroupSimplified"] == "CN"]
ax.scatter(subset["AgeGap"], subset["Age"],
           facecolors=(0.8, 0.8, 0.8),  # 2% gray fill
           edgecolors=(0.05, 0.05, 0.05), # 5% gray stroke
           alpha = 0.1,
           linewidth=0.7, s=30, label="CN")

# Add ±1 SD cutoff bars (vertical lines)
ax.axvline(std_val, color="black", linestyle="--", linewidth=1.5, label="+1 SD")
ax.axvline(-std_val, color="black", linestyle="--", linewidth=1.5, label="−1 SD")

# Formatting
ax.set_xlabel("AgeGap")
ax.set_ylabel("Age")
ax.set_title("Subjects by AgeGap and Real Age with ±1 SD Cutoffs")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
