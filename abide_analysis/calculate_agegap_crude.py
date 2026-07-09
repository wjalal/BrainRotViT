import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Helper: Calculate LOWESS and AgeGap
# -------------------------------
def calculate_lowess_yhat_and_agegap(dfres):
    dfres_agegap = dfres.copy()
    dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["Age"]
    return dfres_agegap


# -------------------------------
# Helper: Plot with MAE and R²
# -------------------------------
def plot_with_metrics(data, x_col, y_col, hue_col, title, x_lim):
    mae = mean_absolute_error(data[x_col], data[y_col])
    r2 = r2_score(data[x_col], data[y_col])
    
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col,
                    palette='coolwarm', hue_norm=(-5, 5))
    plt.xlim(*x_lim)
    plt.title(f"{title}\nMAE: {mae:.2f}, R²: {r2:.2f}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()


# -------------------------------
# Training set
# -------------------------------
dfres_train = pd.read_csv(
    "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_train.csv",
    sep=",", index_col=0
).reset_index()
dfres_train = calculate_lowess_yhat_and_agegap(dfres_train)
dfres_train["Split"] = "Train"  # optional column for tracking

# -------------------------------
# Validation set
# -------------------------------
dfres_val = pd.read_csv(
    "model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv",
    sep=",", index_col=0
).reset_index()
dfres_val = calculate_lowess_yhat_and_agegap(dfres_val)
dfres_val["Split"] = "Validation"  # optional column for tracking

# -------------------------------
# Combine Train + Validation
# -------------------------------
dfres_all = pd.concat([dfres_train, dfres_val], ignore_index=True)

# -------------------------------
# Merge with metadata
# -------------------------------
meta = pd.read_csv("abide_brainrotnet_metadata.csv")  # ASD dataset metadata
df_merged = dfres_all.merge(meta[["ImageID", "Diag"]], on="ImageID", how="inner")

plot_with_metrics(dfres_all, x_col="Age", y_col="Predicted_Age", hue_col="AgeGap",
                  title="ABIDE-II Age gap predictions", x_lim=(0, 40))

plot_with_metrics(dfres_val, x_col="Age", y_col="Predicted_Age", hue_col="AgeGap",
                  title="ABIDE-II Validation Age gap predictions", x_lim=(0, 40))



print(f"Merged dataset shape: {df_merged.shape}")
print(df_merged[["Split", "Diag", "Age", "Predicted_Age", "AgeGap"]].head())


from scipy.stats import chi2_contingency, fisher_exact
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------
# Compute ±1 SD threshold
# -------------------------------
std_threshold = df_merged["AgeGap"].std()
df_merged["HighAgeGap"] = df_merged["AgeGap"] > std_threshold

# -------------------------------
# Case 1: Outcome = ASD
# -------------------------------
df_merged["IsASD"] = df_merged["Diag"] == "ASD"

# # Contingency table
contingency_asd = pd.crosstab(df_merged["HighAgeGap"], df_merged["IsASD"])
print("Contingency table (ASD):\n", contingency_asd, "\n")

# # Extract cell values
# a = contingency_asd.loc[True, True]   # HighAgeGap & ASD
# b = contingency_asd.loc[True, False]  # HighAgeGap & Control
# c = contingency_asd.loc[False, True]  # LowAgeGap & ASD
# d = contingency_asd.loc[False, False] # LowAgeGap & Control

# # Compute statistics
# odds_ratio_asd = (a / b) / (c / d)
# p_high_asd = a / (a + b)
# p_low_asd = c / (c + d)
# relative_risk_asd = p_high_asd / p_low_asd

# # Chi-square and Fisher exact test
# chi2_asd, p_chi_asd, dof_asd, expected_asd = chi2_contingency(contingency_asd)
# _, p_fisher_asd = fisher_exact(contingency_asd)

# # Print results
# print(f"ASD Odds Ratio: {odds_ratio_asd:.4f}")
# print(f"ASD Relative Risk: {relative_risk_asd:.4f}")
# print(f"ASD Chi-square p-value: {p_chi_asd:.6f}")
# print(f"ASD Fisher exact p-value: {p_fisher_asd:.6f}")

# # -------------------------------
# # Proportion-based comparison (High vs Low AgeGap)
# # -------------------------------
# std_val = df_merged["AgeGap"].std()
# high_group = df_merged[df_merged["AgeGap"] > std_val]
# low_group  = df_merged[df_merged["AgeGap"] < std_val]

# prop_high_asd = (high_group["Diag"] == "ASD").mean()
# prop_low_asd  = (low_group["Diag"] == "ASD").mean()
# ratio_asd = prop_high_asd / prop_low_asd if prop_low_asd > 0 else float("inf")

# print(f"\nProportion ASD in High AgeGap (>+1SD): {prop_high_asd:.3f}")
# print(f"Proportion ASD in Low AgeGap (<-1SD): {prop_low_asd:.3f}")
# print(f"Relative likelihood (High vs Low): {ratio_asd:.2f}x")

# # Two-proportion z-test (optional alternative p-value)
# n_high = len(high_group)
# n_low = len(low_group)
# p_pool = (prop_high_asd * n_high + prop_low_asd * n_low) / (n_high + n_low)
# se = np.sqrt(p_pool * (1 - p_pool) * (1/n_high + 1/n_low))
# z_stat = (prop_high_asd - prop_low_asd) / se
# from scipy.stats import norm
# p_two_prop = 2 * (1 - norm.cdf(abs(z_stat)))
# print(f"P-value (difference in ASD proportions, High vs Low): {p_two_prop:.5f}")

# # -------------------------------
# # Visualization 1: Heatmap
# # -------------------------------
# heatmap_data = contingency_asd.rename(
#     index={True: "High Age-Gap (> +1 SD)", False: "Low/Normal Age-Gap"},
#     columns={True: "ASD", False: "Control"}
# )

# plt.figure(figsize=(5, 4))
# sns.heatmap(
#     heatmap_data,
#     annot=True, fmt="d",
#     cmap="Reds",
#     cbar=False,
#     linewidths=1,
#     linecolor="white",
#     annot_kws={"fontsize": 12, "weight": "bold"}
# )
# plt.title("Contingency Table: Age-Gap Group vs ASD Diagnosis", fontsize=14, weight="bold")
# plt.xlabel("Diagnosis")
# plt.ylabel("Age-Gap Group")
# plt.tight_layout()
# plt.show()

# -------------------------------
# Visualization 2: Proportion bar plot
# -------------------------------
prop_table = contingency_asd.div(contingency_asd.sum(axis=1), axis=0).reset_index()
prop_table = prop_table.melt(id_vars="HighAgeGap", var_name="IsASD", value_name="Proportion")

prop_table["HighAgeGap"] = prop_table["HighAgeGap"].map({True: "High Age-Gap (> +1 SD)", False: "Low/Normal Age-Gap"})
prop_table["IsASD"] = prop_table["IsASD"].map({True: "ASD", False: "Control"})

plt.figure(figsize=(6, 5))
sns.barplot(
    data=prop_table,
    x="HighAgeGap",
    y="Proportion",
    hue="IsASD",
    palette=["#b3cde0", "#005b96"]
)
plt.title("Proportion of ASD vs Control by Age-Gap Group", fontsize=14, weight="bold")
plt.xlabel("")
plt.ylabel("Proportion", fontsize=12)
plt.legend(title="Diagnosis", loc="upper right")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


from scipy.stats import ttest_ind, mannwhitneyu

# Split AgeGap by diagnosis
asd_gaps = df_merged.loc[df_merged["IsASD"], "AgeGap"]
ctrl_gaps = df_merged.loc[~df_merged["IsASD"], "AgeGap"]

# Mean & std summary
print(f"ASD mean AgeGap: {asd_gaps.mean():.3f} ± {asd_gaps.std():.3f}")
print(f"Control mean AgeGap: {ctrl_gaps.mean():.3f} ± {ctrl_gaps.std():.3f}")

# Two-sample t-test (equal variances not assumed)
t_stat, p_ttest = ttest_ind(asd_gaps, ctrl_gaps, equal_var=False, alternative='less')
print(f"\nT-test (ASD < Control): t = {t_stat:.3f}, p = {p_ttest:.6f}")

# Mann–Whitney U test (nonparametric alternative)
u_stat, p_mw = mannwhitneyu(asd_gaps, ctrl_gaps, alternative='less')
print(f"Mann–Whitney U test (ASD < Control): U = {u_stat:.3f}, p = {p_mw:.6f}")


# Contingency (HighAgeGap vs ASD)
contingency = pd.crosstab(df_merged["IsASD"], df_merged["HighAgeGap"])
print("\nContingency Table (ASD vs HighAgeGap):\n", contingency)

# Fisher’s exact test (ASD less likely to have high AgeGap)
_, p_fisher = fisher_exact(contingency, alternative='less')
chi2, p_chi, _, _ = chi2_contingency(contingency)

print(f"\nFisher exact test (ASD less likely HighAgeGap): p = {p_fisher:.6f}")
print(f"Chi-square test (two-sided): p = {p_chi:.6f}")

# # -------------------------------
# # Visualization: Age vs AgeGap Scatter
# # -------------------------------
# std_val = df_merged["AgeGap"].std()

# fig, ax = plt.subplots(figsize=(10,6))

# # ASD subjects (red)
# subset = df_merged[df_merged["Diag"] == "ASD"]
# ax.scatter(subset["AgeGap"], subset["Age"],
#            color="darkred", label="ASD", alpha=0.8, s=30)

# # Control subjects (gray)
# subset = df_merged[df_merged["Diag"] == "Control"]
# ax.scatter(subset["AgeGap"], subset["Age"],
#            facecolors=(0.8, 0.8, 0.8),
#            edgecolors=(0.05, 0.05, 0.05),
#            alpha=0.1, linewidth=0.7, s=30, label="Control")

# # ±1 SD cutoff lines
# ax.axvline(std_val, color="black", linestyle="--", linewidth=1.5, label="+1 SD")
# ax.axvline(-std_val, color="black", linestyle="--", linewidth=1.5, label="−1 SD")

# ax.set_xlabel("AgeGap")
# ax.set_ylabel("Age")
# ax.set_title("Subjects by AgeGap and Chronological Age (ASD vs Control)")
# ax.legend(loc="upper right")

# plt.tight_layout()
# plt.show()


import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, norm
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Converse case:
# Outcome = High AgeGap
# Predictor = ASD
# -------------------------------
contingency_highgap = pd.crosstab(df_merged["IsASD"], df_merged["HighAgeGap"])
print("\nContingency table (High AgeGap | ASD):\n", contingency_highgap, "\n")

# Extract cell values
a = contingency_highgap.loc[True, True]   # ASD & HighAgeGap
b = contingency_highgap.loc[True, False]  # ASD & LowAgeGap
c = contingency_highgap.loc[False, True]  # Control & HighAgeGap
d = contingency_highgap.loc[False, False] # Control & LowAgeGap

# Compute statistics
odds_ratio_highgap = (a / b) / (c / d)
p_high_asd = a / (a + b)
p_high_ctrl = c / (c + d)
relative_risk_highgap = p_high_asd / p_high_ctrl

# Chi-square and Fisher exact test
chi2_highgap, p_chi_highgap, dof_highgap, expected_highgap = chi2_contingency(contingency_highgap)
_, p_fisher_highgap = fisher_exact(contingency_highgap)

# -------------------------------
# 95% Confidence Intervals
# -------------------------------

# --- Odds Ratio CI ---
se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
ci_low_or = np.exp(np.log(odds_ratio_highgap) - 1.96 * se_log_or)
ci_high_or = np.exp(np.log(odds_ratio_highgap) + 1.96 * se_log_or)

# --- Relative Risk CI ---
se_log_rr = np.sqrt((1/a - 1/(a + b)) + (1/c - 1/(c + d)))
ci_low_rr = np.exp(np.log(relative_risk_highgap) - 1.96 * se_log_rr)
ci_high_rr = np.exp(np.log(relative_risk_highgap) + 1.96 * se_log_rr)

# Print results
print(f"High AgeGap Odds Ratio (ASD vs Control): {odds_ratio_highgap:.4f} "
      f"[95% CI: {ci_low_or:.4f}, {ci_high_or:.4f}]")

print(f"High AgeGap Relative Risk (ASD vs Control): {relative_risk_highgap:.4f} "
      f"[95% CI: {ci_low_rr:.4f}, {ci_high_rr:.4f}]")

print(f"High AgeGap Chi-square p-value: {p_chi_highgap:.6f}")
print(f"High AgeGap Fisher exact p-value: {p_fisher_highgap:.6f}")

# -------------------------------
# Visualization: Contingency heatmap
# -------------------------------
plt.figure(figsize=(2.5, 4.5))
sns.heatmap(
    contingency_highgap,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    linewidths=0.5,
    annot_kws={"size": 12, "weight": "bold"}
)
plt.title("High AgeGap Frequency by Diagnosis", fontsize=13, weight="bold")
plt.xlabel("High AgeGap (> +1 SD)")
plt.ylabel("Diagnosis (ASD / Control)")
plt.tight_layout()
plt.show()

# -------------------------------
# Visualization: Proportion barplot
# -------------------------------

# Convert contingency table to proportions per diagnosis
prop_table = contingency_highgap.div(contingency_highgap.sum(axis=1), axis=0).reset_index()
prop_table = prop_table.melt(id_vars="IsASD", var_name="HighAgeGap", value_name="Proportion")

# Map boolean labels to descriptive text
prop_table["IsASD"] = prop_table["IsASD"].map({True: "ASD", False: "Control"})
prop_table["HighAgeGap"] = prop_table["HighAgeGap"].map({True: "High Age-Gap (> +1 SD)", False: "Low/Normal Age-Gap"})

# Plot
plt.figure(figsize=(6, 5))
sns.barplot(
    data=prop_table,
    x="IsASD",
    y="Proportion",
    hue="HighAgeGap",
    palette=["#b3cde0", "#005b96"]
)

plt.title("Proportion of High vs Normal Age-Gap by Diagnosis", fontsize=14, weight="bold")
plt.xlabel("")
plt.ylabel("Proportion", fontsize=12)
plt.ylim(0, 1)
plt.legend(title="Age-Gap Group", loc="upper right")
plt.tight_layout()
plt.show()
