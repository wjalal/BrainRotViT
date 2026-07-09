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
    lowess = sm.nonparametric.lowess
    lowess_fit = lowess(dfres_agegap.Predicted_Age.to_numpy(),
                        dfres_agegap.Age.to_numpy(), frac=0.8, it=3)
    lowess_fit_int = interp1d(lowess_fit[:,0], lowess_fit[:,1],
                              bounds_error=False, kind='linear',
                              fill_value=(0, 150)) 
    y_lowess = lowess_fit_int(dfres_agegap.Age)
    dfres_agegap["yhat_lowess"] = y_lowess

    if dfres_agegap["yhat_lowess"].isna().any():
        n_missing = dfres_agegap["yhat_lowess"].isna().sum()
        print(f"Could not predict lowess yhat in {n_missing} samples")
        dfres_agegap = dfres_agegap.dropna(subset=["yhat_lowess"])
    
    dfres_agegap["AgeGap"] = dfres_agegap["Predicted_Age"] - dfres_agegap["yhat_lowess"]
    return dfres_agegap


# -------------------------------
# Helper: Plot with MAE and R²
# -------------------------------
def plot_with_metrics(data, x_col, y_col, hue_col, title, x_lim):
    mae = mean_absolute_error(data[x_col], data[y_col])
    r2 = r2_score(data[x_col], data[y_col])
    
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col,
                    palette='coolwarm', hue_norm=(-12, 12))
    plt.xlim(*x_lim)
    plt.title(f"{title}\nMAE: {mae:.2f}, R²: {r2:.2f}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()


# -------------------------------
# Training set
# -------------------------------
dfres_train = pd.read_csv("model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_train.csv", sep=",", index_col=0).reset_index()
dfres_train = calculate_lowess_yhat_and_agegap(dfres_train)

plot_with_metrics(dfres_train, x_col="Age", y_col="Predicted_Age", hue_col="AgeGap",
                  title="Age gap predictions (Train Set)", x_lim=(0, 35))


# -------------------------------
# Validation set
# -------------------------------
dfres_val = pd.read_csv("model_dumps/mix/cnn_mx_bigdo_ch_sw_res_predicted_ages_val.csv", sep=",", index_col=0).reset_index()
dfres_val = calculate_lowess_yhat_and_agegap(dfres_val)

plot_with_metrics(dfres_val, x_col="Age", y_col="Predicted_Age", hue_col="AgeGap",
                  title="Age gap predictions (Validation Set)", x_lim=(0, 35))


# -------------------------------
# Merge with metadata
# -------------------------------
meta = pd.read_csv("abide_brainrotnet_metadata.csv")  # your ASD dataset metadata
df_merged = dfres_val.merge(meta[["ImageID", "Diag"]], on="ImageID", how="inner")

# Compute ±1 SD threshold
std_threshold = df_merged["AgeGap"].std()
df_merged["HighAgeGap"] = df_merged["AgeGap"] > std_threshold

# -------------------------------
# Case 1: Outcome = ASD
# -------------------------------
df_merged["IsASD"] = df_merged["Diag"] == "ASD"

contingency_asd = pd.crosstab(df_merged["HighAgeGap"], df_merged["IsASD"])
print("Contingency table (ASD):\n", contingency_asd)

a = contingency_asd.loc[True, True]   # HighAgeGap & ASD
b = contingency_asd.loc[True, False]  # HighAgeGap & Control
c = contingency_asd.loc[False, True]  # LowAgeGap & ASD
d = contingency_asd.loc[False, False] # LowAgeGap & Control

odds_ratio_asd = (a/b) / (c/d)
p_high_asd = a / (a + b)
p_low_asd = c / (c + d)
relative_risk_asd = p_high_asd / p_low_asd

print("ASD Odds Ratio:", odds_ratio_asd)
print("ASD Relative Risk:", relative_risk_asd)


# -------------------------------
# High vs Low AgeGap proportions
# -------------------------------
std_val = df_merged["AgeGap"].std()
high_group = df_merged[df_merged["AgeGap"] > std_val]
low_group  = df_merged[df_merged["AgeGap"] < std_val]

prop_high_asd = (high_group["Diag"] == "ASD").mean()
prop_low_asd  = (low_group["Diag"] == "ASD").mean()
ratio_asd = prop_high_asd / prop_low_asd if prop_low_asd > 0 else float("inf")

print(f"\nProportion ASD in High AgeGap (>+1SD): {prop_high_asd:.3f}")
print(f"Proportion ASD in Low AgeGap (<-1SD): {prop_low_asd:.3f}")
print(f"Relative likelihood (High vs Low): {ratio_asd:.2f}x")


# -------------------------------
# Visualization: Proportion Bar Plot
# -------------------------------
groups = ["ASD"]
high_props = [prop_high_asd]
low_props  = [prop_low_asd]

x = range(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(6,5))
bars1 = ax.bar([i - width/2 for i in x], low_props, width, label="AgeGap < -1 SD", color="steelblue")
bars2 = ax.bar([i + width/2 for i in x], high_props, width, label="AgeGap > +1 SD", color="salmon")

ax.set_ylabel("Proportion of Subjects")
ax.set_title("Proportion of ASD by AgeGap group")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()


# -------------------------------
# Visualization: Age vs AgeGap Scatter
# -------------------------------
std_val = df_merged["AgeGap"].std()

fig, ax = plt.subplots(figsize=(10,6))

# ASD subjects (red)
subset = df_merged[df_merged["Diag"] == "ASD"]
ax.scatter(subset["AgeGap"], subset["Age"],
           color="darkred", label="ASD", alpha=0.8, s=30)

# Control subjects (gray)
subset = df_merged[df_merged["Diag"] == "Control"]
ax.scatter(subset["AgeGap"], subset["Age"],
           facecolors=(0.8, 0.8, 0.8),
           edgecolors=(0.05, 0.05, 0.05),
           alpha=0.1, linewidth=0.7, s=30, label="Control")

# ±1 SD cutoff lines
ax.axvline(std_val, color="black", linestyle="--", linewidth=1.5, label="+1 SD")
ax.axvline(-std_val, color="black", linestyle="--", linewidth=1.5, label="−1 SD")

ax.set_xlabel("AgeGap")
ax.set_ylabel("Age")
ax.set_title("Subjects by AgeGap and Chronological Age (ASD vs Control)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
