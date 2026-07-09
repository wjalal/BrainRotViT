import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv("model_dumps/mix/sfcn_predicted_val.csv")

# Scatter plot
plt.figure(figsize=(7, 7))
plt.scatter(df["Age"], df["Predicted_Age"], alpha=0.6, s=20)

# Perfect prediction line
min_age = min(df["Age"].min(), df["Predicted_Age"].min())
max_age = max(df["Age"].max(), df["Predicted_Age"].max())
plt.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label="y = x")

plt.xlabel("Chronological Age")
plt.ylabel("Predicted Age")
plt.title("Predicted Age vs. Chronological Age")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.tight_layout()

plt.show()
