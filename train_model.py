"""
MovieIQ - Data Prep, EDA, Statistical Testing & Model Training
Run this once locally to generate assets/ charts and the trained model (model.pkl).
The Streamlit app (MovieIQ.py) loads model.pkl at runtime.
"""

import ast
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# --- Custom visual identity: indigo / gold palette, replaces default seaborn look ---
ACCENT = "#F2B705"      # gold
PRIMARY = "#5B4FE9"     # indigo
BG = "#12131A"          # near-black background for charts
GRID = "#2A2C3B"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": "#EDEDED",
    "text.color": "#EDEDED",
    "xtick.color": "#B8B9C6",
    "ytick.color": "#B8B9C6",
    "grid.color": GRID,
    "axes.titlecolor": "#EDEDED",
    "font.family": "sans-serif",
})
sns.set_theme(style="darkgrid", rc=plt.rcParams)
CUSTOM_CMAP = sns.color_palette([PRIMARY, ACCENT, "#2FBF71", "#E8443A", "#3AA6D9"])

# ---------------------------------------------------------------------------
# STAGE 1: Data Preparation
# ---------------------------------------------------------------------------
df = pd.read_csv("movies.csv")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.describe())

# No missing values / zero budgets in this dataset, but keep the guard rails
# in place in case the CSV is swapped for a messier one later.
df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

# Target: success = 1 when revenue > budget
df["success"] = (df["revenue"] > df["budget"]).astype(int)
print("Success rate:", df["success"].mean())

# genres arrives as a stringified list of dicts, e.g. "[{'id': 18, 'name': 'Drama'}]"
def extract_genre(g):
    try:
        parsed = ast.literal_eval(g)
        return parsed[0]["name"] if parsed else "Unknown"
    except (ValueError, SyntaxError):
        return "Unknown"


df["genre"] = df["genres"].apply(extract_genre)

# ---------------------------------------------------------------------------
# STAGE 2: Exploratory Data Analysis
# ---------------------------------------------------------------------------

# Budget vs Revenue
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df, x="budget", y="revenue", hue="success",
    palette=[ACCENT, PRIMARY], alpha=0.7, s=45, edgecolor="none",
)
plt.title("Budget vs Revenue")
plt.tight_layout()
plt.savefig("assets/budget_vs_revenue.png", dpi=150)
plt.close()

# Genre counts + success rate by genre
genre_stats = df.groupby("genre")["success"].agg(["count", "mean"]).sort_values("count", ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=genre_stats.index, y=genre_stats["mean"], color=PRIMARY)
plt.ylabel("Success rate")
plt.title("Success Rate by Genre")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig("assets/success_by_genre.png", dpi=150)
plt.close()

# Correlation heatmap
numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
plt.figure(figsize=(6, 5))
sns.heatmap(
    df[numeric_cols].corr(), annot=True, fmt=".2f",
    cmap=sns.diverging_palette(250, 40, s=80, l=45, as_cmap=True),
    linewidths=0.5, linecolor=GRID,
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("assets/correlation_heatmap.png", dpi=150)
plt.close()

print("\nGenre success rates:\n", genre_stats)

# ---------------------------------------------------------------------------
# STAGE 3: Statistical Testing
# ---------------------------------------------------------------------------

# T-test: does vote_average differ between successful and unsuccessful movies?
success_votes = df[df["success"] == 1]["vote_average"]
fail_votes = df[df["success"] == 0]["vote_average"]
t_stat, p_val_t = stats.ttest_ind(success_votes, fail_votes, equal_var=False)
print(f"\nT-test (vote_average): t={t_stat:.3f}, p={p_val_t:.4f}")

# Chi-square: is genre associated with success?
contingency = pd.crosstab(df["genre"], df["success"])
chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square (genre vs success): chi2={chi2:.3f}, p={p_val_chi2:.4f}")

stats_results = {
    "t_stat": t_stat,
    "p_val_t": p_val_t,
    "chi2": chi2,
    "p_val_chi2": p_val_chi2,
}

# ---------------------------------------------------------------------------
# STAGE 4: Predictive Modeling (Random Forest)
# ---------------------------------------------------------------------------
features = ["budget", "popularity", "runtime", "vote_average"]
X = df[features]
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}")
print("Confusion matrix:\n", cm)

# Confusion matrix plot
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap=sns.light_palette(PRIMARY, as_cmap=True),
    xticklabels=["Fail", "Success"], yticklabels=["Fail", "Success"],
    linewidths=0.5, linecolor=GRID,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(6, 4))
sns.barplot(x=importances.values, y=importances.index, color=ACCENT)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=150)
plt.close()

print("\nFeature importance:\n", importances)

metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "confusion_matrix": cm.tolist(),
    "feature_importance": importances.to_dict(),
}

# ---------------------------------------------------------------------------
# Save everything the Streamlit app needs
# ---------------------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "features": features,
            "metrics": metrics,
            "stats_results": stats_results,
        },
        f,
    )

print("\nSaved model.pkl and charts in assets/. Done.")
