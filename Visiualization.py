"""
Note : 
Scatter plot of K-Means clustering results for gaming player 

Shows two feature pairs in side-by-side subplots:
    Left  : hours_per_week  vs  kd_ratio
    Right : monthly_spending vs social_interactions

Centroids are  ransformed to original feature scale and overlaid
as white star markers with black edges.


"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# CONFIGURATION
# ==========================================
RANDOM_SEED = 42
DATA_PATH = "my_dataset.csv"
OUTPUT_PATH = "cluster_scatter.png"
OPTIMAL_K = 3

COLORS = ["#14b8a6", "#8b5cf6", "#f43f5e"]
SEGMENT_NAMES = ["Hardcore Grinders", "Social Casuals", "Whale Competitors"]

# ==========================================
# LOAD AND SCALE
# ==========================================
df = pd.read_csv(DATA_PATH)
feature_cols = ["hours_per_week", "kd_ratio", "monthly_spending", "social_interactions"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

# ==========================================
# FIT MODEL
# ==========================================
kmeans = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=RANDOM_SEED)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Re-map cluster IDs to consistent archetype names by monthly_spending
means = df.groupby("cluster")["monthly_spending"].mean().sort_values()
sorted_ids = means.index.tolist()
name_map = {
    sorted_ids[0]: "Social Casuals",
    sorted_ids[1]: "Hardcore Grinders",
    sorted_ids[2]: "Whale Competitors",
}
segment_order = [sorted_ids[1], sorted_ids[0], sorted_ids[2]]  # Grinders, Casuals, Whales
color_map = {
    sorted_ids[1]: COLORS[0],   # Hardcore Grinders -> teal
    sorted_ids[0]: COLORS[1],   # Social Casuals    -> purple
    sorted_ids[2]: COLORS[2],   # Whale Competitors -> rose
}

# Inverse-transform centroids to original scale
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)

# ==========================================
# PLOT
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "K-Means Player Segmentation (K = 3)\nGaming Behavioral Clusters",
    fontsize=14, y=1.02
)

# -- LEFT: hours_per_week vs kd_ratio
ax = axes[0]
for cid in sorted_ids:
    mask = df["cluster"] == cid
    ax.scatter(
        df.loc[mask, "hours_per_week"],
        df.loc[mask, "kd_ratio"],
        c=color_map[cid],
        label=name_map[cid],
        alpha=0.65,
        s=45,
        edgecolors="white",
        linewidths=0.4,
    )

# Plot centroids (columns 0 and 1 -> hours_per_week, kd_ratio)
ax.scatter(
    centroids_orig[:, 0],
    centroids_orig[:, 1],
    c="white",
    marker="*",
    s=450,
    edgecolors="black",
    linewidths=1.8,
    zorder=10,
    label="Centroids",
)
ax.set_xlabel("Hours per Week", fontsize=12)
ax.set_ylabel("Kill / Death Ratio", fontsize=12)
ax.set_title("Play Intensity vs. Combat Skill", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

# -- RIGHT: monthly_spending vs social_interactions
ax = axes[1]
for cid in sorted_ids:
    mask = df["cluster"] == cid
    ax.scatter(
        df.loc[mask, "monthly_spending"],
        df.loc[mask, "social_interactions"],
        c=color_map[cid],
        label=name_map[cid],
        alpha=0.65,
        s=45,
        edgecolors="white",
        linewidths=0.4,
    )

# Plot centroids (columns 2 and 3 -> monthly_spending, social_interactions)
ax.scatter(
    centroids_orig[:, 2],
    centroids_orig[:, 3],
    c="white",
    marker="*",
    s=450,
    edgecolors="black",
    linewidths=1.8,
    zorder=10,
    label="Centroids",
)
ax.set_xlabel("Monthly In-Game Spending (USD)", fontsize=12)
ax.set_ylabel("Social Interactions per Session", fontsize=12)
ax.set_title("Spending vs. Social Activity", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_PATH}")