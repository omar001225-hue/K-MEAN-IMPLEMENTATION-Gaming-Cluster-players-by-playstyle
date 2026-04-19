"""
Visualization_2.py
------------------
Silhouette plot for the final K-Means model (K = 3) applied to the
gaming player segmentation dataset.

Each cluster's silhouette coefficients are shown as a horizontal bar chart
(knife-plot). The vertical dashed line marks the average silhouette score.
Clusters above the average dashed line indicate well-separated groups.

Output image
------------
    silhouette_plot.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# ==========================================
# CONFIGURATION
# ==========================================
RANDOM_SEED = 42
DATA_PATH = "my_dataset.csv"
OUTPUT_PATH = "silhouette_plot.png"
OPTIMAL_K = 3

COLORS = ["#14b8a6", "#8b5cf6", "#f43f5e"]

# ==========================================
# LOAD, SCALE, FIT
# ==========================================
df = pd.read_csv(DATA_PATH)
feature_cols = ["hours_per_week", "kd_ratio", "monthly_spending", "social_interactions"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

kmeans = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=RANDOM_SEED)
labels = kmeans.fit_predict(X_scaled)

# Re-map cluster IDs to archetype names by monthly_spending
means = df.copy()
means["cluster"] = labels
mean_spend = means.groupby("cluster")["monthly_spending"].mean().sort_values()
sorted_ids = mean_spend.index.tolist()

name_map = {
    sorted_ids[0]: "Social Casuals",
    sorted_ids[1]: "Hardcore Grinders",
    sorted_ids[2]: "Whale Competitors",
}

# ==========================================
# SILHOUETTE COEFFICIENTS
# ==========================================
sil_values = silhouette_samples(X_scaled, labels)
avg_score = silhouette_score(X_scaled, labels)

# ==========================================
# PLOT
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

y_lower = 10
for idx, cid in enumerate(sorted_ids):
    cluster_sil = np.sort(sil_values[labels == cid])
    size = len(cluster_sil)

    ax.fill_betweenx(
        np.arange(y_lower, y_lower + size),
        0,
        cluster_sil,
        alpha=0.75,
        color=COLORS[idx],
        label=name_map[cid],
    )
    ax.text(
        -0.08,
        y_lower + size / 2,
        name_map[cid],
        ha="right",
        va="center",
        fontsize=10,
        color=COLORS[idx],
        fontweight="bold",
    )
    y_lower += size + 12

# Average silhouette line
ax.axvline(
    x=avg_score,
    color="#ef4444",
    linestyle="--",
    linewidth=2,
    label=f"Average Score: {avg_score:.3f}",
)

ax.set_xlim([-0.15, 1.0])
ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Player Records (grouped by cluster)", fontsize=12)
ax.set_title(
    f"Silhouette Plot for K-Means Clustering (K = {OPTIMAL_K})\nGaming Player Segmentation",
    fontsize=13,
)
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="x", alpha=0.2)

# Remove y-tick labels (individual data-point indices are not meaningful)
ax.set_yticks([])

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_PATH}")
print(f"Average Silhouette Score: {avg_score:.4f}")