# 🎮 Gaming Player Segmentation with K-Means Clustering

A machine learning lab project that applies **unsupervised K-Means clustering** to a synthetic gaming player dataset. The goal is to discover meaningful behavioral segments among players using four behavioral features, without any pre-existing labels.

---

## 📁 Project Structure

```
kmeans-gaming-player-segmentation/
│
├── dataset.py               # Generates the synthetic my_dataset.csv
├── pipeline.py              # Main ML pipeline: scaling → K sweep → final model → predictions
├── Visualization.py         # Scatter plots of cluster results (2 feature-pair views)
├── Visualization2.py        # Silhouette knife-plot for the final model
│
├── my_dataset.csv           # Synthetic dataset (300 players, 4 features)
│
├── elbow_silhouette.png     # Elbow + Silhouette score curves (K = 2–10)
├── cluster_scatter.png      # Cluster scatter: play intensity & spending vs. social activity
├── silhouette_plot.png      # Per-sample silhouette coefficients by cluster
│
├── REPORT.md                # Lab reflection answers
└── dataset_README.md        # Dataset generation documentation
```

---

## 🗂️ Dataset

The dataset (`my_dataset.csv`) contains **300 synthetic gaming players** with **4 behavioral features**:

| Feature | Description |
|---|---|
| `hours_per_week` | Average hours played per week |
| `kd_ratio` | Kill/Death ratio (combat performance) |
| `monthly_spending` | In-game spending in USD per month |
| `social_interactions` | Average social interactions per session |

Three player archetypes were embedded during generation using normal distributions with realistic clipping and shuffling to produce natural overlap:

| Archetype | Hours | K/D | Spending | Social |
|---|---|---|---|---|
| **Hardcore Grinders** | High (~35 hrs) | High (~2.8) | Low (~$25) | Low (~4) |
| **Social Casuals** | Low (~8 hrs) | Low (~0.9) | Very Low (~$5) | Very High (~18) |
| **Whale Competitors** | Moderate (~20 hrs) | Moderate (~2.1) | Very High (~$120) | Moderate (~11) |

---

## ⚙️ Pipeline Overview

### 1. `dataset.py` — Data Generation
Generates `my_dataset.csv` with 300 players across three clusters using `numpy` normal distributions. Applies clipping, rounding, and row shuffling to simulate realistic data.

### 2. `pipeline.py` — Core ML Pipeline
Runs the full clustering workflow as specified by the lab:

1. Load `my_dataset.csv`
2. Scale all features with `StandardScaler` (mandatory for K-Means)
3. Sweep K from 2–10, recording **inertia** and **silhouette score** for each K
4. Plot Elbow curve and Silhouette score curve side-by-side → `elbow_silhouette.png`
5. Fit final model at **K = 3** (justified by both metrics)
6. Print per-cluster statistics (mean ± std for each feature)
7. Run 3 manual predictions on hand-crafted player profiles

### 3. `Visualization.py` — Cluster Scatter Plots
Produces `cluster_scatter.png` with two subplots:
- **Left:** Hours per Week vs. K/D Ratio (play intensity vs. combat skill)
- **Right:** Monthly Spending vs. Social Interactions

Cluster centroids are inverse-transformed to the original feature scale and overlaid as white star markers.

### 4. `Visualization2.py` — Silhouette Plot
Produces `silhouette_plot.png` — a horizontal knife-plot showing per-sample silhouette coefficients grouped by cluster, with a dashed line at the overall average score.

---

## 📊 Results

**Optimal K = 3**, confirmed by:
- A clear elbow inflection in the inertia curve at K = 3
- A silhouette score peak of ~0.60 at K = 3

| Segment | Defining Trait |
|---|---|
| **Hardcore Grinders** | High playtime + high skill, minimal spending |
| **Social Casuals** | Low playtime + low skill, very high social activity |
| **Whale Competitors** | Moderate playtime, dominant in monthly spending |

---

## 🚀 Running the Project

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Execution Order

```bash
# Step 1: Generate the dataset
python dataset.py

# Step 2: Run the full clustering pipeline
python pipeline.py

# Step 3: Generate cluster scatter plots
python Visualization.py

# Step 4: Generate the silhouette plot
python Visualization2.py
```

All output images are saved to the working directory.

---

## 🔑 Key Concepts Demonstrated

- **Feature Scaling** — Why `StandardScaler` is mandatory before K-Means (without it, high-magnitude features like `monthly_spending` dominate the distance metric)
- **Elbow Method** — Choosing K by finding the inflection point in Within-Cluster Sum of Squares (WCSS / Inertia)
- **Silhouette Analysis** — Validating cluster quality per sample and overall
- **Manual Prediction** — Applying a fitted model to new, unseen player profiles

---

## ⚠️ Limitations

- K-Means requires specifying K in advance
- Assumes spherical, equal-sized clusters — may fail with complex shapes
- Sensitive to outliers (e.g., a $500/month spender could shift a centroid)
- Hard assignment: every player belongs to exactly one cluster

---

## 📋 Info

**Course:** Machine Learning Lab — Lab 05  
**Topic:** Unsupervised Learning / K-Means Clustering  
**Author:** Azhar  
**Suggested Project Name:** `kmeans-gaming-player-segmentation`
