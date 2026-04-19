"""

Generates a  gaming player dataset for K-Means clustering.
Features:
    - hours_per_week     : Average hours played per week
    - kd_ratio           : Kill/death ratio (combat performance)
    - monthly_spending   : In-game spending in USD per month
    - social_interactions: Average social interactions per session
                           (party invites, messages, team-ups)

"""

import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
RANDOM_SEED = 42
N_TOTAL = 300
N_PER_GROUP = N_TOTAL // 3          # 100 per type
N_REMAINDER = N_TOTAL - 2 * N_PER_GROUP  # handles rounding (100)

np.random.seed(RANDOM_SEED)

# ==========================================
# CLUSTER 1 : "Hardcore Grinders"
# Heavy playtime, high KD, moderate spending,
# low social interaction (solo-focused)
# ==========================================
c1_hours = np.random.normal(35, 6, N_PER_GROUP).clip(15, 60)
c1_kd = np.random.normal(2.8, 0.5, N_PER_GROUP).clip(1.0, 6.0)
c1_spend = np.random.normal(25, 10, N_PER_GROUP).clip(0, 80)
c1_social = np.random.normal(4, 2, N_PER_GROUP).clip(0, 15)

# ==========================================
# CLUSTER 2 : "Social Casuals"
# Low playtime, average KD, low spending,
# very high social interaction
# ==========================================
c2_hours = np.random.normal(8, 3, N_PER_GROUP).clip(1, 20)
c2_kd = np.random.normal(0.9, 0.3, N_PER_GROUP).clip(0.1, 2.5)
c2_spend = np.random.normal(5, 4, N_PER_GROUP).clip(0, 20)
c2_social = np.random.normal(18, 4, N_PER_GROUP).clip(5, 40)

# ==========================================
# CLUSTER 3 : "Whale Competitors"
# Moderate playtime, high KD, very high spending,
# moderate social interaction (guild/team play)
# ==========================================
c3_hours = np.random.normal(20, 5, N_REMAINDER).clip(8, 40)
c3_kd = np.random.normal(2.1, 0.6, N_REMAINDER).clip(0.5, 5.0)
c3_spend = np.random.normal(120, 35, N_REMAINDER).clip(30, 300)
c3_social = np.random.normal(11, 3, N_REMAINDER).clip(2, 25)

# ==========================================
# ASSEMBLE DATAFRAME
# ==========================================
df = pd.DataFrame({
    "hours_per_week": np.concatenate([c1_hours, c2_hours, c3_hours]),
    "kd_ratio": np.concatenate([c1_kd, c2_kd, c3_kd]),
    "monthly_spending": np.concatenate([c1_spend, c2_spend, c3_spend]),
    "social_interactions": np.concatenate([c1_social, c2_social, c3_social]),
})

# Round to sensible precision
df["hours_per_week"] = df["hours_per_week"].round(1)
df["kd_ratio"] = df["kd_ratio"].round(2)
df["monthly_spending"] = df["monthly_spending"].round(2)
df["social_interactions"] = df["social_interactions"].round(0).astype(int)

# Shuffle rows so cluster order is not obvious
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# ==========================================
# SAVE
# ==========================================
OUTPUT_CSV = "my_dataset.csv"
df.to_csv(OUTPUT_CSV, index=False)

print(f"Dataset saved to {OUTPUT_CSV}")
print(f"Shape: {df.shape}")
print()
print(df.describe().round(2))