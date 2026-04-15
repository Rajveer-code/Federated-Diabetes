"""
SCRIPT 01 — DATA PARTITIONING
==============================
Loads nhanes_merged.csv → replicates IEEE paper preprocessing →
partitions into 3 federated nodes with realistic demographic variation.

Run FIRST before any other script.
Usage: python 01_partition_data.py
"""

import pandas as pd
import numpy as np
import os, sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (NHANES_RAW, DATA_DIR, FEATURE_COLS, TARGET_COL,
                                NODE_A_PATH, NODE_B_PATH, NODE_C_PATH,
                                CENTRALISED_PATH, SEED)

np.random.seed(SEED)

print("=" * 65)
print("  FEDERATED DIABETES — STEP 01: DATA PARTITIONING")
print("=" * 65)

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print(f"\n[1/6] Loading: {NHANES_RAW}")
df = pd.read_csv(NHANES_RAW)
print(f"      Raw shape: {df.shape}")

# ─── DIABETES LABEL ───────────────────────────────────────────────────────────
print("\n[2/6] Creating composite diabetes label (same as IEEE paper)...")
#   Priority order: self-report > HbA1c >= 6.5 > fasting glucose >= 126

df['DIABETES'] = np.nan
# Step-wise assignment (later lines override earlier → highest priority last)
df.loc[df['LBXGLU'] < 126,  'DIABETES'] = 0
df.loc[df['LBXGH']  < 6.5,  'DIABETES'] = 0
df.loc[df['DIQ010'] == 2,   'DIABETES'] = 0
df.loc[df['LBXGLU'] >= 126, 'DIABETES'] = 1
df.loc[df['LBXGH']  >= 6.5, 'DIABETES'] = 1
df.loc[df['DIQ010'] == 1,   'DIABETES'] = 1   # self-report wins

# Adults only
df = df[df['RIDAGEYR'] >= 18].copy()
df = df.dropna(subset=['DIABETES'])
df['DIABETES'] = df['DIABETES'].astype(int)
print(f"      After filtering: {len(df):,} samples | "
      f"Prevalence: {df['DIABETES'].mean():.1%}")

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
print("\n[3/6] Engineering features...")

# Smoking: 0=never, 1=former, 2=current
df['SMOKING'] = 0
df.loc[df['SMQ020'] == 1, 'SMOKING'] = 1                              # ever smoked
df.loc[(df['SMQ020'] == 1) & (df['SMQ040'].isin([1, 2])), 'SMOKING'] = 2  # currently

# Physical activity: 1 if any activity domain reports 'Yes'
pa_cols = ['PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665']
df['PHYS_ACTIVITY'] = (df[pa_cols] == 1).any(axis=1).astype(int)

# Binary flags
df['HEART_ATTACK'] = df['MCQ160E'].map({1: 1, 2: 0}).fillna(0).astype(int)
df['STROKE']       = df['MCQ160F'].map({1: 1, 2: 0}).fillna(0).astype(int)

# ─── CLEAN DATASET ────────────────────────────────────────────────────────────
print("\n[4/6] Building clean dataset...")
df_clean = df[FEATURE_COLS + [TARGET_COL]].copy()

# Impute (median for continuous, mode for categorical — same as paper)
df_clean['BMXBMI']  = df_clean['BMXBMI'].fillna(df_clean['BMXBMI'].median())
df_clean['RIAGENDR'] = df_clean['RIAGENDR'].fillna(df_clean['RIAGENDR'].mode()[0])
df_clean['RIDRETH3'] = df_clean['RIDRETH3'].fillna(df_clean['RIDRETH3'].mode()[0])
df_clean = df_clean.reset_index(drop=True)

print(f"      Clean shape: {df_clean.shape}")
print(f"      Missing values: {df_clean.isnull().sum().sum()}")

# ─── PARTITION ────────────────────────────────────────────────────────────────
print("\n[5/6] Partitioning into 3 hospital nodes...")
print("""
  Strategy:
  ┌─────────────────────────────────────────────────────────────┐
  │ Node A — Young Urban   : age<45 OR minority race            │
  │           → Simulates community health clinic               │
  │                                                             │
  │ Node B — Elderly Rural : age≥55 AND (White OR MexAmerican)  │
  │           → Simulates rural critical access hospital        │
  │           → This is the HARD node (elderly fairness gap)    │
  │                                                             │
  │ Node C — Mixed Metro   : remaining population               │
  │           → Simulates academic medical center               │
  └─────────────────────────────────────────────────────────────┘
""")

# Node A: young + urban minority
mask_a = (df_clean['RIDAGEYR'] < 45) | (df_clean['RIDRETH3'].isin([2.0, 4.0]))
pool_a  = df_clean[mask_a]
node_a  = pool_a.sample(n=min(4500, len(pool_a)), random_state=SEED)

# Node B: elderly rural
mask_b = (df_clean['RIDAGEYR'] >= 55) & (df_clean['RIDRETH3'].isin([1.0, 3.0]))
pool_b  = df_clean[mask_b]
node_b  = pool_b.sample(n=min(3500, len(pool_b)), random_state=SEED)

# Node C: remainder
used   = set(node_a.index) | set(node_b.index)
pool_c = df_clean[~df_clean.index.isin(used)]
node_c = pool_c.sample(n=min(4000, len(pool_c)), random_state=SEED)

# ─── SAVE ─────────────────────────────────────────────────────────────────────
print("[6/6] Saving datasets...")
os.makedirs(DATA_DIR, exist_ok=True)

node_a.to_csv(NODE_A_PATH,      index=False)
node_b.to_csv(NODE_B_PATH,      index=False)
node_c.to_csv(NODE_C_PATH,      index=False)
df_clean.to_csv(CENTRALISED_PATH, index=False)

# ─── REPORT ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PARTITION REPORT")
print("=" * 65)

nodes = {
    "Node A — Young Urban  ": node_a,
    "Node B — Elderly Rural": node_b,
    "Node C — Mixed Metro  ": node_c,
}

for name, node in nodes.items():
    young   = (node['RIDAGEYR'] < 40).mean()
    elderly = (node['RIDAGEYR'] >= 60).mean()
    white   = (node['RIDRETH3'] == 3.0).mean()
    black   = (node['RIDRETH3'] == 4.0).mean()
    hisp    = (node['RIDRETH3'].isin([1.0, 2.0])).mean()

    print(f"\n  {name}")
    print(f"    Samples     : {len(node):,}")
    print(f"    Diabetes %  : {node[TARGET_COL].mean():.1%}")
    print(f"    Age mean    : {node['RIDAGEYR'].mean():.1f} yrs  "
          f"(median {node['RIDAGEYR'].median():.0f})")
    print(f"    Age <40     : {young:.1%}  |  Age ≥60: {elderly:.1%}")
    print(f"    BMI mean    : {node['BMXBMI'].mean():.1f}")
    print(f"    Race: White {white:.1%} | Black {black:.1%} | Hispanic {hisp:.1%}")

print(f"\n  Centralised (full) : {len(df_clean):,} samples")
print(f"\n  Saved to: {DATA_DIR}")
print("\n✅  Partitioning complete — run 02_centralised_baseline.py next")
print("=" * 65)
