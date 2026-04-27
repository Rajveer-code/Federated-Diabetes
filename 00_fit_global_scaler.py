"""
00_fit_global_scaler.py
========================
Run this ONCE before any federated training or evaluation.

PURPOSE
-------
Fits a StandardScaler on the NHANES training split ONLY and saves it to
artefacts/global_nhanes_scaler.joblib.  All downstream scripts load this
single scaler via GLOBAL_SCALER_PATH and call .transform() — never .fit_transform().

This eliminates the data-leakage bug where 05_fairness_analysis.py was
fitting a fresh scaler on Node B data, contaminating evaluation with
node-local statistics instead of the NHANES training statistics.

USAGE
-----
  cd D:\\Projects\\diabetes_prediction_project\\federated
  python 00_fit_global_scaler.py

CRITICAL RULE
-------------
Every script that needs scaled features must:
  scaler = joblib.load(GLOBAL_SCALER_PATH)
  X_scaled = scaler.transform(X)   # transform ONLY — never fit_transform
"""

import os, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NHANES_PATH, GLOBAL_SCALER_PATH, ARTEFACTS_DIR,
    FEATURE_COLS, LABEL_COL, RANDOM_SEED, TEST_SIZE,
)

os.makedirs(ARTEFACTS_DIR, exist_ok=True)

print("=" * 60)
print("  00_fit_global_scaler.py")
print("=" * 60)
print(f"\n  Data source : {NHANES_PATH}")
print(f"  Output      : {GLOBAL_SCALER_PATH}")

df = pd.read_csv(NHANES_PATH)
X  = df[FEATURE_COLS].values.astype('float32')
y  = df[LABEL_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

scaler = StandardScaler()
scaler.fit(X_train)          # fit on NHANES train split ONLY — never refit

joblib.dump(scaler, GLOBAL_SCALER_PATH)

print(f"\n  Fitted on {len(X_train):,} NHANES training samples")
print(f"  (held-out test: {len(X_test):,} rows, {y_test.mean():.1%} diabetes)")
print(f"\n  Feature statistics (train split):")
for col, mean_, scale_ in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
    print(f"    {col:<20}  mean={mean_:.4f}  std={scale_:.4f}")

# Sanity-check: mean of scaled train data should be ≈ 0
X_train_sc = scaler.transform(X_train)
assert np.abs(X_train_sc.mean(axis=0)).max() < 1e-5, "Scaler sanity check failed"

<<<<<<< HEAD
print(f"\n  Saved global scaler → {GLOBAL_SCALER_PATH}")
=======
print(f"\n  Saved global scaler -> {GLOBAL_SCALER_PATH}")
>>>>>>> 435718c297f04a6b74b12d2ac00504407237e06b
print("  Sanity check passed: scaled train mean ≈ 0")
print("\n  NEXT: run scripts in order:")
print("    python 02_centralised_baseline.py")
print("    python 03_federated_simulation.py")
print("    python 03b_fednova_corrected.py")
print("    python 05_fairness_analysis.py")
print("    python 07_external_validation.py")
print("    python 07_statistical_analysis.py")
print("=" * 60)
