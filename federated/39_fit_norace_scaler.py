"""
39_fit_norace_scaler.py — global NHANES scaler for the race-free feature set
============================================================================
Mirrors 00_fit_global_scaler.py exactly (same source file, same train split, same
seed) but over FEATURE_COLS_NORACE. The existing scaler cannot be reused: it was
fitted on 8 columns and carries per-column means and scales, so applying it to a
7-column matrix would silently misalign every feature after the dropped one.

Fit on the NHANES training split only, never refit downstream.

Output: artefacts/global_nhanes_scaler_norace.joblib

Usage:  python 39_fit_norace_scaler.py
"""
import os, sys
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NHANES_PATH, GLOBAL_SCALER_NORACE_PATH, GLOBAL_SCALER_PATH, ARTEFACTS_DIR,
    FEATURE_COLS, FEATURE_COLS_NORACE, LABEL_COL, RANDOM_SEED, TEST_SIZE,
)

os.makedirs(ARTEFACTS_DIR, exist_ok=True)

df = pd.read_csv(NHANES_PATH)
X  = df[FEATURE_COLS_NORACE].values.astype('float32')
y  = df[LABEL_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, GLOBAL_SCALER_NORACE_PATH)

print(f"  Dropped   : {sorted(set(FEATURE_COLS) - set(FEATURE_COLS_NORACE))}")
print(f"  Features  : {len(FEATURE_COLS_NORACE)}  {FEATURE_COLS_NORACE}")
print(f"  Fitted on : {len(X_train):,} NHANES training rows")
for col, m, s in zip(FEATURE_COLS_NORACE, scaler.mean_, scaler.scale_):
    print(f"    {col:<16} mean={m:9.4f}  std={s:8.4f}")

X_sc = scaler.transform(X_train)
assert np.abs(X_sc.mean(axis=0)).max() < 1e-5, "scaled train mean is not ~0"

# The retained columns must be scaled identically to the 8-feature scaler —
# dropping a column should not perturb the others' statistics.
old = joblib.load(GLOBAL_SCALER_PATH)
keep = [FEATURE_COLS.index(c) for c in FEATURE_COLS_NORACE]
assert np.allclose(scaler.mean_,  old.mean_[keep],  rtol=1e-6), "means drifted"
assert np.allclose(scaler.scale_, old.scale_[keep], rtol=1e-6), "scales drifted"
print("\n  Checks passed: scaled train mean ~0, and retained columns match the "
      "8-feature scaler exactly.")
print(f"  Saved -> {GLOBAL_SCALER_NORACE_PATH}")
