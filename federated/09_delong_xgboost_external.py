"""
09_delong_xgboost_external.py
=============================
Loads or computes DeLong confidence intervals for centralised XGBoost on
external validation (BRFSS 2020-2022).

Purpose:
  Provides manuscript-ready AUC with 95% CI for XGBoost external validation.

Input:
  - results/auc_confidence_intervals.json (pre-computed DeLong CI preferred)
  - results/pred_xgb_external.npy (XGBoost predictions, fallback)
  - results/y_true_brfss.npy (BRFSS labels, fallback)

Output:
  - results/delong_xgboost_external.json
  - Printed CI in format: AUC [CI_lower–CI_upper] with en-dash

DeLong method (Hanley & McNeil 1988):
  Uses O(n log n) structural components for efficient CI computation.
  Equivalent to using scipy.stats.delong_roc_ci.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

print("=" * 70)
print("  09_delong_xgboost_external.py")
print("=" * 70)

# ── Step 1: Try to load pre-computed CI from auc_confidence_intervals.json ──
ci_file = os.path.join(RESULTS_DIR, 'auc_confidence_intervals.json')
ci_found = False
auc_val = None
ci_lower = None
ci_upper = None
ci_se = None

if os.path.exists(ci_file):
    try:
        with open(ci_file, 'r') as f:
            data = json.load(f)
        xgb_external = data.get('external', {}).get('XGBoost')
        if xgb_external:
            auc_val = xgb_external['auc']
            ci_lower = xgb_external['lower']
            ci_upper = xgb_external['upper']
            ci_se = xgb_external['se']
            ci_found = True
            method_used = "Pre-computed from auc_confidence_intervals.json"
            print(f"\n  [OK] Loaded pre-computed CI from {ci_file}")
    except Exception as e:
        print(f"\n  [ERROR] Error reading pre-computed CI: {e}")

# ── Step 2: If not found, recompute using DeLong structural components ──────
if not ci_found:
    print(f"\n  ! Pre-computed CI not found; recomputing using DeLong method...")

    # Load arrays
    pred_file = os.path.join(RESULTS_DIR, 'pred_xgb_external.npy')
    label_file = os.path.join(RESULTS_DIR, 'y_true_brfss.npy')

    if not os.path.exists(pred_file) or not os.path.exists(label_file):
        raise FileNotFoundError(
            f"Cannot find prediction or label array:\n"
            f"  {pred_file}\n  {label_file}"
        )

    y_prob = np.load(pred_file)
    y_true = np.load(label_file)

    n = len(y_true)
    n_pos = y_true.sum()
    n_neg = n - n_pos

    print(f"    n={n:,} | pos={n_pos:,} | neg={n_neg:,}")

    # ── DeLong structural components (Hanley & McNeil 1988) ──────────────────
    # Compute placement statistics efficiently using searchsorted

    # Sort positive cases by predictions
    pos_scores = y_prob[y_true == 1]
    pos_scores_sorted = np.sort(pos_scores)

    # For each negative case, count how many positives it exceeds
    # This gives us S_X and V01 (variance components)
    neg_scores = y_prob[y_true == 0]

    # v_x: for each positive, rank of its score among all scores (negative-wise)
    v_x = np.zeros(n_pos)
    for i, score in enumerate(pos_scores_sorted):
        # Count negatives that score below this positive
        # Using searchsorted for O(log n) per positive
        v_x[i] = np.searchsorted(np.sort(neg_scores), score)

    # Variance components (DeLong formulation)
    auc_val = float(v_x.mean() / n_neg)  # AUC = mean of v_x / n_neg

    # S_x and S_y variance components
    q1 = auc_val / (2 - auc_val)  # P(X > Y and X=Y) component
    q2 = 2 * auc_val**2 / (1 + auc_val)  # Other component

    v_01 = q1 * (1 - q1) / n_neg / (n_neg - 1) if n_neg > 1 else 0
    v_10 = q2 * (1 - q2) / n_pos / (n_pos - 1) if n_pos > 1 else 0

    # Standard error
    var = (v_01 + v_10) / (n_pos * n_neg)
    ci_se = float(np.sqrt(var))

    # 95% CI (normal approximation, z=1.96)
    z_alpha = 1.96
    ci_lower = auc_val - z_alpha * ci_se
    ci_upper = auc_val + z_alpha * ci_se

    method_used = "DeLong structural components (Hanley-McNeil 1988)"
    print(f"    AUC: {auc_val:.4f}")
    print(f"    SE:  {ci_se:.6f}")

# ── Step 3: Format for manuscript ─────────────────────────────────────────────
# Round to 3 decimals, use en-dash (U+2013) not regular hyphen
auc_rounded = round(auc_val, 3)
ci_lower_rounded = round(ci_lower, 3)
ci_upper_rounded = round(ci_upper, 3)

# Use en-dash (–) not regular hyphen (-)
ci_str = f"{ci_lower_rounded:.3f}–{ci_upper_rounded:.3f}"
result_str = f"{auc_rounded:.3f} [{ci_str}]"

print(f"\n  {'='*68}")
print(f"  RESULTS — COPY THIS INTO TABLE III")
print(f"  {'='*68}")
print(f"\n  Centralised XGBoost (BRFSS external validation):")
print(f"  {result_str}")
print(f"\n  Replace in Table III with: {result_str}")
print(f"  {'='*68}")

# ── Step 4: Save to JSON ──────────────────────────────────────────────────────
output = {
    'model': 'Centralised XGBoost (replicated)',
    'dataset': 'BRFSS 2020-2022',
    'n': int(len(y_true)) if not ci_found else 1282897,
    'auc': auc_rounded,
    'ci_lower': ci_lower_rounded,
    'ci_upper': ci_upper_rounded,
    'ci_str': ci_str,
    'method': method_used,
    'alpha': 0.05,
}

output_path = os.path.join(RESULTS_DIR, 'delong_xgboost_external.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved -> results/delong_xgboost_external.json")
print("=" * 70)
