"""
08_calibration_analysis.py
===========================
Computes ECE with both equal-width and equal-frequency (adaptive) bins.
Also computes calibration slope and intercept.
Run AFTER 07_external_validation.py.

Output: results/calibration_extended.json
        Prints all numbers needed for manuscript Section 4.4
"""
import os, sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

print("=" * 60)
print("  08_calibration_analysis.py")
print("=" * 60)

# Load predictions
y_true = np.load(os.path.join(RESULTS_DIR, 'y_true_brfss.npy'))
y_prob = np.load(os.path.join(RESULTS_DIR, 'pred_fedavg_external.npy'))

print(f"\n  n = {len(y_true):,} | prevalence = {y_true.mean():.1%}")
print(f"  y_prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

# ── 1. Equal-width ECE (10 bins) — what was reported in paper ──────────────
def ece_equal_width(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, bin_info = 0.0, []
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i+1])
        if mask.sum() == 0:
            continue
        frac_pos  = float(y_true[mask].mean())
        mean_pred = float(y_prob[mask].mean())
        bin_info.append({
            'bin': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
            'n': int(mask.sum()),
            'mean_pred': mean_pred,
            'frac_pos': frac_pos,
            'contribution': float(mask.sum() * abs(frac_pos - mean_pred) / n),
        })
        ece += mask.sum() * abs(frac_pos - mean_pred)
    return ece / n, bin_info

# ── 2. Equal-frequency ECE (10 decile bins) ────────────────────────────────
def ece_equal_freq(y_true, y_prob, n_bins=10):
    quantiles = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    ece, bin_info = 0.0, []
    n = len(y_true)
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            mask = (y_prob >= quantiles[i]) & (y_prob < quantiles[i+1])
        else:
            mask = (y_prob >= quantiles[i]) & (y_prob <= quantiles[i+1])
        if mask.sum() == 0:
            continue
        frac_pos  = float(y_true[mask].mean())
        mean_pred = float(y_prob[mask].mean())
        bin_info.append({
            'bin': f"{quantiles[i]:.3f}-{quantiles[i+1]:.3f}",
            'n': int(mask.sum()),
            'mean_pred': mean_pred,
            'frac_pos': frac_pos,
        })
        ece += mask.sum() * abs(frac_pos - mean_pred)
    return ece / n, bin_info

# ── 3. Calibration slope and intercept ─────────────────────────────────────
log_odds = np.log(y_prob.clip(1e-7, 1 - 1e-7) / (1 - y_prob.clip(1e-7, 1 - 1e-7)))
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(log_odds.reshape(-1, 1), y_true)
cal_slope     = float(lr.coef_[0][0])
cal_intercept = float(lr.intercept_[0])

# ── 4. Brier score decomposition ───────────────────────────────────────────
prev      = float(y_true.mean())
brier     = float(np.mean((y_prob - y_true) ** 2))
uncert    = float(prev * (1 - prev))
# Use 10 equal-width bins for decomposition (matching reported values)
_, bins_w = ece_equal_width(y_true, y_prob, n_bins=10)
reliability = float(sum(b['contribution'] for b in bins_w))  # approx
# Standard decomposition
resolution = 0.0
for b in bins_w:
    resolution += (b['n'] / len(y_true)) * (b['frac_pos'] - prev) ** 2

# ── Run everything ──────────────────────────────────────────────────────────
ece_w, bins_info_w = ece_equal_width(y_true, y_prob, n_bins=10)
ece_f, bins_info_f = ece_equal_freq(y_true,  y_prob, n_bins=10)

print(f"\n  {'='*50}")
print(f"  RESULTS — COPY THESE INTO MANUSCRIPT")
print(f"  {'='*50}")
print(f"  ECE (equal-width,     10 bins): {ece_w:.4f}  ← already in paper as 0.276")
print(f"  ECE (equal-frequency, 10 bins): {ece_f:.4f}  ← ADD THIS TO §4.4")
print(f"  Calibration slope:              {cal_slope:.4f}  ← ADD TO §4.4")
print(f"  Calibration intercept:          {cal_intercept:.4f}")
print(f"  Brier score:                    {brier:.4f}")
print(f"  Uncertainty:                    {uncert:.4f}")
print(f"\n  Equal-width bin detail (for checking):")
for b in bins_info_w:
    print(f"    Bin {b['bin']:12s}  n={b['n']:>8,}  "
          f"pred={b['mean_pred']:.3f}  actual={b['frac_pos']:.3f}  "
          f"diff={abs(b['mean_pred']-b['frac_pos']):.3f}")

# Save
results = {
    'ece_equal_width_10':   round(ece_w, 6),
    'ece_equal_freq_10':    round(ece_f, 6),
    'calibration_slope':    round(cal_slope, 6),
    'calibration_intercept':round(cal_intercept, 6),
    'brier':                round(brier, 6),
    'uncertainty':          round(uncert, 6),
    'bins_equal_width':     bins_info_w,
    'bins_equal_freq':      bins_info_f,
}
out = os.path.join(RESULTS_DIR, 'calibration_extended.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved → results/calibration_extended.json")
print("=" * 60)