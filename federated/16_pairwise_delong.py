"""
16_pairwise_delong.py
======================
Computes all three pairwise DeLong tests between FL strategies on BRFSS.
Combines with existing FedAvg-vs-FedProx result.

Saves: results/pairwise_delong_all.json
"""

import os, sys, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR, CI_ALPHA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

print("=" * 65)
print("  16 — Pairwise DeLong Tests (BRFSS External)")
print("=" * 65)

# Import delong_paired_test from 07_statistical_analysis
# Can't import directly due to module-level execution, so replicate the function
from scipy import stats

def delong_ci(y_true, y_score, alpha=CI_ALPHA):
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)

    neg_sorted = np.sort(neg_scores)
    n_below = np.searchsorted(neg_sorted, pos_scores, side='left')
    n_equal = np.searchsorted(neg_sorted, pos_scores, side='right') - n_below
    V10 = (n_below.astype(np.float64) + 0.5 * n_equal) / n_neg

    pos_sorted = np.sort(pos_scores)
    n_above    = n_pos - np.searchsorted(pos_sorted, neg_scores, side='right')
    n_equal_01 = (np.searchsorted(pos_sorted, neg_scores, side='right') -
                  np.searchsorted(pos_sorted, neg_scores, side='left'))
    V01 = (n_above.astype(np.float64) + 0.5 * n_equal_01) / n_pos

    auc = float(V10.mean())
    s10 = np.var(V10, ddof=1) / n_pos
    s01 = np.var(V01, ddof=1) / n_neg
    se  = float(np.sqrt(s10 + s01))
    z   = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        'auc': auc, 'lower': float(max(0.0, auc - z * se)),
        'upper': float(min(1.0, auc + z * se)), 'se': se,
        'V10': V10, 'V01': V01,
    }


def delong_paired_test(y_true, y_score_1, y_score_2):
    r1 = delong_ci(y_true, y_score_1)
    r2 = delong_ci(y_true, y_score_2)
    n_pos = int((np.asarray(y_true) == 1).sum())
    n_neg = int((np.asarray(y_true) == 0).sum())

    cov_12 = (
        np.cov(r1['V10'], r2['V10'], ddof=1)[0, 1] / n_pos
        + np.cov(r1['V01'], r2['V01'], ddof=1)[0, 1] / n_neg
    )
    var_diff = r1['se'] ** 2 + r2['se'] ** 2 - 2.0 * cov_12
    if var_diff <= 0:
        return {
            'auc1': r1['auc'], 'auc2': r2['auc'],
            'delta': r1['auc'] - r2['auc'], 'z': 0.0,
            'p_value': 1.0, 'significant': False,
        }
    z = (r1['auc'] - r2['auc']) / np.sqrt(var_diff)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return {
        'auc1': r1['auc'], 'auc2': r2['auc'],
        'delta': float(r1['auc'] - r2['auc']),
        'z': float(z), 'p_value': p,
        'significant': p < CI_ALPHA,
    }


# Load prediction arrays
def load_npy(fname):
    path = os.path.join(RESULTS_DIR, fname)
    return np.load(path)

print("\n  Loading prediction arrays...")
y_true     = load_npy('y_true_brfss.npy')
p_fedavg   = load_npy('pred_fedavg_external.npy')
p_fedprox  = load_npy('pred_fedprox_external.npy')
p_fednova  = load_npy('pred_fednova_external.npy')
print(f"  n = {len(y_true):,}")

bonferroni_alpha = CI_ALPHA / 3.0

comparisons = [
    ('FedAvg_vs_FedProx', p_fedavg, p_fedprox),
    ('FedAvg_vs_FedNova', p_fedavg, p_fednova),
    ('FedProx_vs_FedNova', p_fedprox, p_fednova),
]

results = {}
for name, p1, p2 in comparisons:
    print(f"\n  Testing: {name}...")
    t = delong_paired_test(y_true, p1, p2)
    t['significant_bonferroni'] = t['p_value'] < bonferroni_alpha
    results[name] = t
    sig_str = "YES" if t['significant_bonferroni'] else "NO"
    print(f"    AUC1={t['auc1']:.4f}  AUC2={t['auc2']:.4f}  "
          f"delta={t['delta']:+.4f}  z={t['z']:.3f}  p={t['p_value']:.2e}  "
          f"Bonferroni-sig: {sig_str}")

out = {
    'method': 'Paired DeLong (two-sided)',
    'dataset': 'BRFSS 2020-2022',
    'n': int(len(y_true)),
    'bonferroni_alpha': bonferroni_alpha,
    'n_comparisons': 3,
    'tests': results,
}
out_path = os.path.join(RESULTS_DIR, 'pairwise_delong_all.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*65}")
print(f"  All 3 pairwise DeLong tests complete")
print(f"  Bonferroni-corrected alpha = {bonferroni_alpha:.4f}")
print(f"  Saved: {out_path}")
print(f"{'='*65}")
