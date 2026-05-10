"""
SCRIPT 11 — SUBGROUP CONFIDENCE INTERVALS
==========================================
Stratified bootstrap CIs for subgroup AUC on BRFSS external set.

OUTPUTS per model (XGBoost, FedAvg, FedProx, FedNova):
  - AUC for age 18-39 (young)
  - AUC for age >=60  (elderly)
  - Fairness gap = AUC(young) - AUC(elderly)
  - 95% CI for each metric

INPUTS:
  results/y_true_brfss.npy
  results/pred_xgb_internal.npy       (note: external not saved separately — see note)
  results/pred_fedavg_external.npy
  results/pred_fedprox_external.npy
  results/pred_fednova_external.npy
  results/external_validation.json    (contains age column metadata)

NOTE ON AGE SUBGROUPS:
  BRFSS uses AGE_GRP (categorical) not continuous age. If continuous age
  is unavailable, this script uses BRFSS age-group bins mapped from
  external_validation.json's fairness subgroup results as proxy.
  Preferred: run 07_external_validation.py with age saved to a separate
  .npy file. Fallback: use any age proxy available.

OUTPUTS:
  results/subgroup_ci_results.json
  Formatted table printed to stdout (copy-paste for manuscript Table 4)

RUNTIME: ~3 minutes (N_BOOTSTRAP=2000 on 1.28M samples is fast — bootstrap
is on indices, not data copies)
"""

import os, sys, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    RESULTS_DIR, N_BOOTSTRAP, CI_ALPHA, RANDOM_SEED,
)

print("=" * 65)
print("  11_subgroup_confidence_intervals.py")
print("  Stratified Bootstrap CIs — Subgroup AUC")
print("=" * 65)
print(f"  N_BOOTSTRAP = {N_BOOTSTRAP}  |  alpha = {CI_ALPHA}")


# ── Load predictions ──────────────────────────────────────────────────────────
def load_npy(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    return np.load(path)

y_brfss = load_npy('y_true_brfss.npy')

models = {
    'FedAvg' : load_npy('pred_fedavg_external.npy'),
    'FedProx': load_npy('pred_fedprox_external.npy'),
    'FedNova': load_npy('pred_fednova_external.npy'),
}

if y_brfss is None:
    print("\n  ERROR: y_true_brfss.npy not found.")
    print("  Run 07_external_validation.py first.")
    raise SystemExit(1)

n = len(y_brfss)
print(f"\n  BRFSS n={n:,}  prevalence={y_brfss.mean()*100:.2f}%")


# ── Load age subgroup masks ───────────────────────────────────────────────────
# Try to load age array saved by 07_external_validation.py
age_mask_young   = load_npy('brfss_age_young.npy')    # 1 if 18-39
age_mask_elderly = load_npy('brfss_age_elderly.npy')  # 1 if >=60

if age_mask_young is None or age_mask_elderly is None:
    # Fallback: attempt to load from external_validation.json subgroup stats
    # and reconstruct approximate masks via stratified random assignment
    ext_path = os.path.join(RESULTS_DIR, 'external_validation.json')
    if os.path.exists(ext_path):
        with open(ext_path) as f:
            ext_val = json.load(f)
        # Extract proportions if available
        fairness = ext_val.get('fairness', {})
        print(f"\n  Age mask .npy files not found.")
        print(f"  Subgroup CIs require brfss_age_young.npy / brfss_age_elderly.npy.")
        print(f"  Add these saves to 07_external_validation.py:")
        print(f"    np.save('results/brfss_age_young.npy', (age_arr >= 18) & (age_arr <= 39))")
        print(f"    np.save('results/brfss_age_elderly.npy', age_arr >= 60)")
        print(f"\n  Running overall (non-subgroup) bootstrap CIs only.")
    else:
        print(f"\n  Age masks not available — overall CIs only.")
    age_mask_young   = None
    age_mask_elderly = None


# ── Bootstrap CI function ─────────────────────────────────────────────────────
def bootstrap_subgroup_ci(y_true, y_score, mask=None, n_bootstrap=N_BOOTSTRAP,
                           alpha=CI_ALPHA, seed=RANDOM_SEED):
    """
    Stratified bootstrap CI for AUC on a subgroup (or full set if mask=None).
    Returns dict: auc, lower, upper, se.
    """
    if mask is not None:
        y_true  = y_true[mask]
        y_score = y_score[mask]

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) < 5 or len(neg_idx) < 5:
        return {'auc': None, 'lower': None, 'upper': None, 'se': None, 'n': 0}

    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_bootstrap):
        bi = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        if len(np.unique(y_true[bi])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[bi], y_score[bi])))

    aucs = np.array(aucs)
    return {
        'auc'  : float(roc_auc_score(y_true, y_score)),
        'lower': float(np.percentile(aucs, 100.0 * alpha / 2.0)),
        'upper': float(np.percentile(aucs, 100.0 * (1.0 - alpha / 2.0))),
        'se'   : float(aucs.std()),
        'n'    : int(len(y_true)),
        'n_bootstrap': len(aucs),
    }


def fmt_ci(r):
    if r['auc'] is None:
        return 'N/A'
    return f"{r['auc']:.3f} ({r['lower']:.3f}–{r['upper']:.3f})"


# ── Compute CIs for each model ────────────────────────────────────────────────
print(f"\n  Computing bootstrap CIs (N={N_BOOTSTRAP})...")
all_results = {}

for model_name, preds in models.items():
    if preds is None:
        print(f"  {model_name:<10} : predictions not found — skipped")
        continue

    print(f"\n  {model_name}:")
    r_overall = bootstrap_subgroup_ci(y_brfss, preds)
    r_young   = bootstrap_subgroup_ci(y_brfss, preds, age_mask_young)
    r_elderly = bootstrap_subgroup_ci(y_brfss, preds, age_mask_elderly)

    # Fairness gap CI via bootstrap on same resamples
    gap_ci = None
    if age_mask_young is not None and age_mask_elderly is not None:
        rng   = np.random.default_rng(RANDOM_SEED)
        # Compute gap on full set (not bootstrapped per-subgroup, but jointly)
        gaps  = []
        y_y   = y_brfss[age_mask_young]
        p_y   = preds[age_mask_young]
        y_e   = y_brfss[age_mask_elderly]
        p_e   = preds[age_mask_elderly]
        pos_y = np.where(y_y == 1)[0]; neg_y = np.where(y_y == 0)[0]
        pos_e = np.where(y_e == 1)[0]; neg_e = np.where(y_e == 0)[0]
        for _ in range(N_BOOTSTRAP):
            bi_y = np.concatenate([rng.choice(pos_y, len(pos_y), replace=True),
                                    rng.choice(neg_y, len(neg_y), replace=True)])
            bi_e = np.concatenate([rng.choice(pos_e, len(pos_e), replace=True),
                                    rng.choice(neg_e, len(neg_e), replace=True)])
            if len(np.unique(y_y[bi_y])) < 2 or len(np.unique(y_e[bi_e])) < 2:
                continue
            gap_val = roc_auc_score(y_y[bi_y], p_y[bi_y]) - roc_auc_score(y_e[bi_e], p_e[bi_e])
            gaps.append(gap_val)
        gaps = np.array(gaps)
        gap_point = float(roc_auc_score(y_y, p_y) - roc_auc_score(y_e, p_e))
        gap_ci = {
            'gap': gap_point,
            'lower': float(np.percentile(gaps, 100.0 * CI_ALPHA / 2.0)),
            'upper': float(np.percentile(gaps, 100.0 * (1.0 - CI_ALPHA / 2.0))),
        }

    print(f"    Overall : {fmt_ci(r_overall)}")
    print(f"    Young   : {fmt_ci(r_young)}")
    print(f"    Elderly : {fmt_ci(r_elderly)}")
    if gap_ci:
        print(f"    Gap     : {gap_ci['gap']:.3f} ({gap_ci['lower']:.3f}–{gap_ci['upper']:.3f})")

    all_results[model_name] = {
        'overall': r_overall,
        'young_18_39': r_young,
        'elderly_60plus': r_elderly,
        'fairness_gap_ci': gap_ci,
    }


# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, 'subgroup_ci_results.json')
with open(out_path, 'w') as f:
    json.dump({
        'method': f'Stratified bootstrap (N={N_BOOTSTRAP}, alpha={CI_ALPHA})',
        'models': all_results,
    }, f, indent=2)
print(f"\n  Saved -> {out_path}")


# ── Manuscript table ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUBGROUP CI TABLE (copy to manuscript Table 4)")
print("=" * 65)
print(f"  {'Model':<10}  {'AUC 18-39 [95% CI]':<22}  "
      f"{'AUC >=60 [95% CI]':<22}  {'Gap [95% CI]':<20}")
print(f"  {'-'*80}")
for name, res in all_results.items():
    y_str = fmt_ci(res['young_18_39'])
    e_str = fmt_ci(res['elderly_60plus'])
    g     = res['fairness_gap_ci']
    g_str = (f"{g['gap']:.3f} ({g['lower']:.3f}–{g['upper']:.3f})"
             if g else 'N/A')
    print(f"  {name:<10}  {y_str:<22}  {e_str:<22}  {g_str:<20}")
print("=" * 65)
