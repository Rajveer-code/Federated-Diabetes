"""
07_statistical_analysis.py
===========================
Run AFTER all models have been trained and predictions saved.

PURPOSE
-------
Produces 95% confidence intervals and paired significance tests for
all AUC values in the paper.  Without CIs, a top-10% reviewer will
reject the paper outright.

METHODS
-------
  Internal validation (NHANES test set, n ~ 3,130):
    → Stratified bootstrap CI (N_BOOTSTRAP=2000, percentile method)

  External validation (BRFSS, n = 1,282,897):
    → DeLong CI (structural components, Hanley-McNeil)
    → Paired DeLong test for FedProx vs FedAvg

  References:
    DeLong, DeLong, Clarke-Pearson (1988) Biometrics 44(3):837-845.
    Hanley & McNeil (1982) Radiology 143(1):29-36.

USAGE
-----
  cd D:\\Projects\\diabetes_prediction_project\\federated

  # Save predictions first (add save calls to each training script):
  #   np.save('results/y_true_internal.npy',       y_true)
  #   np.save('results/pred_fedavg_internal.npy',   y_score_fedavg)
  #   np.save('results/pred_fedprox_internal.npy',  y_score_fedprox)
  #   np.save('results/pred_fednova_internal.npy',  y_score_fednova)
  #   np.save('results/pred_xgb_internal.npy',      y_score_xgb)
  #   np.save('results/y_true_brfss.npy',           y_true_brfss)
  #   np.save('results/pred_fedprox_external.npy',  y_score_fedprox_ext)
  #   np.save('results/pred_fedavg_external.npy',   y_score_fedavg_ext)
  #   np.save('results/pred_fednova_external.npy',  y_score_fednova_ext)

  python 07_statistical_analysis.py

OUTPUT
------
  results/auc_confidence_intervals.json
  (Numbers printed to stdout are copy-paste-ready for manuscript text.)
"""

import os, sys, json, warnings
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR, N_BOOTSTRAP, CI_ALPHA, RANDOM_SEED


# ──────────────────────────────────────────────────────────────────────────────
#  CORE STATISTICAL FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def delong_ci(y_true, y_score, alpha=CI_ALPHA):
    """
    DeLong 95% CI using structural components (Hanley-McNeil method).
    DeLong, DeLong, Clarke-Pearson (1988) Biometrics 44(3):837-845.

    Use this for EXTERNAL validation (large n — BRFSS n=1,282,897).
    With n=1.28M the SE is ~0.0003-0.0005, so CIs will be very tight.

    Returns dict with auc, lower, upper, se, V10, V01.
    V10 / V01 are needed for the paired test.
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)

    # Structural components (kernel matrix approach)
    diff   = pos_scores[:, None] - neg_scores[None, :]
    kernel = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))

    V10 = kernel.mean(axis=1)           # shape (n_pos,)
    V01 = 1.0 - kernel.mean(axis=0)    # shape (n_neg,)
    auc = float(V10.mean())

    s10 = np.var(V10, ddof=1) / n_pos
    s01 = np.var(V01, ddof=1) / n_neg
    se  = float(np.sqrt(s10 + s01))
    z   = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        'auc'  : auc,
        'lower': float(max(0.0, auc - z * se)),
        'upper': float(min(1.0, auc + z * se)),
        'se'   : se,
        'V10'  : V10,   # kept for paired test
        'V01'  : V01,
    }


def delong_paired_test(y_true, y_score_1, y_score_2):
    """
    Paired DeLong test comparing two correlated AUCs on the SAME test set.
    Returns two-sided p-value for H0: AUC1 = AUC2.

    Use this to compare FedProx vs FedAvg on the external BRFSS set.
    With n=1.28M, the 0.011 AUC gap WILL be statistically significant —
    report both p-value and absolute effect size and note the difference
    is modest (~1.1 pp) despite strong significance.
    """
    r1    = delong_ci(y_true, y_score_1)
    r2    = delong_ci(y_true, y_score_2)
    n_pos = int((np.asarray(y_true) == 1).sum())
    n_neg = int((np.asarray(y_true) == 0).sum())

    cov_12 = (
        np.cov(r1['V10'], r2['V10'], ddof=1)[0, 1] / n_pos
        + np.cov(r1['V01'], r2['V01'], ddof=1)[0, 1] / n_neg
    )
    var_diff = r1['se'] ** 2 + r2['se'] ** 2 - 2.0 * cov_12

    if var_diff <= 0:
        return {
            'auc1'   : r1['auc'],
            'auc2'   : r2['auc'],
            'delta'  : r1['auc'] - r2['auc'],
            'p_value': 1.0,
            'note'   : 'var_diff<=0: models may produce identical scores',
        }

    z = (r1['auc'] - r2['auc']) / np.sqrt(var_diff)
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return {
        'auc1'      : r1['auc'],
        'auc2'      : r2['auc'],
        'delta'     : float(r1['auc'] - r2['auc']),
        'z'         : float(z),
        'p_value'   : p,
        'significant': p < CI_ALPHA,
    }


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP,
                     alpha=CI_ALPHA, seed=RANDOM_SEED):
    """
    Stratified bootstrap CI for AUC (percentile method).

    Use this for INTERNAL validation (small n — NHANES test set ~3,130).
    Stratified = positive and negative cases are resampled separately to
    maintain prevalence across bootstrap samples.
    """
    rng    = np.random.default_rng(seed)
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

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
        'n_bootstrap': len(aucs),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def fmt_ci(r):
    """Format AUC + CI for manuscript text: '0.756 (95% CI: 0.755-0.757)'"""
    return f"{r['auc']:.3f} (95% CI: {r['lower']:.3f}\u2013{r['upper']:.3f})"


def load_npy(filename):
    """Load a .npy prediction file from RESULTS_DIR, return None if missing."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    return np.load(path)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  07_statistical_analysis.py")
print("  DeLong CIs + Bootstrap CIs + Paired Tests")
print("=" * 65)

# ── Load prediction arrays ────────────────────────────────────────────────────
print("\n[1/4] Loading prediction arrays from results/...")

y_true_int     = load_npy('y_true_internal.npy')
pred_fedavg    = load_npy('pred_fedavg_internal.npy')
pred_fedprox   = load_npy('pred_fedprox_internal.npy')
pred_fednova   = load_npy('pred_fednova_internal.npy')
pred_xgb       = load_npy('pred_xgb_internal.npy')

y_true_brfss   = load_npy('y_true_brfss.npy')
pred_fedprox_x = load_npy('pred_fedprox_external.npy')
pred_fedavg_x  = load_npy('pred_fedavg_external.npy')
pred_fednova_x = load_npy('pred_fednova_external.npy')

# ── Internal AUCs (Bootstrap CI) ─────────────────────────────────────────────
print("\n[2/4] Internal AUC (Bootstrap CI, N_BOOTSTRAP={})...".format(N_BOOTSTRAP))
int_results = {}

internal_models = [
    ('FedAvg',  pred_fedavg),
    ('FedProx', pred_fedprox),
    ('FedNova', pred_fednova),
    ('XGBoost', pred_xgb),
]

if y_true_int is not None:
    for name, preds in internal_models:
        if preds is None:
            print(f"  {name:<12}  [skipped — predictions not found]")
            continue
        r = bootstrap_auc_ci(y_true_int, preds)
        int_results[name] = r
        print(f"  {name:<12}  AUC = {fmt_ci(r)}  (SE={r['se']:.4f}, "
              f"n_boot={r['n_bootstrap']})")
else:
    print("  Skipped — y_true_internal.npy not found.")
    print("  Add np.save('results/y_true_internal.npy', y_true) to training scripts.")

# ── External AUCs (DeLong CI) ─────────────────────────────────────────────────
print("\n[3/4] External AUC (DeLong CI, large-n BRFSS)...")
ext_results = {}

external_models = [
    ('FedProx', pred_fedprox_x),
    ('FedAvg',  pred_fedavg_x),
    ('FedNova', pred_fednova_x),
]

if y_true_brfss is not None:
    for name, preds in external_models:
        if preds is None:
            print(f"  {name:<12}  [skipped — predictions not found]")
            continue
        r = delong_ci(y_true_brfss, preds)
        ext_results[name] = {k: v for k, v in r.items()
                             if k not in ('V10', 'V01')}   # omit large arrays
        print(f"  {name:<12}  AUC = {fmt_ci(r)}  (SE={r['se']:.5f})")

    # ── Paired DeLong test: FedProx vs FedAvg ─────────────────────────────────
    print("\n[4/4] Paired DeLong test: FedProx vs FedAvg (external)...")
    paired_result = None
    if pred_fedprox_x is not None and pred_fedavg_x is not None:
        t = delong_paired_test(y_true_brfss, pred_fedprox_x, pred_fedavg_x)
        paired_result = t
        print(f"  FedProx AUC : {t['auc1']:.4f}")
        print(f"  FedAvg  AUC : {t['auc2']:.4f}")
        print(f"  delta       : {t['delta']:+.4f}")
        print(f"  z-statistic : {t['z']:.3f}")
        print(f"  p-value     : {t['p_value']:.6f}")
        sig = "YES" if t['significant'] else "NO"
        print(f"  Significant at alpha={CI_ALPHA}: {sig}")
        if t['significant'] and abs(t['delta']) < 0.02:
            print(f"\n  NOTE: p < {CI_ALPHA} with |delta| = {abs(t['delta']):.3f}")
            print(f"  Statistical significance does NOT imply clinical importance.")
            print(f"  In manuscript: frame as 'modest but statistically robust'.")
    else:
        print("  Skipped — external prediction files not found.")
else:
    print("  Skipped — y_true_brfss.npy not found.")
    print("  Run 07_external_validation.py and add np.save calls first.")
    paired_result = None

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    'method_internal': f'Stratified bootstrap (N={N_BOOTSTRAP}, alpha={CI_ALPHA})',
    'method_external': 'DeLong structural components (Hanley-McNeil 1988)',
    'alpha'          : CI_ALPHA,
    'internal'       : int_results,
    'external'       : ext_results,
    'paired_test_fedprox_vs_fedavg_external': paired_result,
}

out_path = os.path.join(RESULTS_DIR, 'auc_confidence_intervals.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved → {out_path}")

# ── Manuscript summary ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MANUSCRIPT NUMBER SUMMARY")
print("  Copy these into the paper — replace X.XXX placeholders")
print("=" * 65)

if int_results:
    print("\n  === Internal AUC (NHANES test set) ===")
    for name, r in int_results.items():
        print(f"  {name:<12}  {fmt_ci(r)}")

if ext_results:
    print("\n  === External AUC (BRFSS) ===")
    for name, r in ext_results.items():
        print(f"  {name:<12}  {fmt_ci(r)}")

if paired_result and 'z' in paired_result:
    print(f"\n  === Paired DeLong: FedProx vs FedAvg ===")
    print(f"  delta = {paired_result['delta']:+.4f}, "
          f"z = {paired_result['z']:.3f}, "
          f"p = {paired_result['p_value']:.6f}")

print("\n" + "=" * 65)
print("  All CIs computed. Proceed to manuscript text edits.")
print("=" * 65)
