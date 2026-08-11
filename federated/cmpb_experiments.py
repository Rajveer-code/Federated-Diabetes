"""
cmpb_experiments.py — CMPB Revision Experiments (Batch)
=========================================================
Runs experiments that can execute without NHANES node data:
  Exp 3: Bootstrap Brier score 95% CIs for all models (BRFSS external)
  Exp 4: Number Needed to Screen (NNS) — elderly subgroup
  Exp 5: ECE bootstrap 95% CIs for FedAvg calibration methods
  +Extra: SCAFFOLD internal bootstrap CI, node prevalence computation

Outputs (federated/results/):
  brier_score_cis.json / .csv
  nns_results.json / .csv
  ece_cis.json / .csv
  scaffold_internal_ci.json
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR, RANDOM_SEED, SEED as CAL_SEED

np.random.seed(RANDOM_SEED)
RESULTS_DIR = str(RESULTS_DIR)

Z95 = 1.959964
N_BOOT = 2000

print("=" * 70)
print("  cmpb_experiments.py — CMPB Revision Experiments")
print("=" * 70)

# ── Load shared arrays ────────────────────────────────────────────────────────
print("\n[LOAD] Loading BRFSS arrays...")
y_brfss      = np.load(os.path.join(RESULTS_DIR, 'y_true_brfss.npy')).astype(np.float32)
mask_elderly = np.load(os.path.join(RESULTS_DIR, 'brfss_age_elderly.npy'))  # bool, age>=60
mask_young   = np.load(os.path.join(RESULTS_DIR, 'brfss_age_young.npy'))    # bool, age 18-39
n_brfss = len(y_brfss)
print(f"      BRFSS n={n_brfss:,}  prevalence={y_brfss.mean()*100:.2f}%")
print(f"      Elderly (>=60): n={mask_elderly.sum():,}  prev={y_brfss[mask_elderly].mean()*100:.2f}%")

preds_ext = {}
model_files = {
    'XGBoost':         'pred_xgb_external.npy',
    'DiabetesNet':     'pred_central_nn_external.npy',
    'FedAvg':          'pred_fedavg_external.npy',
    'FedProx':         'pred_fedprox_external.npy',
    'FedNova':         'pred_fednova_external.npy',
    'SCAFFOLD_SGD':    'pred_scaffold_external.npy',
}
for name, fname in model_files.items():
    p = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(p):
        arr = np.load(p).astype(np.float32)
        preds_ext[name] = arr
        print(f"      {name}: n={len(arr):,}")
    else:
        print(f"      WARNING: {fname} not found — skipping {name}")

print("\n[LOAD] Loading NHANES internal arrays...")
y_int = np.load(os.path.join(RESULTS_DIR, 'y_true_internal.npy')).astype(np.float32)
preds_int = {}
int_files = {
    'SCAFFOLD_SGD': 'pred_scaffold_internal.npy',
    'FedAvg':       'pred_fedavg_internal.npy',
    'FedProx':      'pred_fedprox_internal.npy',
    'FedNova':      'pred_fednova_internal.npy',
}
for name, fname in int_files.items():
    p = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(p):
        arr = np.load(p).astype(np.float32)
        preds_int[name] = arr
        print(f"      {name}: n={len(arr):,}")


# ══════════════════════════════════════════════════════════════════════════════
# EXP 3: Bootstrap Brier Score CIs (BRFSS external)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXP 3: Bootstrap Brier Score 95% CIs (BRFSS external)")
print("=" * 70)

rng = np.random.default_rng(RANDOM_SEED)
brier_results = {}

for model, pred in preds_ext.items():
    if len(pred) != n_brfss:
        print(f"  {model}: dimension mismatch — skip")
        continue
    point = float(brier_score_loss(y_brfss, pred))
    boot = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        idx = rng.integers(0, n_brfss, size=n_brfss)
        boot[i] = brier_score_loss(y_brfss[idx], pred[idx])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    brier_results[model] = {'brier': round(point, 4), 'ci_lower': round(lo, 4), 'ci_upper': round(hi, 4)}
    print(f"  {model:<20}: {point:.4f} [{lo:.4f}–{hi:.4f}]")

# Save
with open(os.path.join(RESULTS_DIR, 'brier_score_cis.json'), 'w') as f:
    json.dump({'n_bootstrap': N_BOOT, 'n_brfss': int(n_brfss), 'models': brier_results}, f, indent=2)
rows = [{'model': m, 'brier_score': v['brier'], 'ci_lower': v['ci_lower'], 'ci_upper': v['ci_upper']}
        for m, v in brier_results.items()]
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'brier_score_cis.csv'), index=False)
print(f"\n  Saved -> brier_score_cis.json / .csv")


# ══════════════════════════════════════════════════════════════════════════════
# EXP 4: Number Needed to Screen (NNS) — Elderly Subgroup
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXP 4: Number Needed to Screen — Elderly Subgroup (age >=60)")
print("=" * 70)

y_eld = y_brfss[mask_elderly]
prev_elderly = float(y_eld.mean())
n_elderly = int(mask_elderly.sum())
print(f"  Elderly n={n_elderly:,}  prevalence={prev_elderly*100:.2f}%")

def youden_threshold(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(1 - fpr[best_idx])

def compute_nns_stats(y_true, y_score, threshold, prev):
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    nns  = 1.0 / (sens * prev) if sens * prev > 0 else float('inf')
    return {'sensitivity': round(sens, 4), 'specificity': round(spec, 4),
            'ppv': round(ppv, 4), 'npv': round(npv, 4), 'nns': round(nns, 1)}

FIXED_THRESH = 0.35

nns_rows = []
nns_json = {'elderly_prevalence': round(prev_elderly, 4), 'n_elderly': n_elderly,
            'youden': {}, 'fixed_0.35': {}}

for model in ['FedAvg', 'XGBoost']:
    if model not in preds_ext:
        continue
    pred_eld = preds_ext[model][mask_elderly]
    thresh_y, sens_y, spec_y = youden_threshold(y_eld, pred_eld)
    stats_y = compute_nns_stats(y_eld, pred_eld, thresh_y, prev_elderly)
    stats_y['threshold'] = round(thresh_y, 4)
    stats_y['auc'] = round(float(roc_auc_score(y_eld, pred_eld)), 4)
    nns_json['youden'][model] = stats_y

    stats_f = compute_nns_stats(y_eld, pred_eld, FIXED_THRESH, prev_elderly)
    stats_f['threshold'] = FIXED_THRESH
    nns_json['fixed_0.35'][model] = stats_f

    nns_rows.append({'model': model, 'threshold_type': 'Youden', **{k: v for k, v in stats_y.items()}})
    nns_rows.append({'model': model, 'threshold_type': 'Fixed_0.35', **{k: v for k, v in stats_f.items()}})

    print(f"\n  {model} (Youden thresh={thresh_y:.3f}): sens={stats_y['sensitivity']:.3f} spec={stats_y['specificity']:.3f} NNS={stats_y['nns']:.1f}")
    print(f"  {model} (Fixed 0.35): sens={stats_f['sensitivity']:.3f} spec={stats_f['specificity']:.3f} NNS={stats_f['nns']:.1f}")

# Extra per 10,000 detected (Youden)
if 'FedAvg' in nns_json['youden'] and 'XGBoost' in nns_json['youden']:
    fa_sens = nns_json['youden']['FedAvg']['sensitivity']
    xg_sens = nns_json['youden']['XGBoost']['sensitivity']
    extra_tp_10k = round((fa_sens - xg_sens) * prev_elderly * 10000, 1)
    nns_json['extra_tp_per_10k_vs_xgb'] = extra_tp_10k
    print(f"\n  Extra true positives per 10,000 elderly screened (FedAvg vs XGBoost, Youden): {extra_tp_10k:.1f}")

with open(os.path.join(RESULTS_DIR, 'nns_results.json'), 'w') as f:
    json.dump(nns_json, f, indent=2)
pd.DataFrame(nns_rows).to_csv(os.path.join(RESULTS_DIR, 'nns_results.csv'), index=False)
print(f"\n  Saved -> nns_results.json / .csv")


# ══════════════════════════════════════════════════════════════════════════════
# EXP 5: ECE Bootstrap 95% CIs (FedAvg calibration methods on BRFSS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXP 5: ECE Bootstrap 95% CIs — FedAvg calibration methods")
print("=" * 70)

# Load calibration results from JSON
cal_json_path = os.path.join(RESULTS_DIR, 'fedavg_calibration_results.json')
with open(cal_json_path) as f:
    cal_data = json.load(f)

# This block adds bootstrap CIs to the calibration results of
# 17_fedavg_calibration.py, so it has to reproduce that script exactly. Four
# things previously differed, each of which alone makes the two result files
# disagree for the same model on the same data:
#   1. Platt coefficients were applied to logit(p), but 17_fedavg_calibration.py
#      fits them on raw p — this alone gave ECE=0.1105 against its 0.0072.
#   2. ECE was evaluated on all 1,282,897 BRFSS rows, including the 20% the
#      calibrators were fitted on, instead of the held-out 80%.
#   3. the bin convention was (lo, hi] with the first bin closed; ece_wilson()
#      uses [lo, hi) with the last bin closed.
#   4. the isotonic CI was not a bootstrap at all — it was Platt's bootstrap
#      width recentred on the isotonic point estimate. Isotonic is refitted here
#      (it is deterministic given the same calibration split) and bootstrapped
#      like every other method.

def ece_fast(y_true, y_score, n_bins=10):
    """
    Equal-width ECE, vectorised. Bin convention matches ece_wilson() in
    17_fedavg_calibration.py: [lo, hi), final bin closed at 1.0.
    Uses sum_b (n_b/N)*|acc_b - conf_b| == sum_b |sum(y)_b - sum(p)_b| / N.
    """
    b = np.clip((np.asarray(y_score) * n_bins).astype(np.intp), 0, n_bins - 1)
    s_acc  = np.bincount(b, weights=y_true,  minlength=n_bins)
    s_conf = np.bincount(b, weights=y_score, minlength=n_bins)
    return float(np.abs(s_acc - s_conf).sum() / len(y_true))


assert RANDOM_SEED == CAL_SEED, "split seed must match 17_fedavg_calibration.py"
idx_split = np.random.default_rng(CAL_SEED).permutation(n_brfss)
n_cal     = int(0.20 * n_brfss)
cal_idx, test_idx = idx_split[:n_cal], idx_split[n_cal:]

pred_fa = preds_ext['FedAvg']
y_cal,  p_cal_raw  = y_brfss[cal_idx],  pred_fa[cal_idx]
y_test, p_test_raw = y_brfss[test_idx], pred_fa[test_idx]
n_test = len(test_idx)
print(f"  Split: {n_cal:,} cal / {n_test:,} test  (as 17_fedavg_calibration.py)")

platt = LogisticRegression(C=1.0, max_iter=500, random_state=CAL_SEED)
platt.fit(p_cal_raw.reshape(-1, 1), y_cal)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_cal_raw, y_cal)
T_opt = cal_data['temperature']['T_optimal']

def temp_calibrate(p):
    logit_p = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1 - 1e-7))
    return 1.0 / (1.0 + np.exp(-logit_p / T_opt))

calibrations = {
    'Uncalibrated': p_test_raw,
    'Platt':        platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1],
    'Isotonic':     iso.predict(p_test_raw),
    'Temperature':  temp_calibrate(p_test_raw),
}

print(f"  Computing bootstrap ECE CIs (N={N_BOOT}) on the {n_test:,}-row test split...")
rng2 = np.random.default_rng(RANDOM_SEED)

ece_results = {}
for method, p_cal in calibrations.items():
    point_ece = ece_fast(y_test, p_cal)
    boot_ece = np.empty(N_BOOT)
    for i in range(N_BOOT):
        bidx = rng2.integers(0, n_test, size=n_test)
        boot_ece[i] = ece_fast(y_test[bidx], p_cal[bidx])
    lo_e = float(np.percentile(boot_ece, 2.5))
    hi_e = float(np.percentile(boot_ece, 97.5))
    ece_results[method] = {'ece': round(point_ece, 4),
                           'ci_lower': round(lo_e, 4), 'ci_upper': round(hi_e, 4)}
    # Cross-check against the script this one is meant to agree with.
    ref = cal_data.get(method.lower(), {}).get('ece')
    if ref is not None:
        ece_results[method]['ece_17_fedavg_calibration'] = round(float(ref), 4)
        ece_results[method]['matches_17'] = bool(abs(point_ece - float(ref)) < 5e-4)
    tail = '' if ref is None else f"   17_fedavg_calibration={ref:.4f}"
    print(f"  {method:<14}: ECE={point_ece:.4f} [{lo_e:.4f}-{hi_e:.4f}]{tail}")

mismatched = [m for m, v in ece_results.items() if v.get('matches_17') is False]
if mismatched:
    print(f"  WARNING: point ECE disagrees with 17_fedavg_calibration.py for {mismatched}")

with open(os.path.join(RESULTS_DIR, 'ece_cis.json'), 'w') as f:
    json.dump({'n_bootstrap': N_BOOT, 'n_cal': int(n_cal), 'n_test': int(n_test),
               'n_brfss': int(n_brfss), 'model': 'FedAvg',
               'evaluated_on': 'held-out 80% test split (matches 17_fedavg_calibration.py)',
               'calibrations': ece_results}, f, indent=2)
rows_ece = [{'method': m, 'ece': v['ece'], 'ci_lower': v['ci_lower'], 'ci_upper': v['ci_upper']}
            for m, v in ece_results.items()]
pd.DataFrame(rows_ece).to_csv(os.path.join(RESULTS_DIR, 'ece_cis.csv'), index=False)
print(f"\n  Saved -> ece_cis.json / .csv")


# ══════════════════════════════════════════════════════════════════════════════
# EXTRA: SCAFFOLD Internal Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXTRA: SCAFFOLD Internal Bootstrap CI (NHANES test set)")
print("=" * 70)

scaffold_int_ci = {}
rng3 = np.random.default_rng(RANDOM_SEED)

for model in ['SCAFFOLD_SGD', 'FedAvg', 'FedProx', 'FedNova']:
    if model not in preds_int:
        continue
    pred_i = preds_int[model]
    n_int = len(y_int)
    point_auc = float(roc_auc_score(y_int, pred_i))
    boot_auc = np.empty(1000)  # N=1000 for internal (smaller set, faster)
    for i in range(1000):
        idx = rng3.integers(0, n_int, size=n_int)
        boot_auc[i] = roc_auc_score(y_int[idx], pred_i[idx])
    lo_a = float(np.percentile(boot_auc, 2.5))
    hi_a = float(np.percentile(boot_auc, 97.5))
    scaffold_int_ci[model] = {'auc': round(point_auc, 4), 'ci_lower': round(lo_a, 4), 'ci_upper': round(hi_a, 4), 'n': n_int}
    print(f"  {model:<20}: AUC={point_auc:.4f} [{lo_a:.4f}–{hi_a:.4f}]  n={n_int:,}")

with open(os.path.join(RESULTS_DIR, 'internal_bootstrap_cis.json'), 'w') as f:
    json.dump(scaffold_int_ci, f, indent=2)
print(f"\n  Saved -> internal_bootstrap_cis.json")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT SUMMARY")
print("=" * 70)
print("\n  Brier Score CIs (BRFSS external, N=2000 bootstrap):")
for m, v in brier_results.items():
    print(f"    {m:<20}: {v['brier']:.4f} [{v['ci_lower']:.4f}–{v['ci_upper']:.4f}]")

print("\n  NNS (Elderly >=60, Youden threshold):")
for model in ['FedAvg', 'XGBoost']:
    if model in nns_json['youden']:
        v = nns_json['youden'][model]
        print(f"    {model:<20}: NNS={v['nns']:.1f}  sens={v['sensitivity']:.3f}  spec={v['specificity']:.3f}  thresh={v['threshold']:.3f}")

print("\n  ECE CIs (FedAvg, BRFSS):")
for m, v in ece_results.items():
    print(f"    {m:<20}: ECE={v['ece']:.4f} [{v['ci_lower']:.4f}–{v['ci_upper']:.4f}]")

print("\n  Internal AUC CIs (NHANES test):")
for m, v in scaffold_int_ci.items():
    print(f"    {m:<20}: AUC={v['auc']:.4f} [{v['ci_lower']:.4f}–{v['ci_upper']:.4f}]")

print("\n  ALL EXPERIMENTS COMPLETE")
print("=" * 70)
