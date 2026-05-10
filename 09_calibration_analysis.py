"""
SCRIPT 09 — POST-HOC CALIBRATION ANALYSIS
==========================================
Evaluates and corrects calibration of federated model predictions on BRFSS.

METHODS:
  A. Platt scaling       — sigmoid fit: log-odds = a*logit(p) + b
  B. Isotonic regression — non-parametric monotone transformation
  C. Temperature scaling — p_cal = sigmoid(logit(p) / T), minimise NLL
  D. Subgroup calibration — elderly (>=60) vs young (18-39) before/after Platt

CALIBRATION METRIC:
  Expected Calibration Error (ECE) with 10 equal-width bins.
  Wilson 95% CI per bin (proportion CI on binned accuracy).

INPUTS:
  results/pred_fedprox_external.npy   (primary: best FL model)
  results/pred_fedavg_external.npy
  results/y_true_brfss.npy
  data/centralised_full.csv           (for age subgroup labels on internal set)

OUTPUTS:
  results/calibration_results.json
  plots/calibration_reliability_diagram.png   (4-panel)
  plots/calibration_subgroup.png              (elderly vs young ECE)

RUNTIME: ~5 minutes (no model training — post-hoc only)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    RESULTS_DIR, PLOTS_DIR, CENTRALISED_PATH,
    FEATURE_COLS, TARGET_COL, SEED,
)

print("=" * 65)
print("  09_calibration_analysis.py")
print("  Post-hoc Calibration: Platt / Isotonic / Temperature")
print("=" * 65)


# ── Load predictions ──────────────────────────────────────────────────────────
def load_npy(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    return np.load(path)

y_brfss     = load_npy('y_true_brfss.npy')
p_fedprox   = load_npy('pred_fedprox_external.npy')
p_fedavg    = load_npy('pred_fedavg_external.npy')

if y_brfss is None or p_fedprox is None:
    print("\n  ERROR: required prediction files not found.")
    print("  Run 07_external_validation.py first to generate:")
    print("    results/y_true_brfss.npy")
    print("    results/pred_fedprox_external.npy")
    raise SystemExit(1)

print(f"\n  BRFSS n={len(y_brfss):,}  "
      f"prevalence={y_brfss.mean()*100:.2f}%")


# ── ECE function ──────────────────────────────────────────────────────────────
def ece_wilson(y_true, probs, n_bins=10):
    """
    Expected Calibration Error with 10 equal-width bins.
    Returns dict: ece, bin_data (list of dicts per bin).
    """
    bins     = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece_sum  = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if lo == bins[-2]:  # include right edge for last bin
            mask = (probs >= lo) & (probs <= hi)
        n = mask.sum()
        if n == 0:
            continue
        acc  = y_true[mask].mean()
        conf = probs[mask].mean()
        # Wilson CI for accuracy
        z    = 1.96
        denom = 1 + z**2 / n
        p_hat = (acc + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2)) / denom
        bin_data.append({
            'lo': lo, 'hi': hi, 'n': int(n),
            'acc': float(acc), 'conf': float(conf),
            'ci_lo': float(max(0, p_hat - margin)),
            'ci_hi': float(min(1, p_hat + margin)),
        })
        ece_sum += (n / len(y_true)) * abs(acc - conf)

    return {'ece': float(ece_sum), 'bins': bin_data}


# ── Temperature scaling ────────────────────────────────────────────────────────
def temperature_scale(p_train, y_train, p_test):
    """Minimise NLL over temperature T; returns calibrated probs on test."""
    logits_train = logit(np.clip(p_train, 1e-7, 1 - 1e-7))

    def nll(T):
        T = max(T, 1e-3)
        p_cal = expit(logits_train / T)
        return -np.mean(y_train * np.log(p_cal + 1e-12) +
                        (1 - y_train) * np.log(1 - p_cal + 1e-12))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    T_opt = float(res.x)
    logits_test = logit(np.clip(p_test, 1e-7, 1 - 1e-7))
    return expit(logits_test / T_opt), T_opt


# ── Split BRFSS 20/80 for calibration ─────────────────────────────────────────
rng   = np.random.default_rng(SEED)
n     = len(y_brfss)
idx   = rng.permutation(n)
n_cal = int(0.20 * n)
cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

y_cal,  p_cal_raw  = y_brfss[cal_idx],  p_fedprox[cal_idx]
y_test, p_test_raw = y_brfss[test_idx], p_fedprox[test_idx]

print(f"\n  Calibration split: {n_cal:,} cal / {len(test_idx):,} test")
print(f"  Uncalibrated AUC (FedProx, test): "
      f"{roc_auc_score(y_test, p_test_raw):.4f}")


# ── A. Platt scaling ──────────────────────────────────────────────────────────
print("\n  [A] Platt scaling...")
platt = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
platt.fit(p_cal_raw.reshape(-1, 1), y_cal)
p_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]
ece_raw   = ece_wilson(y_test, p_test_raw)
ece_platt = ece_wilson(y_test, p_platt)
print(f"    ECE before: {ece_raw['ece']:.4f}  |  ECE after Platt: {ece_platt['ece']:.4f}")
print(f"    AUC after Platt: {roc_auc_score(y_test, p_platt):.4f}")


# ── B. Isotonic regression ────────────────────────────────────────────────────
print("\n  [B] Isotonic regression...")
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_cal_raw, y_cal)
p_iso  = iso.predict(p_test_raw)
ece_iso = ece_wilson(y_test, p_iso)
print(f"    ECE after Isotonic: {ece_iso['ece']:.4f}")
print(f"    AUC after Isotonic: {roc_auc_score(y_test, p_iso):.4f}")


# ── C. Temperature scaling ────────────────────────────────────────────────────
print("\n  [C] Temperature scaling...")
p_temp, T_opt = temperature_scale(p_cal_raw, y_cal, p_test_raw)
ece_temp = ece_wilson(y_test, p_temp)
print(f"    Optimal T: {T_opt:.4f}")
print(f"    ECE after Temperature: {ece_temp['ece']:.4f}")
print(f"    AUC after Temperature: {roc_auc_score(y_test, p_temp):.4f}")


# ── D. Subgroup calibration (elderly vs young) ────────────────────────────────
print("\n  [D] Subgroup calibration...")
# Need age labels for BRFSS — load from external validation results if available
age_path = os.path.join(RESULTS_DIR, 'external_validation.json')
subgroup_results = {}

if os.path.exists(age_path):
    with open(age_path) as f:
        ext_val = json.load(f)
    print(f"    External validation results loaded for subgroup context.")
else:
    print(f"    external_validation.json not found — subgroup analysis skipped.")

# Internal set subgroup calibration (using NHANES centralised data for age)
try:
    df_c   = pd.read_csv(CENTRALISED_PATH)
    ages_c = df_c['RIDAGEYR'].values
    p_int  = load_npy('pred_fedprox_internal.npy')
    y_int  = load_npy('y_true_internal.npy')

    if p_int is not None and y_int is not None and len(ages_c) == len(y_int):
        for grp_name, (lo, hi) in [('Young (18-39)', (18, 39)),
                                     ('Elderly (>=60)', (60, 130))]:
            mask = (ages_c >= lo) & (ages_c <= hi)
            if mask.sum() < 30:
                continue
            ece_grp     = ece_wilson(y_int[mask], p_int[mask])
            platt_grp   = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
            # Cal on complementary subgroup for demonstration
            platt_grp.fit(p_int[~mask].reshape(-1, 1), y_int[~mask])
            p_cal_grp   = platt_grp.predict_proba(p_int[mask].reshape(-1, 1))[:, 1]
            ece_grp_cal = ece_wilson(y_int[mask], p_cal_grp)
            subgroup_results[grp_name] = {
                'n': int(mask.sum()),
                'ece_raw': ece_grp['ece'],
                'ece_platt': ece_grp_cal['ece'],
            }
            print(f"    {grp_name}: ECE raw={ece_grp['ece']:.4f} | after Platt={ece_grp_cal['ece']:.4f}")
except Exception as e:
    print(f"    Subgroup calibration skipped: {e}")


# ── Save results ──────────────────────────────────────────────────────────────
out = {
    'n_cal'      : n_cal,
    'n_test'     : int(len(test_idx)),
    'uncalibrated': {
        'ece': ece_raw['ece'],
        'auc': float(roc_auc_score(y_test, p_test_raw)),
    },
    'platt': {
        'ece': ece_platt['ece'],
        'auc': float(roc_auc_score(y_test, p_platt)),
        'coef': float(platt.coef_[0][0]),
        'intercept': float(platt.intercept_[0]),
    },
    'isotonic': {
        'ece': ece_iso['ece'],
        'auc': float(roc_auc_score(y_test, p_iso)),
    },
    'temperature': {
        'ece': ece_temp['ece'],
        'auc': float(roc_auc_score(y_test, p_temp)),
        'T_optimal': T_opt,
    },
    'subgroup_internal': subgroup_results,
}
with open(os.path.join(RESULTS_DIR, 'calibration_results.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> results/calibration_results.json")


# ── 4-panel reliability diagram ───────────────────────────────────────────────
def reliability_ax(ax, y_true, probs, title, color='#2563EB'):
    data  = ece_wilson(y_true, probs)
    confs = [b['conf'] for b in data['bins']]
    accs  = [b['acc']  for b in data['bins']]
    lo_ci = [b['acc'] - b['ci_lo'] for b in data['bins']]
    hi_ci = [b['ci_hi'] - b['acc'] for b in data['bins']]
    ax.errorbar(confs, accs, yerr=[lo_ci, hi_ci],
                fmt='o', color=color, capsize=4, markersize=6, lw=1.5, label='Observed')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    ax.set_title(f"{title}\nECE={data['ece']:.4f}", fontsize=10, fontweight='bold')
    ax.set_xlabel('Mean predicted probability', fontsize=9)
    ax.set_ylabel('Fraction of positives', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    reliability_ax(axes[0, 0], y_test, p_test_raw, 'Uncalibrated (FedProx)', '#94A3B8')
    reliability_ax(axes[0, 1], y_test, p_platt,    'Platt Scaling',          '#2563EB')
    reliability_ax(axes[1, 0], y_test, p_iso,       'Isotonic Regression',    '#7C3AED')
    reliability_ax(axes[1, 1], y_test, p_temp,      f'Temperature (T={T_opt:.2f})', '#DC2626')
    plt.suptitle('Calibration Reliability Diagrams — FedProx on BRFSS',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'calibration_reliability_diagram.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> plots/calibration_reliability_diagram.png")
except Exception as e:
    print(f"  Reliability plot skipped: {e}")


# ── Subgroup ECE bar chart ─────────────────────────────────────────────────────
if subgroup_results:
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups  = list(subgroup_results.keys())
        raw     = [subgroup_results[g]['ece_raw']   for g in groups]
        cal     = [subgroup_results[g]['ece_platt'] for g in groups]
        x       = np.arange(len(groups))
        ax.bar(x - 0.2, raw, 0.35, label='Uncalibrated', color='#94A3B8')
        ax.bar(x + 0.2, cal, 0.35, label='Platt scaled',  color='#2563EB')
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
        ax.set_ylabel('ECE (lower = better)', fontsize=11)
        ax.set_title('Subgroup Calibration — Internal NHANES Set', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'calibration_subgroup.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved -> plots/calibration_subgroup.png")
    except Exception as e:
        print(f"  Subgroup plot skipped: {e}")

print("\n" + "=" * 65)
print("  CALIBRATION SUMMARY")
print("=" * 65)
print(f"  {'Method':<22}  {'ECE':>6}  {'AUC':>6}")
print(f"  {'-'*38}")
print(f"  {'Uncalibrated':<22}  {ece_raw['ece']:>6.4f}  {roc_auc_score(y_test, p_test_raw):>6.4f}")
print(f"  {'Platt scaling':<22}  {ece_platt['ece']:>6.4f}  {roc_auc_score(y_test, p_platt):>6.4f}")
print(f"  {'Isotonic regression':<22}  {ece_iso['ece']:>6.4f}  {roc_auc_score(y_test, p_iso):>6.4f}")
print(f"  {'Temperature (T={:.2f})'.format(T_opt):<22}  {ece_temp['ece']:>6.4f}  {roc_auc_score(y_test, p_temp):>6.4f}")
print("=" * 65)
