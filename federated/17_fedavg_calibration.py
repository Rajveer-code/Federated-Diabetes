"""
17_fedavg_calibration.py
=========================
Post-hoc calibration analysis for FedAvg on BRFSS external set.
Mirrors 09_calibration_analysis.py (FedProx) with identical split/methods.

Saves: results/fedavg_calibration_results.json
"""

import os, sys, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

from scipy.special import logit, expit
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR, SEED

print("=" * 65)
print("  17 — FedAvg Post-Hoc Calibration (BRFSS)")
print("=" * 65)


def load_npy(fname):
    return np.load(os.path.join(RESULTS_DIR, fname))


def ece_wilson(y_true, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece_sum = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if lo == bins[-2]:
            mask = (probs >= lo) & (probs <= hi)
        n = mask.sum()
        if n == 0:
            continue
        acc  = y_true[mask].mean()
        conf = probs[mask].mean()
        z    = 1.96
        denom = 1 + z**2 / n
        p_hat = (acc + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2)) / denom
        bin_data.append({
            'lo': float(lo), 'hi': float(hi), 'n': int(n),
            'acc': float(acc), 'conf': float(conf),
            'ci_lo': float(max(0, p_hat - margin)),
            'ci_hi': float(min(1, p_hat + margin)),
        })
        ece_sum += (n / len(y_true)) * abs(acc - conf)
    return {'ece': float(ece_sum), 'bins': bin_data}


def temperature_scale(p_train, y_train, p_test):
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


# Load predictions
y_brfss  = load_npy('y_true_brfss.npy')
p_fedavg = load_npy('pred_fedavg_external.npy')
print(f"\n  BRFSS n={len(y_brfss):,}  prevalence={y_brfss.mean()*100:.2f}%")

# 20/80 cal/test split — IDENTICAL to 09_calibration_analysis.py
rng   = np.random.default_rng(SEED)
n     = len(y_brfss)
idx   = rng.permutation(n)
n_cal = int(0.20 * n)
cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

y_cal,  p_cal_raw  = y_brfss[cal_idx],  p_fedavg[cal_idx]
y_test, p_test_raw = y_brfss[test_idx], p_fedavg[test_idx]

print(f"  Split: {n_cal:,} cal / {len(test_idx):,} test")
print(f"  Uncalibrated AUC (FedAvg, test): {roc_auc_score(y_test, p_test_raw):.4f}")

# A. Platt scaling
print("\n  [A] Platt scaling...")
platt = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
platt.fit(p_cal_raw.reshape(-1, 1), y_cal)
p_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]
ece_raw   = ece_wilson(y_test, p_test_raw)
ece_platt = ece_wilson(y_test, p_platt)
print(f"    ECE before: {ece_raw['ece']:.4f}  |  ECE after Platt: {ece_platt['ece']:.4f}")
print(f"    AUC after Platt: {roc_auc_score(y_test, p_platt):.4f}")

# B. Isotonic regression
print("\n  [B] Isotonic regression...")
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_cal_raw, y_cal)
p_iso = iso.predict(p_test_raw)
ece_iso = ece_wilson(y_test, p_iso)
print(f"    ECE after Isotonic: {ece_iso['ece']:.4f}")
print(f"    AUC after Isotonic: {roc_auc_score(y_test, p_iso):.4f}")

# C. Temperature scaling
print("\n  [C] Temperature scaling...")
p_temp, T_opt = temperature_scale(p_cal_raw, y_cal, p_test_raw)
ece_temp = ece_wilson(y_test, p_temp)
print(f"    Optimal T: {T_opt:.4f}")
print(f"    ECE after Temperature: {ece_temp['ece']:.4f}")
print(f"    AUC after Temperature: {roc_auc_score(y_test, p_temp):.4f}")

# Save
out = {
    'model': 'FedAvg',
    'n_cal': n_cal,
    'n_test': int(len(test_idx)),
    'uncalibrated': {
        'ece': ece_raw['ece'],
        'auc': float(roc_auc_score(y_test, p_test_raw)),
        'bins': ece_raw['bins'],
    },
    'platt': {
        'ece': ece_platt['ece'],
        'auc': float(roc_auc_score(y_test, p_platt)),
        'coef': float(platt.coef_[0][0]),
        'intercept': float(platt.intercept_[0]),
        'bins': ece_platt['bins'],
    },
    'isotonic': {
        'ece': ece_iso['ece'],
        'auc': float(roc_auc_score(y_test, p_iso)),
        'bins': ece_iso['bins'],
    },
    'temperature': {
        'ece': ece_temp['ece'],
        'auc': float(roc_auc_score(y_test, p_temp)),
        'T_optimal': T_opt,
        'bins': ece_temp['bins'],
    },
}

out_path = os.path.join(RESULTS_DIR, 'fedavg_calibration_results.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*65}")
print(f"  FedAvg Calibration Summary (BRFSS test n={len(test_idx):,})")
print(f"  {'Method':<24}  {'ECE':>8}  {'AUC':>8}")
print(f"  {'-'*42}")
print(f"  {'Uncalibrated':<24}  {ece_raw['ece']:>8.4f}  {roc_auc_score(y_test, p_test_raw):>8.4f}")
print(f"  {'Platt scaling':<24}  {ece_platt['ece']:>8.4f}  {roc_auc_score(y_test, p_platt):>8.4f}")
print(f"  {'Isotonic regression':<24}  {ece_iso['ece']:>8.4f}  {roc_auc_score(y_test, p_iso):>8.4f}")
print(f"  {'Temperature (T={:.2f})'.format(T_opt):<24}  {ece_temp['ece']:>8.4f}  {roc_auc_score(y_test, p_temp):>8.4f}")
print(f"  Saved: {out_path}")
print(f"{'='*65}")
