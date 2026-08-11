"""
40_calibration_norace.py — recompute calibration on the corrected pipeline
==========================================================================
The manuscript's ECE figures (0.276 -> 0.001) came from
pred_fedavg_external.npy, produced before the BatchNorm-buffer fix and before
race was dropped as a model input. They therefore describe a model that no
longer exists. This recomputes them from the cached race-free predictions.

Method matches 17_fedavg_calibration.py exactly: a 20/80 calibration/test split
of the BRFSS rows, calibrators fitted on the 20% and ECE evaluated on the held-out
80%, with [lo, hi) bins and the final bin closed. Run per seed so the paper can
report mean +/- sd like every other number, rather than a single draw.

Output: results/calibration_norace.json

Usage:  python 40_calibration_norace.py [--model FedAvg]
"""
import os, sys, json, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')
from scipy.optimize import minimize_scalar
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR, SEED

PRED_DIR = os.path.join(RESULTS_DIR, 'preds_multiseed')


def ece(y, p, n_bins=10):
    """[lo, hi) bins, final bin closed — matches ece_wilson()."""
    b = np.clip((np.asarray(p) * n_bins).astype(np.intp), 0, n_bins - 1)
    s_acc  = np.bincount(b, weights=y, minlength=n_bins)
    s_conf = np.bincount(b, weights=p, minlength=n_bins)
    return float(np.abs(s_acc - s_conf).sum() / len(y))


def temperature(p_tr, y_tr, p_te):
    lg = logit(np.clip(p_tr, 1e-7, 1 - 1e-7))
    f = lambda T: -np.mean(y_tr * np.log(expit(lg / max(T, 1e-3)) + 1e-12)
                           + (1 - y_tr) * np.log(1 - expit(lg / max(T, 1e-3)) + 1e-12))
    T = float(minimize_scalar(f, bounds=(0.1, 10.0), method='bounded').x)
    return expit(logit(np.clip(p_te, 1e-7, 1 - 1e-7)) / T), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='FedAvg')
    args = ap.parse_args()

    y = np.load(os.path.join(PRED_DIR, '_y_brfss.npy'))
    seeds = sorted(int(f.split('_seed')[1].split('_')[0])
                   for f in os.listdir(PRED_DIR)
                   if f.startswith(f'{args.model}_seed') and f.endswith('_external.npy'))
    if not seeds:
        sys.exit(f"No cached predictions for {args.model}")

    runs = []
    for s in seeds:
        p = np.load(os.path.join(PRED_DIR, f'{args.model}_seed{s}_external.npy'))
        idx = np.random.default_rng(SEED).permutation(len(y))   # same split every seed
        n_cal = int(0.20 * len(y))
        ci_, ti_ = idx[:n_cal], idx[n_cal:]
        y_c, p_c, y_t, p_t = y[ci_], p[ci_], y[ti_], p[ti_]

        pl = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        pl.fit(p_c.reshape(-1, 1), y_c)
        p_platt = pl.predict_proba(p_t.reshape(-1, 1))[:, 1]
        p_iso = IsotonicRegression(out_of_bounds='clip').fit(p_c, y_c).predict(p_t)
        p_temp, T = temperature(p_c, y_c, p_t)

        runs.append({
            'seed': s, 'T_optimal': round(T, 4),
            'auc': round(float(roc_auc_score(y_t, p_t)), 5),
            'ece_uncalibrated': round(ece(y_t, p_t), 5),
            'ece_platt':        round(ece(y_t, p_platt), 5),
            'ece_isotonic':     round(ece(y_t, p_iso), 5),
            'ece_temperature':  round(ece(y_t, p_temp), 5),
            'brier_uncalibrated': round(float(brier_score_loss(y_t, p_t)), 5),
            'brier_isotonic':     round(float(brier_score_loss(y_t, p_iso)), 5),
        })
        print(f"  seed {s}: ECE raw={runs[-1]['ece_uncalibrated']:.4f} "
              f"platt={runs[-1]['ece_platt']:.4f} iso={runs[-1]['ece_isotonic']:.4f} "
              f"temp={runs[-1]['ece_temperature']:.4f}")

    summary = {k: {'mean': round(float(np.mean([r[k] for r in runs])), 5),
                   'sd':   round(float(np.std([r[k] for r in runs], ddof=1)), 5)}
               for k in runs[0] if k != 'seed'}

    print(f"\n  {args.model}, {len(runs)} seeds, test n={len(y) - int(0.20 * len(y)):,}")
    for k in ['ece_uncalibrated', 'ece_platt', 'ece_isotonic', 'ece_temperature']:
        print(f"    {k:<20} {summary[k]['mean']:.4f} +/- {summary[k]['sd']:.4f}")

    out = os.path.join(RESULTS_DIR, 'calibration_norace.json')
    with open(out, 'w') as f:
        json.dump({'model': args.model, 'seeds': seeds, 'n_brfss': int(len(y)),
                   'n_cal': int(0.20 * len(y)), 'n_test': int(len(y) - int(0.20 * len(y))),
                   'note': 'race-free 7-feature models with aggregated BN buffers; '
                           'supersedes fedavg_calibration_results.json',
                   'runs': runs, 'summary': summary}, f, indent=2)
    print(f"  Saved -> {out}")


if __name__ == '__main__':
    main()
