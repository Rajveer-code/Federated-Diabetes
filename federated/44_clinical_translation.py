"""
44_clinical_translation.py — what does a 0.007 AUC difference actually buy?
==========================================================================
An AUC difference of 0.0066 is statistically overwhelming on 1.28M rows and
means nothing to a clinician. A screening programme does not deploy an AUC; it
deploys a threshold and a fixed capacity. So the question that matters is: at a
realistic screening capacity, how many additional true cases does the federated
model find in each stratum, and does the majority stratum lose any?

One global threshold is applied to every stratum, which is how such a programme
would actually run -- per-group thresholds would be a different (and legally
fraught) intervention. The threshold is set so that a fixed fraction of the whole
external cohort is flagged for follow-up.

Reported per stratum:
  sensitivity (TPR)  -- of those who have diabetes, the fraction flagged
  delta cases/100k   -- additional true cases detected per 100,000 screened,
                        relative to the composition-matched centralised control

Output: results/clinical_translation.json

Usage:  python 44_clinical_translation.py [--rates 0.10 0.20 0.30]
"""
import os, sys, json, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

PRED_DIR = os.path.join(RESULTS_DIR, 'preds_multiseed')
STRATA = {'White': [1.0], 'Black': [2.0], 'Hispanic': [8.0],
          'Asian': [4.0], 'Other': [3.0, 5.0, 6.0, 7.0]}
REF = 'Centralised-matched'
MODELS = ['FedAvg', 'FedProx', 'SCAFFOLD', 'Centralised-full', REF]


def load(n):
    return np.load(os.path.join(PRED_DIR, f'{n}.npy'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', type=float, nargs='+', default=[0.10, 0.20, 0.30])
    args = ap.parse_args()

    y = load('_y_brfss')
    race = load('_race_brfss')
    masks = {k: np.isin(race, v) for k, v in STRATA.items()}
    seeds = sorted(int(f.split('_seed')[1].split('_')[0])
                   for f in os.listdir(PRED_DIR)
                   if f.startswith('FedAvg_seed') and f.endswith('_external.npy'))

    out = {'n_brfss': int(len(y)), 'seeds': seeds,
           'note': 'One global threshold per flag rate, applied to every stratum.',
           'strata_n': {k: int(m.sum()) for k, m in masks.items()},
           'strata_prevalence': {k: round(float(y[m].mean()), 5) for k, m in masks.items()},
           'by_rate': {}}

    for rate in args.rates:
        per_model = {}
        for model in MODELS:
            acc = {k: [] for k in STRATA}
            acc['_overall'] = []
            for s in seeds:
                p = load(f'{model}_seed{s}_external')
                thr = np.quantile(p, 1 - rate)      # flag the top `rate` fraction
                flag = p >= thr
                acc['_overall'].append(float(flag[y == 1].mean()))
                for k, m in masks.items():
                    yk, fk = y[m], flag[m]
                    acc[k].append(float(fk[yk == 1].mean()))   # sensitivity
            per_model[model] = {k: {'tpr_mean': round(float(np.mean(v)), 5),
                                    'tpr_sd': round(float(np.std(v, ddof=1)), 5)}
                                for k, v in acc.items()}
        # Translate deltas into cases per 100,000 screened in that stratum.
        for model in MODELS:
            if model == REF:
                continue
            for k in STRATA:
                d = per_model[model][k]['tpr_mean'] - per_model[REF][k]['tpr_mean']
                prev = float(y[masks[k]].mean())
                per_model[model][k]['delta_tpr'] = round(d, 5)
                per_model[model][k]['delta_cases_per_100k'] = round(d * prev * 1e5, 1)
        out['by_rate'][f'{rate:.2f}'] = per_model

        print(f"\n{'='*78}\n  Flag rate {rate:.0%} of the cohort — sensitivity by stratum\n{'='*78}")
        print(f"  {'stratum':<10}{'prev':>7}{'ref TPR':>10}{'FedAvg':>10}{'d TPR':>9}{'d cases/100k':>14}")
        for k in STRATA:
            r = per_model[REF][k]['tpr_mean']; f_ = per_model['FedAvg'][k]
            print(f"  {k:<10}{y[masks[k]].mean():>7.3f}{r:>10.4f}{f_['tpr_mean']:>10.4f}"
                  f"{f_['delta_tpr']:>+9.4f}{f_['delta_cases_per_100k']:>+14.1f}")
        wb_ref = per_model[REF]['White']['tpr_mean'] - per_model[REF]['Black']['tpr_mean']
        wb_fed = per_model['FedAvg']['White']['tpr_mean'] - per_model['FedAvg']['Black']['tpr_mean']
        print(f"  White-Black sensitivity gap: ref {wb_ref:+.4f} -> FedAvg {wb_fed:+.4f} "
              f"({wb_fed - wb_ref:+.4f})")
        out['by_rate'][f'{rate:.2f}']['_white_black_tpr_gap'] = {
            'reference': round(wb_ref, 5), 'FedAvg': round(wb_fed, 5),
            'delta': round(wb_fed - wb_ref, 5)}

    path = os.path.join(RESULTS_DIR, 'clinical_translation.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == '__main__':
    main()
