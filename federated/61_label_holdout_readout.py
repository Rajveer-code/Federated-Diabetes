"""
61_label_holdout_readout.py — the headline comparison on two further test sets
==============================================================================
Reads any 33_multiseed_all_models.py prediction cache and reports, FedAvg (and
every strategy) minus the composition-matched control, mean over seeds with
95% across-seed t-intervals:
  per-stratum within-group AUC, xAUC_{g->White}, and selection rate /
  sensitivity under one pooled top-20% cutoff,
on (a) the BRFSS external cohort and (b) the NHANES internal hold-out (rows
outside the node caps, never seen by the FL models or the matched control,
labelled as in training).

Run once per cache: results_v2 (composite NHANES label) and results_sr (models
trained on the self-report-only label, reviewer cqFk).

Usage: python 61_label_holdout_readout.py results_v2 results_sr
Output: <dir>/label_holdout_readout.json
"""
import os, sys, json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import PROJECT_ROOT

REF = 'Centralised-matched'
STRATEGIES = ['FedAvg', 'FedProx', 'FedNova', 'FedAvg-EqualW', 'SCAFFOLD']
SEEDS = list(range(42, 52))
Q = 0.20
STRATA = {
    'external': {'White': [1.0], 'Black': [2.0], 'Hispanic': [8.0], 'Asian': [4.0],
                 'Other': [3.0, 5.0, 6.0, 7.0]},                       # BRFSS _RACE
    'holdout': {'White': [3.0], 'Black': [4.0], 'Hispanic': [1.0, 2.0], 'Asian': [6.0],
                'Other': [7.0]},                                       # NHANES RIDRETH3
}


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {'mean': float(x.mean()), 'lo': float(x.mean() - h), 'hi': float(x.mean() + h),
            'n_pos': int((x > 0).sum())}


def metrics(p, y, masks):
    thr = np.sort(p)[int(len(p) * (1 - Q))]
    flag = p >= thr
    wn = masks['White'] & (y == 0)
    out = {}
    for g, mk in masks.items():
        pos = mk & (y == 1)
        out[g] = {'auc': roc_auc_score(y[mk], p[mk]),
                  'xauc_to_W': (roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(wn.sum())],
                                              np.r_[p[pos], p[wn]]) if g != 'White' else np.nan),
                  'sel': flag[mk].mean(), 'sens': flag[pos].mean()}
    return out


def readout(pred_dir, split):
    yname, rname = ('_y_brfss', '_race_brfss') if split == 'external' else ('_y_holdout', '_race_holdout')
    y = np.load(os.path.join(pred_dir, f'{yname}.npy')).astype(int)
    race = np.load(os.path.join(pred_dir, f'{rname}.npy'))
    masks = {g: np.isin(race, v) for g, v in STRATA[split].items()}
    per = {s: {m: metrics(np.load(os.path.join(pred_dir, f'{m}_seed{s}_{split}.npy')), y, masks)
               for m in STRATEGIES + [REF]} for s in SEEDS}
    res = {'n': int(len(y)), 'strata_n': {g: int(mk.sum()) for g, mk in masks.items()},
           'strata_pos': {g: int((mk & (y == 1)).sum()) for g, mk in masks.items()}}
    for m in STRATEGIES:
        d = {}
        for g in masks:
            for k in ('auc', 'xauc_to_W', 'sel', 'sens'):
                if g == 'White' and k == 'xauc_to_W':
                    continue
                d[f'{k}_{g}'] = ci([per[s][m][g][k] - per[s][REF][g][k] for s in SEEDS])
        d['auc_gap_WB'] = ci([(per[s][m]['White']['auc'] - per[s][m]['Black']['auc'])
                              - (per[s][REF]['White']['auc'] - per[s][REF]['Black']['auc']) for s in SEEDS])
        d['sens_gap_WB'] = ci([(per[s][m]['White']['sens'] - per[s][m]['Black']['sens'])
                               - (per[s][REF]['White']['sens'] - per[s][REF]['Black']['sens']) for s in SEEDS])
        res[m] = d
    return res


def main():
    for sub in sys.argv[1:]:
        rdir = os.path.join(PROJECT_ROOT, sub)
        pred_dir = os.path.join(rdir, 'preds_multiseed')
        out = {split: readout(pred_dir, split) for split in ('external', 'holdout')}
        with open(os.path.join(rdir, 'label_holdout_readout.json'), 'w') as f:
            json.dump(out, f, indent=1)
        for split in ('external', 'holdout'):
            f = out[split]['FedAvg']
            print(f"{sub:11s} {split:8s} n={out[split]['n']:>9,}  dAUC_B {f['auc_Black']['mean']:+.4f} "
                  f"[{f['auc_Black']['lo']:+.4f},{f['auc_Black']['hi']:+.4f}]  "
                  f"dAUCgap {f['auc_gap_WB']['mean']:+.4f}  dxAUC_B->W {f['xauc_to_W_Black']['mean']:+.4f}  "
                  f"dsens_B {f['sens_Black']['mean']:+.4f} [{f['sens_Black']['lo']:+.4f},{f['sens_Black']['hi']:+.4f}]  "
                  f"dsensgap {f['sens_gap_WB']['mean']:+.4f} ({f['sens_gap_WB']['n_pos']}/10)")


if __name__ == '__main__':
    main()
