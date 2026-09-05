"""
57_budget_bn_readout.py -- score the two control arms on the paper's own metrics
================================================================================
56_budget_and_bn_controls.py trained three extra arms and cached their prediction
vectors. This scores them with the same quantities the manuscript reports, so the
robustness claim is made on the identical metric definitions, not on a proxy:

  * per-stratum AUC and the White-Black AUC gap
  * xAUC_{g->White}: P(case from g outranks a White non-case)
  * selection rate and sensitivity at 20% capacity under one global threshold
  * the White-Black sensitivity-gap change

Two comparisons are reported:

  BUDGET   FedAvg  vs  Centralised-matched-250ep
           Does the divergence survive when the centralised control is given the
           same number of example passes each federated client gets
           (FL_NUM_ROUNDS x NN_LOCAL_EPOCHS), rather than FL_NUM_ROUNDS epochs?

  BATCHNORM  FedAvg-noBN  vs  Centralised-matched-noBN
           Does the divergence survive with nn.BatchNorm1d removed, so no
           client statistics are pooled at all?

Headline reference (recomputed here so all three sit on one code path):
  HEADLINE   FedAvg  vs  Centralised-matched

Input : results/preds_multiseed/*.npy   (needs 56_ to have finished)
Output: results/budget_bn_readout.json
Usage:  python 57_budget_bn_readout.py
"""
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

_dl = __import__('35_delong_deltas')
load = _dl.load
RACE_STRATA, RACE_REPORTED, Z95 = _dl.RACE_STRATA, _dl.RACE_REPORTED, _dl.Z95
PRED_DIR = _dl.PRED_DIR

CAPACITY = 0.20
COMPARISONS = [
    ('headline',  'FedAvg',      'Centralised-matched'),
    ('budget',    'FedAvg',      'Centralised-matched-250ep'),
    ('batchnorm', 'FedAvg-noBN', 'Centralised-matched-noBN'),
]


def seeds_for(models):
    """Seeds where every model in `models` has a cached external prediction."""
    out = []
    for s in range(42, 52):
        if all(os.path.exists(os.path.join(PRED_DIR, f'{m}_seed{s}_external.npy'))
               for m in models):
            out.append(s)
    return out


def auc(y, p):
    """Mann-Whitney U / |pos||neg|. Ties contribute 0.5, as DeLong's does."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks within tie groups so tied pairs score 0.5
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def xauc(p, pos_mask, neg_mask):
    """xAUC_{g->h}: ordinary AUC on {positives of g} vs {negatives of h}."""
    y = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
    s = np.concatenate([p[pos_mask], p[neg_mask]])
    return auc(y, s)


def threshold_from(scores, capacity):
    ps = np.sort(scores)
    return ps[int(np.clip(len(ps) * (1.0 - capacity), 0, len(ps) - 1))]


def summarise(deltas):
    d = np.asarray(deltas, dtype=float)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return dict(mean=round(m, 6), se=round(se, 6),
                ci_low=round(m - Z95 * se, 6), ci_high=round(m + Z95 * se, 6),
                n_seeds_same_sign=int(max((d > 0).sum(), (d < 0).sum())),
                n_seeds=len(d))


def main():
    y = load('_y_brfss')
    race = load('_race_brfss')
    n = len(y)

    strat = {g: np.isin(race, codes) for g, codes in RACE_STRATA.items()}
    pos = {g: m & (y == 1) for g, m in strat.items()}
    neg = {g: m & (y == 0) for g, m in strat.items()}
    print(f"BRFSS n={n:,}   " + "  ".join(f"{g}={strat[g].sum():,}" for g in RACE_REPORTED))

    out = {'capacity': CAPACITY, 'comparisons': {}}

    for tag, treat, ref in COMPARISONS:
        sds = seeds_for([treat, ref])
        if not sds:
            print(f"\n[{tag}] SKIPPED — no complete seeds for {treat} vs {ref}")
            continue
        print(f"\n{'='*72}\n  {tag.upper():<10} {treat}  vs  {ref}   ({len(sds)} seeds)\n{'='*72}")

        acc = {'auc': {g: [] for g in RACE_REPORTED},
               'xauc_w': {g: [] for g in RACE_REPORTED},
               'sel': {g: [] for g in RACE_REPORTED},
               'sens': {g: [] for g in RACE_REPORTED},
               'auc_gap_wb': [], 'sens_gap_wb': []}
        absolute = {m: {'sens_b': [], 'sens_w': [], 'sel_b': [], 'sel_w': []}
                    for m in (treat, ref)}

        for s in sds:
            per = {}
            for m in (treat, ref):
                p = load(f'{m}_seed{s}_external')
                thr = threshold_from(p, CAPACITY)
                flag = p >= thr
                r = {'auc': {}, 'xauc_w': {}, 'sel': {}, 'sens': {}}
                for g in RACE_REPORTED:
                    r['auc'][g]    = auc(y[strat[g]], p[strat[g]])
                    r['xauc_w'][g] = xauc(p, pos[g], neg['White'])
                    r['sel'][g]    = float(flag[strat[g]].mean())
                    r['sens'][g]   = float(flag[pos[g]].mean())
                r['auc_gap_wb']  = r['auc']['White']  - r['auc']['Black']
                r['sens_gap_wb'] = r['sens']['White'] - r['sens']['Black']
                per[m] = r
                absolute[m]['sens_b'].append(r['sens']['Black'])
                absolute[m]['sens_w'].append(r['sens']['White'])
                absolute[m]['sel_b'].append(r['sel']['Black'])
                absolute[m]['sel_w'].append(r['sel']['White'])
            for key in ('auc', 'xauc_w', 'sel', 'sens'):
                for g in RACE_REPORTED:
                    acc[key][g].append(per[treat][key][g] - per[ref][key][g])
            acc['auc_gap_wb'].append(per[treat]['auc_gap_wb'] - per[ref]['auc_gap_wb'])
            acc['sens_gap_wb'].append(per[treat]['sens_gap_wb'] - per[ref]['sens_gap_wb'])
            print(f"  seed {s} done")

        res = {
            'treatment': treat, 'reference': ref, 'n_seeds': len(sds), 'seeds': sds,
            'delta_auc':      {g: summarise(acc['auc'][g])    for g in RACE_REPORTED},
            'delta_xauc_to_white': {g: summarise(acc['xauc_w'][g]) for g in RACE_REPORTED},
            'delta_selection': {g: summarise(acc['sel'][g])    for g in RACE_REPORTED},
            'delta_sensitivity': {g: summarise(acc['sens'][g]) for g in RACE_REPORTED},
            'delta_wb_auc_gap':  summarise(acc['auc_gap_wb']),
            'delta_wb_sens_gap': summarise(acc['sens_gap_wb']),
            'absolute': {m: {k: round(float(np.mean(v)), 4) for k, v in d.items()}
                         for m, d in absolute.items()},
        }
        out['comparisons'][tag] = res

        print(f"  {'stratum':<10}{'dAUC':>10}{'dxAUC->W':>11}{'dSel':>10}{'dSens':>10}")
        for g in RACE_REPORTED:
            print(f"  {g:<10}{res['delta_auc'][g]['mean']:>+10.4f}"
                  f"{res['delta_xauc_to_white'][g]['mean']:>+11.4f}"
                  f"{res['delta_selection'][g]['mean']:>+10.4f}"
                  f"{res['delta_sensitivity'][g]['mean']:>+10.4f}")
        a, b = res['delta_wb_auc_gap'], res['delta_wb_sens_gap']
        print(f"  W-B AUC  gap change : {a['mean']:+.4f} [{a['ci_low']:+.4f},{a['ci_high']:+.4f}]"
              f"  {a['n_seeds_same_sign']}/{a['n_seeds']} seeds same sign")
        print(f"  W-B SENS gap change : {b['mean']:+.4f} [{b['ci_low']:+.4f},{b['ci_high']:+.4f}]"
              f"  {b['n_seeds_same_sign']}/{b['n_seeds']} seeds same sign")

    path = os.path.join(RESULTS_DIR, 'budget_bn_readout.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == '__main__':
    main()
