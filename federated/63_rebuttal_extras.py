"""
63_rebuttal_extras.py — readouts added after an external review of the rebuttal
===============================================================================
Reads the corrected prediction cache (results_v2) and writes one JSON with:

  cdc_weights   Survey-weighted readout with CDC's sample-size-proportional
                multi-year adjustment (w = _LLCPWT * n_year / n_total) instead of
                _LLCPWT / 3. Same weighted cutoff / sensitivity / selection / AUC.
  det_topk      Exact top-k (k = floor(0.2 n)) with a fixed tie order: score
                descending, then BRFSS row index ascending. No randomness.
  boot_all      The paired-bootstrap intervals of 59 for all five strategies,
                copied into one table.
  cutoff_band   Who sits near the 20% cutoff: among non-cases scoring in the
                17.5-22.5% capacity band, the White share and the White mean age;
                among Black positives, the share above the cutoff. Seed means.

Output: <RESULTS_DIR>/rebuttal_extras.json
"""
import os, sys, json, importlib
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR, BRFSS_PATH

r59 = importlib.import_module('59_rebuttal_threshold')
STRATEGIES, REF, SEEDS, G = r59.STRATEGIES, r59.REF, r59.SEEDS, r59.G
Q = 0.20


def main():
    y = r59.load('_y_brfss').astype(int)
    race = r59.load('_race_brfss')
    age = r59.load('_age_brfss')
    masks = {g: np.isin(race, v) for g, v in r59.RACE_STRATA.items()}
    d = pd.read_csv(BRFSS_PATH, usecols=['Diabetes_Outcome', '_LLCPWT', 'YEAR'])
    d = d.dropna(subset=['Diabetes_Outcome']).reset_index(drop=True)
    assert len(d) == len(y) and (d['Diabetes_Outcome'].values == y).all()
    scores = {(m, s): r59.load(f'{m}_seed{s}_external') for m in STRATEGIES + [REF] for s in SEEDS}
    out = {}

    # CDC sample-size-proportional multi-year weights
    n_year = d['YEAR'].map(d['YEAR'].value_counts()).values
    w = d['_LLCPWT'].values * n_year / len(d)
    out['cdc_weights'] = {'year_n': {int(k): int(v) for k, v in d['YEAR'].value_counts().sort_index().items()},
                          'alloc': r59.summarise([{m: r59.alloc(scores[(m, s)], y, masks, Q, w=w)
                                                   for m in STRATEGIES + [REF]} for s in SEEDS])}
    out['cdc_weights']['auc'] = {}
    for m in STRATEGIES:
        wa = lambda mm, s, g: r59.weighted_auc(scores[(mm, s)][masks[g]], y[masks[g]], w[masks[g]])
        out['cdc_weights']['auc'][m] = {
            'Black': r59.seed_ci([wa(m, s, 'Black') - wa(REF, s, 'Black') for s in SEEDS]),
            'gap_WB': r59.seed_ci([(wa(m, s, 'White') - wa(m, s, 'Black')) - (wa(REF, s, 'White') - wa(REF, s, 'Black'))
                                   for s in SEEDS])}

    # deterministic exact top-k
    k = int(Q * len(y))
    idx = np.arange(len(y))
    per = []
    for s in SEEDS:
        r = {}
        for m in STRATEGIES + [REF]:
            p = scores[(m, s)]
            flag = np.zeros(len(p), bool)
            flag[np.lexsort((idx, -p))[:k]] = True
            r[m] = r59.alloc(p, y, masks, Q, flag=flag)
        per.append(r)
    out['det_topk'] = r59.summarise(per)

    # bootstrap intervals for all five strategies (from 59's output)
    rt = json.load(open(os.path.join(RESULTS_DIR, 'rebuttal_threshold.json')))
    out['boot_all'] = rt['boot']['0.2']

    # cutoff band composition
    band = {}
    for m in ['FedAvg', REF]:
        wsh, wage, bpos = [], [], []
        for s in SEEDS:
            p = scores[(m, s)]
            ps = np.sort(p)
            lo, hi, thr = ps[int(len(p) * 0.775)], ps[int(len(p) * 0.825)], ps[int(len(p) * 0.8)]
            nb = (p >= lo) & (p < hi) & (y == 0)
            wsh.append(masks['White'][nb].mean())
            wage.append(age[nb & masks['White']].mean())
            bpos.append((p[masks['Black'] & (y == 1)] >= thr).mean())
        band[m] = {'white_share_noncases_in_band': float(np.mean(wsh)),
                   'white_noncase_mean_age_in_band': float(np.mean(wage)),
                   'black_positive_share_above_cutoff': float(np.mean(bpos))}
    band['all_noncases_white_share'] = float(masks['White'][y == 0].mean())
    band['all_white_noncase_mean_age'] = float(age[masks['White'] & (y == 0)].mean())
    out['cutoff_band'] = band

    with open(os.path.join(RESULTS_DIR, 'rebuttal_extras.json'), 'w') as f:
        json.dump(out, f, indent=1)

    f = lambda c: f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] ({c['n_pos']}/10)"
    print('year n', out['cdc_weights']['year_n'])
    for m in STRATEGIES:
        a = out['cdc_weights']['alloc'][m]
        print(f"CDC-wt {m:14s} sens_B {f(a['sens_Black'])} sel_B {f(a['sel_Black'])} gap {f(a['sens_gap_WB'])}"
              f" | AUC_B {f(out['cdc_weights']['auc'][m]['Black'])} AUCgap {f(out['cdc_weights']['auc'][m]['gap_WB'])}")
    for m in STRATEGIES:
        a = out['det_topk'][m]
        print(f"det-topk {m:14s} sens_B {f(a['sens_Black'])} gap {f(a['sens_gap_WB'])}")
    for m, v in out['boot_all'].items():
        print(f"boot {m:14s} " + ' '.join(f"{k} [{x['lo']:+.4f},{x['hi']:+.4f}]" for k, x in v.items()))
    print('band', json.dumps(band, indent=1))


if __name__ == '__main__':
    main()
