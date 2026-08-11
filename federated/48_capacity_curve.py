"""
48_capacity_curve.py — is the reversal a property of 20%, or of the deployment?
==============================================================================
44_clinical_translation.py reports three capacities (10/20/30%). The obvious
reviewer objection is that the operating point was chosen after seeing the
result. This script removes the choice: it sweeps the screening capacity
continuously from 1% to 50% and reports, at every point, the full allocation
picture rather than sensitivity alone.

Per stratum, at each capacity, three quantities that answer three different
questions:

  selection rate  P(flagged | g)          who is offered the scarce slot
  sensitivity     P(flagged | g, Y=1)     of those who have the disease, who is found
  PPV             P(Y=1 | flagged, g)     what fraction of the offered slots pay off

Reporting all three closes the "you picked sensitivity because it looked bad"
objection: the three cannot all be gamed at once at a fixed global threshold.

One global threshold per capacity, applied to every stratum -- the same rule
44_clinical_translation.py uses, and the one a real programme deploys.

Method note: thresholds come from the pooled score distribution, and each
stratum's flagged counts come from `searchsorted` on its pre-sorted scores, so
the whole 50-point sweep costs one sort per model-seed rather than one per
capacity.

Input : results/preds_multiseed/*.npy
Output: results/capacity_curve.json

Usage:  python 48_capacity_curve.py
"""
import os, sys, json, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

_dl = __import__('35_delong_deltas')
load, discover = _dl.load, _dl.discover
RACE_STRATA, RACE_REPORTED = _dl.RACE_STRATA, _dl.RACE_REPORTED

Z95 = 1.959963984540054


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='Centralised-matched')
    ap.add_argument('--lo', type=float, default=0.01)
    ap.add_argument('--hi', type=float, default=0.50)
    ap.add_argument('--n', type=int, default=50)
    args = ap.parse_args()

    models, seeds = discover()
    ref = args.reference
    rates = np.linspace(args.lo, args.hi, args.n)

    y = load('_y_brfss')
    race = load('_race_brfss')
    masks = {g: np.isin(race, RACE_STRATA[g]) for g in RACE_REPORTED}

    idx_pos = {g: np.flatnonzero(masks[g] & (y == 1)) for g in RACE_REPORTED}
    idx_neg = {g: np.flatnonzero(masks[g] & (y == 0)) for g in RACE_REPORTED}

    # [model][stratum][metric] -> (n_seeds, n_rates)
    acc = {m: {g: {k: [] for k in ('sel', 'sens', 'ppv')} for g in RACE_REPORTED}
           for m in models}

    for seed in seeds:
        for m in models:
            p = load(f'{m}_seed{seed}_external')
            # One sort of the pooled scores gives every threshold in the sweep.
            ps = np.sort(p)
            k = np.clip((len(ps) * (1.0 - rates)).astype(int), 0, len(ps) - 1)
            thr = ps[k]                                   # ascending capacity -> descending thr
            for g in RACE_REPORTED:
                sp = np.sort(p[idx_pos[g]])
                sn = np.sort(p[idx_neg[g]])
                # flagged = scores >= thr; searchsorted 'left' counts scores < thr
                fp = len(sp) - np.searchsorted(sp, thr, side='left')
                fn = len(sn) - np.searchsorted(sn, thr, side='left')
                n_g = len(sp) + len(sn)
                acc[m][g]['sel'].append((fp + fn) / n_g)
                acc[m][g]['sens'].append(fp / len(sp))
                with np.errstate(divide='ignore', invalid='ignore'):
                    acc[m][g]['ppv'].append(np.where(fp + fn > 0, fp / (fp + fn), np.nan))
        print(f"  seed {seed} done")

    out = {'rates': [round(float(r), 4) for r in rates],
           'seeds': seeds, 'reference': ref, 'n_brfss': int(len(y)),
           'note': 'One global threshold per capacity, applied to every stratum. '
                   'Arrays are means over seeds unless suffixed _sd.',
           'strata_n': {g: int(masks[g].sum()) for g in RACE_REPORTED},
           'strata_prevalence': {g: round(float(y[masks[g]].mean()), 5)
                                 for g in RACE_REPORTED},
           'by_model': {}}

    for m in models:
        out['by_model'][m] = {}
        for g in RACE_REPORTED:
            e = {}
            for k in ('sel', 'sens', 'ppv'):
                a = np.asarray(acc[m][g][k])              # (n_seeds, n_rates)
                e[k] = [round(float(v), 6) for v in a.mean(0)]
                e[f'{k}_sd'] = [round(float(v), 6) for v in a.std(0, ddof=1)]
            out['by_model'][m][g] = e

    # White - Black gap curves, paired per seed so the seed term is the right one.
    gaps = {}
    for m in models:
        g_seed = (np.asarray(acc[m]['White']['sens'])
                  - np.asarray(acc[m]['Black']['sens']))          # (n_seeds, n_rates)
        gaps[m] = g_seed
        out['by_model'][m]['_wb_sens_gap'] = [round(float(v), 6) for v in g_seed.mean(0)]
        out['by_model'][m]['_wb_sens_gap_sd'] = [round(float(v), 6)
                                                 for v in g_seed.std(0, ddof=1)]

    # Paired change in the gap vs the reference, at every capacity.
    out['wb_gap_delta_vs_reference'] = {}
    for m in models:
        if m == ref:
            continue
        d = gaps[m] - gaps[ref]                                   # paired by seed
        mean, sd = d.mean(0), d.std(0, ddof=1)
        se = sd / np.sqrt(len(seeds))
        out['wb_gap_delta_vs_reference'][m] = {
            'delta': [round(float(v), 6) for v in mean],
            'ci_low': [round(float(v), 6) for v in mean - Z95 * se],
            'ci_high': [round(float(v), 6) for v in mean + Z95 * se],
            'n_seeds_positive': [int(v) for v in (d > 0).sum(0)],
            'frac_rates_ci_excludes_zero':
                round(float(((mean - Z95 * se) > 0).mean() + ((mean + Z95 * se) < 0).mean()), 4),
        }

    print(f"\n{'='*88}\n  White-Black SENSITIVITY gap vs screening capacity"
          f"\n{'='*88}")
    print(f"  {'capacity':>9}{'ref gap':>10}{'FedAvg gap':>12}{'delta':>10}"
          f"{'95% CI':>22}{'seeds+':>8}")
    show = list(range(0, len(rates), max(1, len(rates) // 12)))
    dd = out['wb_gap_delta_vs_reference']['FedAvg']
    for i in show:
        txt = f"[{dd['ci_low'][i]:+.4f}, {dd['ci_high'][i]:+.4f}]"
        print(f"  {rates[i]:>8.0%}{out['by_model'][ref]['_wb_sens_gap'][i]:>10.4f}"
              f"{out['by_model']['FedAvg']['_wb_sens_gap'][i]:>12.4f}"
              f"{dd['delta'][i]:>+10.4f}{txt:>22}{dd['n_seeds_positive'][i]:>6}/10")

    print(f"\n  fraction of the 1-50% sweep where the gap change is significant:")
    for m, d in out['wb_gap_delta_vs_reference'].items():
        allpos = int(np.sum(np.asarray(d['n_seeds_positive']) == len(seeds)))
        print(f"    {m:<20}{d['frac_rates_ci_excludes_zero']:>7.1%}"
              f"   ({allpos}/{len(rates)} capacities with 10/10 seeds positive)")

    print(f"\n{'='*88}\n  Full allocation picture at 20% capacity (FedAvg vs {ref})"
          f"\n{'='*88}")
    j = int(np.argmin(np.abs(rates - 0.20)))
    print(f"  {'stratum':<10}{'prev':>7}"
          f"{'sel ref':>9}{'sel Fed':>9}{'d sel':>9}"
          f"{'sens ref':>10}{'sens Fed':>10}{'d sens':>9}"
          f"{'PPV ref':>9}{'PPV Fed':>9}{'d PPV':>8}")
    for g in RACE_REPORTED:
        r_, f_ = out['by_model'][ref][g], out['by_model']['FedAvg'][g]
        print(f"  {g:<10}{out['strata_prevalence'][g]:>7.3f}"
              f"{r_['sel'][j]:>9.4f}{f_['sel'][j]:>9.4f}{f_['sel'][j]-r_['sel'][j]:>+9.4f}"
              f"{r_['sens'][j]:>10.4f}{f_['sens'][j]:>10.4f}{f_['sens'][j]-r_['sens'][j]:>+9.4f}"
              f"{r_['ppv'][j]:>9.4f}{f_['ppv'][j]:>9.4f}{f_['ppv'][j]-r_['ppv'][j]:>+8.4f}")

    path = os.path.join(RESULTS_DIR, 'capacity_curve.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == '__main__':
    main()
