"""
55_race_missingness_all_strategies.py -- extends 54_ across all five FL strategies
====================================================================================
54_race_missingness_sensitivity.py checked FedAvg only, at 20% capacity, against
the composition-matched control. The manuscript's headline claim is "for all five
strategies" (FedAvg, FedProx, FedNova, SCAFFOLD, FedAvg-EqualW). This script closes
that gap: same race-known-threshold sensitivity check, run for every strategy, at
every capacity from 1% to 50% -- the full sweep 48_capacity_curve.py already uses,
not just the single 20% point.

Uses cached multiseed predictions only. No retraining.

Input : results/preds_multiseed/*.npy
Output: results/race_missingness_all_strategies.json
Usage:  python 55_race_missingness_all_strategies.py
"""
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

_dl = __import__('35_delong_deltas')
load, discover = _dl.load, _dl.discover
RACE_STRATA, Z95 = _dl.RACE_STRATA, _dl.Z95

REFERENCE = 'Centralised-matched'
STRATEGIES = ['FedAvg', 'FedProx', 'FedNova', 'FedAvg-EqualW', 'SCAFFOLD']
RATES = np.linspace(0.01, 0.50, 50)


def threshold_from(scores, capacity):
    ps = np.sort(scores)
    k = int(np.clip(len(ps) * (1.0 - capacity), 0, len(ps) - 1))
    return ps[k]


def sensitivity(p, thr, mask_pos):
    sp = np.sort(p[mask_pos])
    fp = len(sp) - np.searchsorted(sp, thr, side='left')
    return fp / len(sp) if len(sp) else float('nan')


def paired_ci(deltas):
    deltas = np.asarray(deltas, dtype=float)
    mean = float(deltas.mean())
    se = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
    return dict(
        mean=round(mean, 6), se=round(se, 6),
        ci_low=round(mean - Z95 * se, 6), ci_high=round(mean + Z95 * se, 6),
        n_seeds_positive=int((deltas > 0).sum()), n_seeds=len(deltas),
    )


def main():
    models, seeds = discover()
    missing = [m for m in STRATEGIES + [REFERENCE] if m not in models]
    if missing:
        raise RuntimeError(f"Missing cached models: {missing}. Found: {sorted(models)}")

    y = load('_y_brfss')
    race = load('_race_brfss')
    n_total = len(y)

    known_codes = sorted({c for codes in RACE_STRATA.values() for c in codes})
    known_mask = np.isin(race, known_codes)

    white_mask = np.isin(race, RACE_STRATA['White'])
    black_mask = np.isin(race, RACE_STRATA['Black'])
    assert white_mask.sum() == 617782 and black_mask.sum() == 62770
    # Sensitivity is over CASES in the stratum, not the whole stratum. Passing
    # the bare stratum mask would compute the selection rate instead, which is a
    # different quantity from the one 54_ reports and the manuscript quotes.
    w_pos, b_pos = white_mask & (y == 1), black_mask & (y == 1)

    # Cache all scores once per (model, seed) -- avoid re-reading npy per capacity.
    cache = {}
    for seed in seeds:
        for m in STRATEGIES + [REFERENCE]:
            cache[(m, seed)] = load(f'{m}_seed{seed}_external')

    out = {'reference': REFERENCE, 'strategies': STRATEGIES,
           'capacities_swept': [round(float(r), 4) for r in RATES],
           'capacity_20pct': {}, 'full_sweep_summary': {}}

    for treatment in STRATEGIES:
        # --- 20% capacity, both policies, matching 54_'s exact quantity ---
        gap_a, gap_b = [], []
        for seed in seeds:
            row = {}
            for policy, pop_mask in (('a', np.ones(n_total, dtype=bool)), ('b', known_mask)):
                for m in (REFERENCE, treatment):
                    p = cache[(m, seed)]
                    thr = threshold_from(p[pop_mask], 0.20)
                    sens_w = sensitivity(p, thr, w_pos)
                    sens_b = sensitivity(p, thr, b_pos)
                    row[(policy, m)] = sens_w - sens_b
            gap_a.append(row[('a', treatment)] - row[('a', REFERENCE)])
            gap_b.append(row[('b', treatment)] - row[('b', REFERENCE)])
        a_ci, b_ci = paired_ci(gap_a), paired_ci(gap_b)
        pct_change = (b_ci['mean'] - a_ci['mean']) / a_ci['mean'] * 100 if a_ci['mean'] != 0 else float('nan')
        out['capacity_20pct'][treatment] = {
            'policy_a_full_cohort': a_ci, 'policy_b_race_known': b_ci,
            'pct_change_a_to_b': round(pct_change, 1),
            'direction_preserved': bool((a_ci['mean'] > 0) == (b_ci['mean'] > 0)),
        }

        # --- full 1-50% sweep, policy (a) vs (b), summarised ---
        n_capacities_direction_preserved = 0
        n_capacities_both_10of10 = 0
        worst_pct_change = 0.0
        for rate in RATES:
            gap_a_r, gap_b_r = [], []
            for seed in seeds:
                row = {}
                for policy, pop_mask in (('a', np.ones(n_total, dtype=bool)), ('b', known_mask)):
                    for m in (REFERENCE, treatment):
                        p = cache[(m, seed)]
                        thr = threshold_from(p[pop_mask], rate)
                        sens_w = sensitivity(p, thr, w_pos)
                        sens_b = sensitivity(p, thr, b_pos)
                        row[(policy, m)] = sens_w - sens_b
                gap_a_r.append(row[('a', treatment)] - row[('a', REFERENCE)])
                gap_b_r.append(row[('b', treatment)] - row[('b', REFERENCE)])
            ma, mb = np.mean(gap_a_r), np.mean(gap_b_r)
            same_dir = (ma > 0) == (mb > 0)
            if same_dir:
                n_capacities_direction_preserved += 1
            pos_a = sum(1 for v in gap_a_r if v > 0)
            pos_b = sum(1 for v in gap_b_r if v > 0)
            if pos_a == 10 and pos_b == 10:
                n_capacities_both_10of10 += 1
            if ma != 0:
                pct = abs((mb - ma) / ma * 100)
                worst_pct_change = max(worst_pct_change, pct)

        out['full_sweep_summary'][treatment] = {
            'n_capacities': len(RATES),
            'direction_preserved_at_n_capacities': n_capacities_direction_preserved,
            'both_policies_10of10_seeds_at_n_capacities': n_capacities_both_10of10,
            'worst_case_pct_change_across_sweep': round(worst_pct_change, 1),
        }
        print(f"{treatment:16s} 20%: a={a_ci['mean']:+.4f} b={b_ci['mean']:+.4f} "
              f"({pct_change:+.1f}%)  |  sweep: direction preserved "
              f"{n_capacities_direction_preserved}/{len(RATES)}, "
              f"worst-case change {worst_pct_change:.1f}%")

    path = os.path.join(RESULTS_DIR, 'race_missingness_all_strategies.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == '__main__':
    main()
