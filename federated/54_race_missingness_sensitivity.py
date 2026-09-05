"""
54_race_missingness_sensitivity.py -- does the threshold-setting population matter?
====================================================================================
36.1% of BRFSS respondents have no usable race code. The paper's threshold is set
on the pooled cohort INCLUDING these rows (policy a -- what the manuscript reports),
because race is unobserved at deployment. The obvious reviewer question: does the
headline White-Black sensitivity-gap widening survive if the threshold is instead
set on the race-known subset only (policy b)?

Uses cached multiseed predictions only. No retraining.

Input : results/preds_multiseed/*.npy
Output: results/race_missingness_sensitivity.json
Usage:  python 54_race_missingness_sensitivity.py
"""
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

_dl = __import__('35_delong_deltas')
load, discover = _dl.load, _dl.discover
RACE_STRATA, RACE_REPORTED, Z95 = _dl.RACE_STRATA, _dl.RACE_REPORTED, _dl.Z95

REFERENCE = 'Centralised-matched'
TREATMENT = 'FedAvg'
CAPACITY = 0.20


def threshold_from(scores, capacity):
    ps = np.sort(scores)
    k = int(np.clip(len(ps) * (1.0 - capacity), 0, len(ps) - 1))
    return ps[k]


def selection_sensitivity(p, thr, mask_pos, mask_neg):
    sp = np.sort(p[mask_pos])
    sn = np.sort(p[mask_neg])
    fp = len(sp) - np.searchsorted(sp, thr, side='left')
    fn = len(sn) - np.searchsorted(sn, thr, side='left')
    n_g = len(sp) + len(sn)
    sel = (fp + fn) / n_g
    sens = fp / len(sp) if len(sp) else float('nan')
    return sel, sens


def paired_ci(deltas):
    deltas = np.asarray(deltas, dtype=float)
    mean = float(deltas.mean())
    se = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
    return dict(
        mean=round(mean, 6),
        se=round(se, 6),
        ci_low=round(mean - Z95 * se, 6),
        ci_high=round(mean + Z95 * se, 6),
        n_seeds_positive=int((deltas > 0).sum()),
        n_seeds=len(deltas),
    )


def main():
    models, seeds = discover()
    if REFERENCE not in models or TREATMENT not in models:
        raise RuntimeError(f"Expected models missing. Found: {sorted(models)}")

    y = load('_y_brfss')
    race = load('_race_brfss')
    n_total = len(y)

    known_codes = sorted({c for codes in RACE_STRATA.values() for c in codes})
    known_mask = np.isin(race, known_codes)
    n_known, n_unknown = int(known_mask.sum()), int((~known_mask).sum())
    print(f"BRFSS n={n_total:,}  race-known={n_known:,} ({n_known/n_total:.1%})  "
          f"race-unknown={n_unknown:,} ({n_unknown/n_total:.1%})")

    white_mask = np.isin(race, RACE_STRATA['White'])
    black_mask = np.isin(race, RACE_STRATA['Black'])
    w_pos, w_neg = white_mask & (y == 1), white_mask & (y == 0)
    b_pos, b_neg = black_mask & (y == 1), black_mask & (y == 0)
    print(f"White n={white_mask.sum():,}  Black n={black_mask.sum():,}")
    assert white_mask.sum() == 617782, f"White stratum mismatch: {white_mask.sum()}"
    assert black_mask.sum() == 62770, f"Black stratum mismatch: {black_mask.sum()}"

    per_seed = {'a': {'gap': []}, 'b': {'gap': []}}
    per_seed_by_model = {
        m: {'a': {'sel_w': [], 'sel_b': [], 'sens_w': [], 'sens_b': []},
            'b': {'sel_w': [], 'sel_b': [], 'sens_w': [], 'sens_b': []}}
        for m in (REFERENCE, TREATMENT)
    }

    for seed in seeds:
        gaps = {}
        for policy, pop_mask in (('a', np.ones(n_total, dtype=bool)), ('b', known_mask)):
            for m in (REFERENCE, TREATMENT):
                p = load(f'{m}_seed{seed}_external')
                thr = threshold_from(p[pop_mask], CAPACITY)
                sel_w, sens_w = selection_sensitivity(p, thr, w_pos, w_neg)
                sel_b, sens_b = selection_sensitivity(p, thr, b_pos, b_neg)
                per_seed_by_model[m][policy]['sel_w'].append(sel_w)
                per_seed_by_model[m][policy]['sel_b'].append(sel_b)
                per_seed_by_model[m][policy]['sens_w'].append(sens_w)
                per_seed_by_model[m][policy]['sens_b'].append(sens_b)
                gaps[(policy, m)] = sens_w - sens_b
            gap_ref = gaps[(policy, REFERENCE)]
            gap_trt = gaps[(policy, TREATMENT)]
            per_seed[policy]['gap'].append(gap_trt - gap_ref)
        print(f"  seed {seed} done")

    out = {
        'capacity': CAPACITY,
        'reference': REFERENCE, 'treatment': TREATMENT,
        'n_total': n_total, 'n_known': n_known, 'n_unknown': n_unknown,
        'pct_known': round(n_known / n_total, 4), 'pct_unknown': round(n_unknown / n_total, 4),
        'n_white': int(white_mask.sum()), 'n_black': int(black_mask.sum()),
        'policy_a_full_cohort_threshold': {
            'description': 'Threshold set on the pooled cohort including race-unknown rows (the manuscript reported policy).',
            'wb_sensitivity_gap_delta_treatment_minus_reference': paired_ci(per_seed['a']['gap']),
        },
        'policy_b_race_known_threshold': {
            'description': 'Threshold set on the race-known subset only (sensitivity check).',
            'wb_sensitivity_gap_delta_treatment_minus_reference': paired_ci(per_seed['b']['gap']),
        },
        'by_model': {},
    }
    for m in (REFERENCE, TREATMENT):
        out['by_model'][m] = {}
        for policy in ('a', 'b'):
            d = per_seed_by_model[m][policy]
            out['by_model'][m][policy] = {
                k: {'mean': round(float(np.mean(v)), 6), 'sd': round(float(np.std(v, ddof=1)), 6)}
                for k, v in d.items()
            }

    a = out['policy_a_full_cohort_threshold']['wb_sensitivity_gap_delta_treatment_minus_reference']
    b = out['policy_b_race_known_threshold']['wb_sensitivity_gap_delta_treatment_minus_reference']
    pct_change = (b['mean'] - a['mean']) / a['mean'] * 100 if a['mean'] != 0 else float('nan')
    out['magnitude_comparison'] = {
        'wb_sensitivity_gap_delta': {
            'policy_a_full_cohort': a['mean'],
            'policy_b_race_known': b['mean'],
            'pct_change_a_to_b': round(pct_change, 1),
            'direction_preserved': bool((a['mean'] > 0) == (b['mean'] > 0)),
        },
        'black_selection_rate_delta': {
            'policy_a': round(float(np.mean(per_seed_by_model[TREATMENT]['a']['sel_b'])
                                     - np.mean(per_seed_by_model[REFERENCE]['a']['sel_b'])), 6),
            'policy_b': round(float(np.mean(per_seed_by_model[TREATMENT]['b']['sel_b'])
                                     - np.mean(per_seed_by_model[REFERENCE]['b']['sel_b'])), 6),
        },
        'white_selection_rate_delta': {
            'policy_a': round(float(np.mean(per_seed_by_model[TREATMENT]['a']['sel_w'])
                                     - np.mean(per_seed_by_model[REFERENCE]['a']['sel_w'])), 6),
            'policy_b': round(float(np.mean(per_seed_by_model[TREATMENT]['b']['sel_w'])
                                     - np.mean(per_seed_by_model[REFERENCE]['b']['sel_w'])), 6),
        },
    }

    print(f"\n{'='*88}\n  Race-missingness sensitivity: threshold population, White-Black gap\n{'='*88}")
    print(f"  {'quantity':<38}{'policy (a) full':>16}{'policy (b) known':>18}{'% change':>11}")
    mc = out['magnitude_comparison']
    print(f"  {'WB sensitivity gap delta (Fed-ctrl)':<38}{mc['wb_sensitivity_gap_delta']['policy_a_full_cohort']:>16.4f}"
          f"{mc['wb_sensitivity_gap_delta']['policy_b_race_known']:>18.4f}"
          f"{mc['wb_sensitivity_gap_delta']['pct_change_a_to_b']:>10.1f}%")
    print(f"  {'Black selection-rate delta':<38}{mc['black_selection_rate_delta']['policy_a']:>16.4f}"
          f"{mc['black_selection_rate_delta']['policy_b']:>18.4f}")
    print(f"  {'White selection-rate delta':<38}{mc['white_selection_rate_delta']['policy_a']:>16.4f}"
          f"{mc['white_selection_rate_delta']['policy_b']:>18.4f}")
    print(f"\n  direction preserved under policy (b): {mc['wb_sensitivity_gap_delta']['direction_preserved']}")
    print(f"  policy (a) 95% CI: [{a['ci_low']:+.4f}, {a['ci_high']:+.4f}]  ({a['n_seeds_positive']}/{a['n_seeds']} seeds positive)")
    print(f"  policy (b) 95% CI: [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  ({b['n_seeds_positive']}/{b['n_seeds']} seeds positive)")

    path = os.path.join(RESULTS_DIR, 'race_missingness_sensitivity.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == '__main__':
    main()
