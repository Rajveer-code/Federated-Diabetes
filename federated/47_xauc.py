"""
47_xauc.py — the bridge between the two panels: cross-group ranking
===================================================================
The paper's two readings disagree because AUC is a WITHIN-group statistic and a
screening threshold is a GLOBAL one. That prose claim has an exact quantity
behind it. Kallus & Zhou (NeurIPS 2019) define

    xAUC_{g->h} = P( score(positive from g) > score(negative from h) )

and note that xAUC_{g->g} is the ordinary within-group AUC. The pooled AUC is
then an exact convex combination of the whole matrix:

    AUC_pooled = sum_g sum_h  pi^+_g  pi^-_h  xAUC_{g->h}                   (*)

where pi^+_g is g's share of all positives and pi^-_h is h's share of all
negatives. Both sides are computed here and asserted equal, so the decomposition
is verified numerically rather than asserted.

That identity is what makes the reversal legible. Federation can raise every
DIAGONAL cell (within-stratum AUC, the thing a standard fairness audit reports)
while lowering the OFF-DIAGONAL cells that place minority cases against majority
non-cases -- and it is the off-diagonal cells, not the diagonal, that decide who
clears one global threshold. Sensitivity at fixed capacity is a consequence;
xAUC is the rank-based, threshold-free measurement of the same shift.

Reported per model, versus the composition-matched centralised control:
  * the full 5x5 xAUC matrix and its change
  * xAUC disparity  D_g = xAUC_{g->White} - xAUC_{White->g}, negative meaning
    g's cases are ranked below White non-cases more often than the converse
  * the identity check for (*)

Inference reuses the DeLong machinery in 35_delong_deltas.py: xAUC_{g->h} is an
ordinary AUC on the subproblem {positives of g} vs {negatives of h}, so the
paired two-model DeLong covariance applies unchanged, and the seed-averaged CI
uses the same shared-component variance model (every seed scores the same BRFSS
rows, so that term does not shrink with more seeds).

Input : results/preds_multiseed/*.npy
Output: results/xauc.json

Usage:  python 47_xauc.py [--reference Centralised-matched]
"""
import os, sys, json, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
_dl = __import__('35_delong_deltas')
paired_delta, delong_single, ci, across_seeds = (
    _dl.paired_delta, _dl.delong_single, _dl.ci, _dl.across_seeds)
RACE_STRATA, RACE_REPORTED = _dl.RACE_STRATA, _dl.RACE_REPORTED
load, discover = _dl.load, _dl.discover

PIVOT = 'White'   # the majority stratum the global threshold is effectively set by


def xauc_rows(y, race_mask_g, race_mask_h):
    """
    Row indices and labels for the xAUC_{g->h} subproblem: positives drawn from
    stratum g, negatives from stratum h. Returns (idx, y_sub).

    For g == h this is exactly the within-stratum AUC subproblem, so the
    diagonal of the matrix needs no special case.
    """
    pos = np.flatnonzero(race_mask_g & (y == 1))
    neg = np.flatnonzero(race_mask_h & (y == 0))
    idx = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return idx, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='Centralised-matched')
    args = ap.parse_args()

    models, seeds = discover()
    ref = args.reference
    if ref not in models:
        sys.exit(f"Reference '{ref}' not in cache. Available: {models}")
    others = [m for m in models if m != ref]

    y = load('_y_brfss')
    race = load('_race_brfss')
    masks = {g: np.isin(race, RACE_STRATA[g]) for g in RACE_REPORTED}

    # Positive/negative shares over the five reported strata only -- the
    # 'Unknown' stratum is a missingness artefact and is excluded throughout,
    # so the identity below is stated over that restricted population.
    keep = np.zeros(len(y), dtype=bool)
    for m in masks.values():
        keep |= m
    n_pos = {g: int((masks[g] & (y == 1)).sum()) for g in RACE_REPORTED}
    n_neg = {g: int((masks[g] & (y == 0)).sum()) for g in RACE_REPORTED}
    tot_pos, tot_neg = sum(n_pos.values()), sum(n_neg.values())
    pi_pos = {g: n_pos[g] / tot_pos for g in RACE_REPORTED}
    pi_neg = {g: n_neg[g] / tot_neg for g in RACE_REPORTED}

    print(f"Cache: {len(models)} models x {len(seeds)} seeds  |  "
          f"BRFSS n={len(y):,}, reported strata n={int(keep.sum()):,}")
    print(f"Reference: {ref}\n")
    print(f"  {'stratum':<10}{'n+':>9}{'n-':>10}{'pi+':>8}{'pi-':>8}")
    for g in RACE_REPORTED:
        print(f"  {g:<10}{n_pos[g]:>9,}{n_neg[g]:>10,}"
              f"{pi_pos[g]:>8.4f}{pi_neg[g]:>8.4f}")

    # Precompute the 25 subproblems once; they do not depend on the model.
    cells = {}
    for g in RACE_REPORTED:
        for h in RACE_REPORTED:
            cells[(g, h)] = xauc_rows(y, masks[g], masks[h])

    per_seed = {m: {f'{g}->{h}': [] for g in RACE_REPORTED for h in RACE_REPORTED}
                for m in others}
    abs_xauc = {m: {f'{g}->{h}': [] for g in RACE_REPORTED for h in RACE_REPORTED}
                for m in models}
    identity = []

    for si, seed in enumerate(seeds, 1):
        p_ref = load(f'{ref}_seed{seed}_external')
        p_by = {m: load(f'{m}_seed{seed}_external') for m in others}

        for (g, h), (idx, lab) in cells.items():
            key = f'{g}->{h}'
            a_ref, _ = delong_single(lab, p_ref[idx])
            abs_xauc[ref][key].append(a_ref)
            for m in others:
                a_m, _, d, v = paired_delta(lab, p_by[m][idx], p_ref[idx])
                abs_xauc[m][key].append(a_m)
                per_seed[m][key].append((d, float(np.sqrt(v))))

        # Identity check (*): pooled AUC over the five reported strata must equal
        # the pi-weighted sum of the 25 cells. Checked for every model, every seed.
        for m in models:
            p = p_ref if m == ref else p_by[m]
            pooled, _ = delong_single(y[keep], p[keep])
            recon = sum(pi_pos[g] * pi_neg[h] * abs_xauc[m][f'{g}->{h}'][-1]
                        for g in RACE_REPORTED for h in RACE_REPORTED)
            identity.append({'model': m, 'seed': seed,
                             'auc_pooled': round(float(pooled), 8),
                             'auc_reconstructed': round(float(recon), 8),
                             'abs_error': float(f'{abs(pooled - recon):.3g}')})
        print(f"  seed {seed} done ({si}/{len(seeds)})")

    max_err = max(r['abs_error'] for r in identity)
    print(f"\n  decomposition identity: max |pooled - reconstructed| = {max_err:.2e} "
          f"over {len(identity)} model-seed pairs")
    assert max_err < 1e-8, f"decomposition identity violated: {max_err}"

    # ── aggregate across seeds ────────────────────────────────────────────────
    across = {}
    for m in others:
        across[m] = {}
        for key, vals in per_seed[m].items():
            d = [x[0] for x in vals]
            s = [x[1] for x in vals]          # all seeds share the same BRFSS rows
            across[m][key] = {**across_seeds(d, s), 'shared_component': 'self'}

        # xAUC disparity vs the pivot stratum. The two cells g->White and
        # White->g use disjoint row sets on the positive side but SHARE the
        # negative side across strata, so their deltas are not independent;
        # rather than assume a covariance, the disparity is formed per seed and
        # its scatter is carried by the seed term alone (shared term set to the
        # conservative sum of the two SEs).
        for g in RACE_REPORTED:
            if g == PIVOT:
                continue
            a = per_seed[m][f'{g}->{PIVOT}']
            b = per_seed[m][f'{PIVOT}->{g}']
            d = [x[0] - z[0] for x, z in zip(a, b)]
            s = [x[1] + z[1] for x, z in zip(a, b)]   # conservative (Cov >= -1)
            across[m][f'disparity_{g}'] = {**across_seeds(d, s),
                                           'shared_component': 'conservative_sum'}

    # ── report ────────────────────────────────────────────────────────────────
    hdr = 'pos \\ neg'.ljust(10) + ''.join(f'{h:>10}' for h in RACE_REPORTED)

    def matrix(model, which):
        src = abs_xauc[model]
        print(f"\n  {which}\n  {hdr}")
        for g in RACE_REPORTED:
            row = ''.join(f"{np.mean(src[f'{g}->{h}']):>10.4f}" for h in RACE_REPORTED)
            print(f"  {g:<10}{row}")

    print(f"\n{'='*80}\n  xAUC matrices, mean over {len(seeds)} seeds "
          f"(diagonal = within-stratum AUC)\n{'='*80}")
    matrix(ref, f'{ref} (reference)')
    matrix('FedAvg', 'FedAvg')

    print(f"\n  FedAvg - {ref}\n  {hdr}")
    for g in RACE_REPORTED:
        row = ''.join(f"{across['FedAvg'][f'{g}->{h}']['mean']:>+10.4f}"
                      for h in RACE_REPORTED)
        print(f"  {g:<10}{row}")

    print(f"\n{'='*94}\n  Change in cross-group ranking vs {PIVOT} "
          f"(negative = that stratum's cases pushed down)\n{'='*94}")
    print(f"  {'model':<18}{'cell':<20}{'mean d':>9}{'SE':>9}"
          f"{'95% CI':>22}{'p':>10}{'sign':>8}")
    for m in others:
        for g in RACE_REPORTED:
            if g == PIVOT:
                continue
            s = across[m][f'{g}->{PIVOT}']; c = s['combined']
            star = '*' if c['excludes_zero'] else ' '
            txt = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
            sign = f"{s['n_negative']}-/{s['n_positive']}+"
            cell = f'{g}->{PIVOT}'
            print(f"  {m:<18}{cell:<20}{s['mean']:>+9.4f}{c['se']:>9.4f}"
                  f"{txt:>21}{star}{c['p']:>10.3g}{sign:>8}")
        print()

    out = os.path.join(RESULTS_DIR, 'xauc.json')
    with open(out, 'w') as f:
        json.dump({
            'definition': 'xAUC_{g->h} = P(score(positive from g) > score(negative '
                          'from h)); Kallus & Zhou, NeurIPS 2019. Diagonal is the '
                          'ordinary within-stratum AUC.',
            'identity': 'AUC_pooled = sum_g sum_h pi+_g pi-_h xAUC_{g->h}, over the '
                        'five reported strata; verified numerically below.',
            'identity_max_abs_error': max_err,
            'identity_checks': identity,
            'reference': ref, 'seeds': seeds, 'pivot': PIVOT,
            'n_positive': n_pos, 'n_negative': n_neg,
            'share_positive': {k: round(v, 6) for k, v in pi_pos.items()},
            'share_negative': {k: round(v, 6) for k, v in pi_neg.items()},
            'xauc_mean': {m: {k: round(float(np.mean(v)), 5)
                              for k, v in d.items()} for m, d in abs_xauc.items()},
            'across_seeds': across,
        }, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
