"""
35_delong_deltas.py — T1.4 (paired Δ statistics with CIs) + T1.5 (subgroup decomposition)
=========================================================================================
Turns the multi-seed point estimates from 33_multiseed_all_models.py into
inferential statements: for every model, the difference from the
composition-matched centralised control, with a 95% CI that accounts for the
fact that both models are scored on the *same* BRFSS rows.

Method — DeLong, not bootstrap
------------------------------
A paired bootstrap over 1,282,897 rows x 7 models x N seeds is computationally
prohibitive. DeLong (1988), computed with the O(n log n) algorithm of Sun & Xu
(2014), gives the covariance of two AUCs on shared data analytically. It is the
standard estimator for exactly this comparison.

Variance algebra used here
--------------------------
* Delta external AUC — both models scored on identical rows, so the DeLong 2x2
  covariance is used directly:  Var = C00 + C11 - 2*C01.
* Delta subgroup AUC  — same, restricted to the subgroup's rows.
* Delta fairness gap  — fair_gap = AUC_young - AUC_elderly, so
  Delta fair_gap = Delta_young - Delta_elderly. The young and elderly row sets
  are disjoint, hence Cov(Delta_young, Delta_elderly) = 0 and
  Var = Var(Delta_young) + Var(Delta_elderly).
* Delta generalisation gap — gap = AUC_internal - AUC_external, and the internal
  and external sets are disjoint datasets, so
  Var = Var(Delta_internal) + Var(Delta_external).
  For the internal term: two federated strategies at the same seed share one
  held-out set, so the paired DeLong covariance applies. A federated model and a
  centralised model do NOT share a held-out set (each is a different 20% split of
  the same 11,906-row cohort, overlapping by roughly 20% of their rows), so their
  AUCs are treated as independent and the variances are added. That overlap is
  positively correlated, so adding the variances OVERSTATES the true variance —
  the resulting CI is conservative, which is the safe direction.

Cross-seed evidence
-------------------
Internal AUC is measured on only ~2,400 rows, so its DeLong SE (~0.015) dominates
the generalisation-gap CI and a single seed cannot resolve an effect of ~0.014.
Consistency of sign across independent seeds is therefore the stronger evidence,
and both a paired t-test and an exact sign test across seeds are reported
alongside the per-seed CIs. Neither replaces the other; both are printed.

Input : results/preds_multiseed/*.npy   (written by 33_multiseed_all_models.py)
Output: results/delong_deltas.json

Usage:
  python 35_delong_deltas.py
  python 35_delong_deltas.py --reference Centralised-full
"""
import os, sys, json, glob, argparse, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

PRED_DIR = os.path.join(RESULTS_DIR, 'preds_multiseed')
FL_MODELS = {'FedAvg', 'FedProx', 'FedNova', 'FedAvg-EqualW'}
Z95 = 1.959963984540054

# Age bands must match fairness_gap() in 33_multiseed_all_models.py.
SUBGROUPS = {
    'young':   lambda a: (a >= 18) & (a <= 39),
    'mid':     lambda a: (a >= 40) & (a <= 59),
    'elderly': lambda a: a >= 60,
}

# Race strata are an EVALUATION grouping only — race is not a model input; see
# config_paths.FEATURE_COLS_NORACE. Codes are pre-imputation, so the 34.7% of
# BRFSS rows with race missing stay their own stratum instead of being folded
# into the mode. 'Unknown' is reported but excluded from the disparity summary,
# since it is a missingness artefact rather than a population.
RACE_STRATA = {
    'White':    [1.0],
    'Black':    [2.0],
    'Hispanic': [8.0],
    'Asian':    [4.0],
    'Other':    [3.0, 5.0, 6.0, 7.0],
}
RACE_REPORTED = ['White', 'Black', 'Hispanic', 'Asian', 'Other']


# ── DeLong ────────────────────────────────────────────────────────────────────

def compute_midrank(x):
    """Midranks of x (ties share the average of the ranks they span)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """
    Sun & Xu (2014) O(n log n) DeLong. `preds_sorted` is (k, n) with all m
    positive-label columns first. Returns (aucs[k], covariance[k, k]).
    """
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(preds_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01  = (tz[:, :m] - tx) / n
    v10  = 1.0 - (tz[:, m:] - ty) / m
    cov  = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_pair(y, p_a, p_b):
    """Two AUCs on the SAME rows + their 2x2 DeLong covariance."""
    order = np.argsort(-y, kind='stable')          # positives first
    m = int(y.sum())
    if m == 0 or m == len(y):
        raise ValueError('subgroup has only one outcome class')
    aucs, cov = _fast_delong(np.vstack([p_a[order], p_b[order]]), m)
    return float(aucs[0]), float(aucs[1]), cov


def delong_single(y, p):
    """AUC and its DeLong variance for one model."""
    a, _, cov = delong_pair(y, p, p)
    return a, float(cov[0, 0])


def paired_delta(y, p_a, p_b):
    """(auc_a, auc_b, delta, var_delta) for two models on shared rows."""
    a, b, C = delong_pair(y, p_a, p_b)
    var = float(C[0, 0] + C[1, 1] - 2.0 * C[0, 1])
    return a, b, a - b, max(var, 0.0)


def ci(delta, var):
    """95% CI + two-sided z p-value for an estimate with known variance."""
    se = float(np.sqrt(var))
    lo, hi = delta - Z95 * se, delta + Z95 * se
    if se == 0:
        p = 1.0
    else:
        from math import erfc, sqrt
        p = float(erfc(abs(delta / se) / sqrt(2.0)))
    return dict(delta=round(delta, 5), se=round(se, 5),
                ci_low=round(lo, 5), ci_high=round(hi, 5),
                p=float(f'{p:.3g}'), excludes_zero=bool(lo > 0 or hi < 0))


# ── Cache access ──────────────────────────────────────────────────────────────

def load(name):
    return np.load(os.path.join(PRED_DIR, f'{name}.npy'))


def internal_y_name(model, seed):
    """FL strategies share one held-out set per seed; others have their own."""
    return (f'_y_internal_fl_seed{seed}' if model in FL_MODELS
            else f'_y_internal_{model}_seed{seed}')


def discover():
    """
    (models, seeds) present in the cache. Only seeds for which every model has
    both an external and an internal prediction file are returned, so this is
    safe to run while 33_multiseed_all_models.py is still training later seeds.
    """
    by_seed = {}
    models = set()
    for f in glob.glob(os.path.join(PRED_DIR, '*_external.npy')):
        base = os.path.basename(f)[:-len('_external.npy')]
        if base.startswith('_'):
            continue
        model, _, s = base.rpartition('_seed')
        models.add(model)
        by_seed.setdefault(int(s), set()).add(model)

    def complete(seed):
        return all(
            os.path.exists(os.path.join(PRED_DIR, f'{m}_seed{seed}_internal.npy'))
            and os.path.exists(os.path.join(PRED_DIR, internal_y_name(m, seed) + '.npy'))
            for m in models
        )

    seeds = sorted(s for s, ms in by_seed.items() if ms == models and complete(s))
    dropped = sorted(set(by_seed) - set(seeds))
    if dropped:
        print(f"  (skipping incomplete seeds: {dropped})")
    return sorted(models), seeds


# ── Cross-seed tests ──────────────────────────────────────────────────────────

def across_seeds(deltas, ses):
    """
    Combine the two independent sources of uncertainty in the seed-averaged delta.

    Write the per-seed estimate as  delta_s = theta + a + b_s, where `a` is the
    error from data that is IDENTICAL at every seed and `b_s` is everything that
    varies with the seed. Then

        Var(mean delta) = Var(a) + Var(b) / n_seeds

    and, crucially, Var(b) is estimated by the empirical seed variance sd^2 —
    because `a` is common to all seeds it contributes no scatter at all. So the
    caller passes the DeLong SE of the SHARED component only, in `ses`; adding
    the full per-seed DeLong variance instead would double-count every source of
    noise that already shows up in sd^2.

    Which component is shared depends on the metric:
      * external / subgroup / fairness deltas — all scored on the same
        1,282,897 BRFSS rows at every seed, so the whole DeLong variance is
        shared and does not shrink no matter how many seeds are run;
      * the internal-AUC delta — each seed draws a DIFFERENT held-out split, so
        nothing is shared; its sampling error lives entirely in sd^2 and shrinks
        as 1/n_seeds;
      * the generalisation gap (internal minus external) — only its external
        term is shared, so that is the term passed in.

    Reporting the DeLong CI alone understates uncertainty (it ignores
    initialisation); reporting the seed SD alone understates it too (it ignores
    the shared test set). Both components are kept below so the split stays
    visible.

    Also reports an exact two-sided sign test, which assumes nothing about the
    variance structure at all.
    """
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    out = {'n_seeds': n,
           'mean': round(float(d.mean()), 5),
           'sd': round(float(d.std(ddof=1)), 5) if n > 1 else None,
           'n_negative': int((d < 0).sum()),
           'n_positive': int((d > 0).sum())}

    var_test  = float(np.mean(np.asarray(ses, dtype=float) ** 2))   # shared data
    var_train = float(d.std(ddof=1) ** 2 / n) if n > 1 else 0.0     # per-seed
    se_comb   = float(np.sqrt(var_test + var_train))
    out['se_shared_component']  = round(float(np.sqrt(var_test)), 5)
    out['se_perseed_component'] = round(float(np.sqrt(var_train)), 5)
    out['combined'] = ci(float(d.mean()), var_test + var_train)
    out['combined']['se'] = round(se_comb, 5)
    if n > 1 and d.std(ddof=1) > 0:
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
        out['t'] = round(t, 3)
        try:
            from scipy import stats
            out['t_p'] = float(f"{2 * stats.t.sf(abs(t), n - 1):.3g}")
        except ImportError:
            out['t_p'] = None
    # Exact sign test: P(all n on one side | fair coin), two-sided.
    k = max(out['n_negative'], out['n_positive'])
    from math import comb
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    out['sign_test_p'] = float(f'{min(1.0, 2 * tail):.4g}')
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='Centralised-matched',
                    help='model every other model is compared against')
    args = ap.parse_args()

    if not os.path.isdir(PRED_DIR):
        sys.exit(f"No prediction cache at {PRED_DIR}. Run 33_multiseed_all_models.py first.")

    models, seeds = discover()
    ref = args.reference
    if ref not in models:
        sys.exit(f"Reference '{ref}' not in cache. Available: {models}")
    others = [m for m in models if m != ref]

    y_b  = load('_y_brfss')
    age  = load('_age_brfss')
    race = load('_race_brfss')
    masks = {k: fn(age) for k, fn in SUBGROUPS.items()}
    for name, codes in RACE_STRATA.items():
        masks[f'race_{name}'] = np.isin(race, codes)
    # Code 9 is "don't know/refused"; NaN is item-missing. Both are missingness,
    # not a population, so they are reported but kept out of the disparity metric.
    masks['race_Unknown'] = np.isnan(race) | np.isin(race, [9.0])

    print(f"Cache: {len(models)} models x {len(seeds)} seeds  |  BRFSS n={len(y_b):,}")
    print(f"Reference: {ref}")
    for k, m in masks.items():
        n = int(m.sum())
        print(f"  {k:<14} n={n:>9,} ({n / len(y_b):5.1%})  "
              f"prevalence={y_b[m].mean():.4f}")

    per_seed, absolutes = [], []

    for seed in seeds:
        p_ref = load(f'{ref}_seed{seed}_external')
        y_ref_int = load(internal_y_name(ref, seed))
        p_ref_int = load(f'{ref}_seed{seed}_internal')
        auc_ref_int, var_ref_int = delong_single(y_ref_int, p_ref_int)

        for model in models:
            p_m = load(f'{model}_seed{seed}_external')
            a_ext, var_ext = delong_single(y_b, p_m)
            row = {'model': model, 'seed': seed,
                   'auc_external': round(a_ext, 5),
                   'auc_external_ci': [round(a_ext - Z95 * np.sqrt(var_ext), 5),
                                       round(a_ext + Z95 * np.sqrt(var_ext), 5)]}
            for g, mask in masks.items():
                a_g, v_g = delong_single(y_b[mask], p_m[mask])
                row[f'auc_{g}'] = round(a_g, 5)
                row[f'auc_{g}_ci'] = [round(a_g - Z95 * np.sqrt(v_g), 5),
                                      round(a_g + Z95 * np.sqrt(v_g), 5)]
            row['fair_gap'] = round(row['auc_young'] - row['auc_elderly'], 5)
            row['race_gap'] = round(row['auc_race_White'] - row['auc_race_Black'], 5)
            row['race_spread'] = round(
                max(row[f'auc_race_{r}'] for r in RACE_REPORTED)
                - min(row[f'auc_race_{r}'] for r in RACE_REPORTED), 5)
            absolutes.append(row)

        for model in others:
            p_m = load(f'{model}_seed{seed}_external')
            rec = {'model': model, 'reference': ref, 'seed': seed}

            # --- external discrimination -------------------------------------
            _, _, d_ext, v_ext = paired_delta(y_b, p_m, p_ref)
            rec['d_external'] = ci(d_ext, v_ext)

            # --- T1.5 subgroup decomposition ---------------------------------
            sub = {}
            for g, mask in masks.items():
                _, _, d_g, v_g = paired_delta(y_b[mask], p_m[mask], p_ref[mask])
                sub[g] = (d_g, v_g)
                rec[f'd_{g}'] = ci(d_g, v_g)

            # --- fairness gaps: disjoint subgroups, so variances add ----------
            d_fair = sub['young'][0] - sub['elderly'][0]
            rec['d_fair_gap'] = ci(d_fair, sub['young'][1] + sub['elderly'][1])
            # T2.1: the White-Black AUC gap, the race analogue of the age gap.
            d_race = sub['race_White'][0] - sub['race_Black'][0]
            rec['d_race_gap'] = ci(d_race,
                                   sub['race_White'][1] + sub['race_Black'][1])

            # --- generalisation gap ------------------------------------------
            y_m_int = load(internal_y_name(model, seed))
            p_m_int = load(f'{model}_seed{seed}_internal')
            shared_internal = (model in FL_MODELS and ref in FL_MODELS)
            if shared_internal:
                a_mi, a_ri, d_int, v_int = paired_delta(y_m_int, p_m_int, p_ref_int)
            else:
                a_mi, v_mi = delong_single(y_m_int, p_m_int)
                a_ri, v_ri = auc_ref_int, var_ref_int
                d_int, v_int = a_mi - a_ri, v_mi + v_ri   # conservative
            rec['internal_pairing'] = 'shared' if shared_internal else 'independent'
            rec['auc_internal'] = round(a_mi, 5)
            rec['d_internal'] = ci(d_int, v_int)
            rec['d_gen_gap'] = ci(d_int - d_ext, v_int + v_ext)
            per_seed.append(rec)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    metrics = (['d_external', 'd_young', 'd_mid', 'd_elderly',
                'd_fair_gap', 'd_internal', 'd_gen_gap', 'd_race_gap']
               + [f'd_race_{r}' for r in RACE_REPORTED])

    # Which part of each delta is computed on data identical at every seed, and
    # therefore does not shrink when more seeds are averaged. See across_seeds().
    SHARED_COMPONENT = {
        'd_external': 'self', 'd_young': 'self', 'd_mid': 'self',
        'd_elderly': 'self', 'd_fair_gap': 'self', 'd_race_gap': 'self',
        'd_internal': None,           # a different held-out split at every seed
        'd_gen_gap': 'd_external',    # only the external term is shared
        **{f'd_race_{r}': 'self' for r in RACE_REPORTED},
    }
    seed_tests = {}
    for model in others:
        rows = [r for r in per_seed if r['model'] == model]
        seed_tests[model] = {}
        for m in metrics:
            src = SHARED_COMPONENT[m]
            if src is None:            # nothing shared across seeds
                ses = [0.0] * len(rows)
            else:                      # 'self', or another metric's shared part
                ses = [r[m if src == 'self' else src]['se'] for r in rows]
            seed_tests[model][m] = {
                **across_seeds([r[m]['delta'] for r in rows], ses),
                'shared_component': src,
                'n_seeds_ci_excludes_zero': sum(r[m]['excludes_zero'] for r in rows),
            }

    # ── Report ────────────────────────────────────────────────────────────────
    def block(metric, title, note=''):
        print(f"\n{'='*94}\n  {title}   (vs {ref}){'   ' + note if note else ''}\n{'='*94}")
        # ASCII only: the Windows console codepage (cp1252) cannot encode Greek.
        print(f"  {'model':<16}{'mean d':>9}{'comb SE':>9}"
              f"{'95% CI (combined)':>22}{'p':>9}{'sign':>8}{'sign p':>9}"
              f"{'1-seed CIs':>12}")
        print(f"  {'-'*92}")
        for model in others:
            s = seed_tests[model][metric]
            c = s['combined']
            sign = f"{s['n_negative']}-/{s['n_positive']}+"
            star = '*' if c['excludes_zero'] else ' '
            ci_txt = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
            print(f"  {model:<16}{s['mean']:>+9.4f}{c['se']:>9.4f}"
                  f"{ci_txt:>21}{star}"
                  f"{c['p']:>9.4g}{sign:>8}{s['sign_test_p']:>9.4f}"
                  f"{s['n_seeds_ci_excludes_zero']:>7}/{len(seeds)}")

    print("\n  comb SE = sqrt( mean(DeLong var)  +  seed_sd^2 / n_seeds ):"
          " test-sampling error (common to all seeds, does not shrink)"
          "\n            plus training/initialisation error (shrinks with seeds)."
          "\n  '*' = combined 95% CI excludes zero."
          "  '1-seed CIs' = seeds whose DeLong-only CI excludes zero.")

    block('d_external', 'T1.4  EXTERNAL AUC')
    block('d_fair_gap', 'T1.4  FAIRNESS GAP (young - elderly)', 'negative = smaller disparity')
    block('d_gen_gap',  'T1.4  GENERALISATION GAP', 'negative = smaller gap')
    block('d_young',    'T1.5  YOUNG (18-39) AUC')
    block('d_mid',      'T1.5  MID (40-59) AUC')
    block('d_elderly',  'T1.5  ELDERLY (60+) AUC')
    block('d_race_gap', 'T2.1  RACE GAP (White - Black AUC)',
          'negative = smaller disparity')
    for r in RACE_REPORTED:
        block(f'd_race_{r}', f'T2.1  {r.upper()} AUC')

    out = os.path.join(RESULTS_DIR, 'delong_deltas.json')
    with open(out, 'w') as f:
        json.dump({
            'method': 'DeLong (1988) via Sun & Xu (2014) fast algorithm; '
                      '95% CI = estimate +/- 1.96*SE',
            'reference': ref,
            'seeds': seeds,
            'n_brfss': int(len(y_b)),
            'subgroup_n': {k: int(v.sum()) for k, v in masks.items()},
            'subgroup_prevalence': {k: round(float(y_b[v].mean()), 5)
                                    for k, v in masks.items()},
            'absolute_per_seed': absolutes,
            'deltas_per_seed': per_seed,
            'across_seeds': seed_tests,
        }, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
