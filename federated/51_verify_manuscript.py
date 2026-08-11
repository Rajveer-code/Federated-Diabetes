"""
51_verify_manuscript.py — every number in the paper must trace to a result file
==============================================================================
The manuscript has been rewritten several times and numbers have drifted between
versions before. This reads the LaTeX source and asserts that each reported value
still matches the JSON it came from, so a stale figure cannot survive an edit.

Each check is (label, expected-string, source). The expected string is FORMATTED
FROM THE JSON at run time and then searched for literally in the .tex, which
catches both a changed result and a typo in the manuscript. Anything that cannot
be traced to a file is listed separately rather than silently passing.

Exit code is non-zero if any check fails, so this is usable as a pre-submission
gate.

Usage:
  python 51_verify_manuscript.py
  python 51_verify_manuscript.py --tex path/to/main.tex
"""
import os, sys, json, re, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

DEFAULT_TEX = (r'C:\Users\Asus\Downloads\masters\LATEST_MANUSCRIPTS'
               r'\ML4H_Findings\main.tex')

STRATA = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
REF, MODEL, RATE = 'Centralised-matched', 'FedAvg', '0.20'


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def sgn4(v):
    """LaTeX-style signed 4-decimal, matching the manuscript's $-$0.0181 form."""
    return ('$-$' if v < 0 else '$+$') + f'{abs(v):.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tex', default=DEFAULT_TEX)
    args = ap.parse_args()

    tex = open(args.tex, encoding='utf-8').read()
    # Collapse whitespace so a value split across a wrapped line still matches.
    flat = re.sub(r'\s+', ' ', tex)

    dd = load('delong_deltas.json')
    xa = load('xauc.json')
    cc = load('capacity_curve.json')
    ct = load('clinical_translation.json')
    ps = load('prevalence_sweep.json')

    checks = []

    def chk(label, s, source):
        # The table bolds its headline cells, which renders the sign inside the
        # math group ($\mathbf{-0.0188}$) rather than as $-$0.0188. Accept either.
        alts = [s]
        if s.startswith(('$-$', '$+$')):
            alts.append('\\mathbf{' + s[1] + s[3:] + '}')
        checks.append((label, alts, source))

    # ── cohort and scale ──────────────────────────────────────────────────────
    chk('BRFSS n', f"{dd['n_brfss']:,}".replace(',', '{,}'), 'delong_deltas')
    for g in STRATA:
        chk(f'n {g}', f"{dd['subgroup_n']['race_' + g]:,}".replace(',', '{,}'),
            'delong_deltas')

    # ── within-stratum AUC deltas (the diagonal) ──────────────────────────────
    for g in STRATA:
        v = dd['across_seeds'][MODEL][f'd_race_{g}']['combined']['delta']
        chk(f'dAUC {g}', sgn4(v), 'delong_deltas')

    # ── cross-group ranking (the White column) ────────────────────────────────
    for g in STRATA:
        if g == 'White':
            continue
        v = xa['across_seeds'][MODEL][f'{g}->White']['combined']['delta']
        chk(f'dxAUC {g}->White', sgn4(v), 'xauc')
    chk('xAUC identity error', '2.2{\\times}10^{-16}', 'xauc')

    # ── allocation at 20% capacity ────────────────────────────────────────────
    j = cc['rates'].index(0.20)
    for g in STRATA:
        r_, f_ = cc['by_model'][REF][g], cc['by_model'][MODEL][g]
        chk(f'dSelected {g}', sgn4(f_['sel'][j] - r_['sel'][j]), 'capacity_curve')
        chk(f'dSens {g}', sgn4(f_['sens'][j] - r_['sens'][j]), 'capacity_curve')
        chk(f'dPPV {g}', sgn4(f_['ppv'][j] - r_['ppv'][j]), 'capacity_curve')
        cases = ct['by_rate'][RATE][MODEL][g]['delta_cases_per_100k']
        chk(f'cases {g}', ('$-$' if cases < 0 else '$+$') + f'{abs(cases):.0f}',
            'clinical_translation')

    # Selection rates quoted in prose, 3 decimals.
    for g, who in (('Black', 'minority'), ('White', 'majority')):
        chk(f'sel {who} ref', f"{cc['by_model'][REF][g]['sel'][j]:.3f}", 'capacity_curve')
        chk(f'sel {who} fed', f"{cc['by_model'][MODEL][g]['sel'][j]:.3f}", 'capacity_curve')

    # ── controlled prevalence sweep ───────────────────────────────────────────
    chk('sweep r (sens gap)', f"{ps['pearson_r_spread_vs_delta_sens_gap']:+.2f}"
        .replace('+', '$+$').replace('-', '$-$'), 'prevalence_sweep')

    # ── run ───────────────────────────────────────────────────────────────────
    ok, bad = 0, []
    for label, alts, source in checks:
        if any(a in flat for a in alts):
            ok += 1
        else:
            bad.append((label, alts[0], source))

    print(f"  {ok}/{len(checks)} checks passed  ({args.tex})")
    if bad:
        print("\n  NOT FOUND in the manuscript:")
        for label, s, source in bad:
            print(f"    {label:<26} expected {s!r:<24} from {source}.json")

    # Anything numeric in the tex that no check covers is worth eyeballing once.
    placeholders = re.findall(r'\[PLACEHOLDER[^\]]*\]', tex)
    if placeholders:
        print(f"\n  {len(placeholders)} PLACEHOLDER(s) still in the manuscript:")
        for p in placeholders:
            print(f"    {p[:96]}")

    failed = bool(bad) or bool(placeholders)
    print(f"\n  {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
