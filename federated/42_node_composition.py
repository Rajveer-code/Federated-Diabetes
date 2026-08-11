"""
42_node_composition.py — racial composition of the federated partition
======================================================================
The nodes are defined on age and urbanicity, but that split separates the cohort
racially as a side effect, and that separation is the proposed mechanism for the
per-stratum result. The manuscript quotes these shares, so they need a script
that regenerates them rather than a number typed once.

Output: results/node_composition.json

Usage:  python 42_node_composition.py
"""
import os, sys, json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import NODE_PATHS, NODE_NAMES, CENTRALISED_PATH, RESULTS_DIR, RACE_COL

# NHANES RIDRETH3 codes.
LABELS = {1.0: 'Mexican American', 2.0: 'Other Hispanic', 3.0: 'NH White',
          4.0: 'NH Black', 6.0: 'NH Asian', 7.0: 'Other/Multi'}


def shares(df):
    vc = df[RACE_COL].value_counts(normalize=True)
    return {lab: round(100 * float(vc.get(code, 0.0)), 2) for code, lab in LABELS.items()}


def main():
    out = {'note': 'Percentages of each node by NHANES RIDRETH3. The partition is '
                   'defined on age and urbanicity; the racial separation is a '
                   'side effect of it.',
           'nodes': {}}

    for name, path in zip(NODE_NAMES, NODE_PATHS):
        df = pd.read_csv(path)
        out['nodes'][name] = {'n': int(len(df)), 'race_pct': shares(df)}

    pool = pd.read_csv(CENTRALISED_PATH)
    out['pooled'] = {'n': int(len(pool)), 'race_pct': shares(pool)}

    # The three figures quoted in the manuscript.
    elderly = next(k for k in out['nodes'] if 'lder' in k)
    young   = next(k for k in out['nodes'] if 'oung' in k)
    absent  = [lab for lab, v in out['nodes'][elderly]['race_pct'].items() if v == 0.0]
    out['manuscript_claims'] = {
        'elderly_rural_NH_White_pct':  out['nodes'][elderly]['race_pct']['NH White'],
        'elderly_rural_absent_strata': absent,
        'young_urban_NH_Black_pct':    out['nodes'][young]['race_pct']['NH Black'],
        'pooled_NH_White_pct':         out['pooled']['race_pct']['NH White'],
    }

    hdr = f"{'node':<24}{'n':>7}" + "".join(f"{l:>18}" for l in LABELS.values())
    print(hdr); print('-' * len(hdr))
    for name, d in out['nodes'].items():
        print(f"{name:<24}{d['n']:>7,}" + "".join(f"{d['race_pct'][l]:>17.1f}%"
                                                  for l in LABELS.values()))
    print(f"{'POOLED':<24}{out['pooled']['n']:>7,}"
          + "".join(f"{out['pooled']['race_pct'][l]:>17.1f}%" for l in LABELS.values()))

    print("\nQuoted in the manuscript:")
    for k, v in out['manuscript_claims'].items():
        print(f"  {k:<30} {v}")

    path = os.path.join(RESULTS_DIR, 'node_composition.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == '__main__':
    main()
