"""
60_nhanes_label_variants.py — same rows, BRFSS-style label; plus an internal hold-out
====================================================================================
Reviewer cqFk: NHANES labels diabetes from labs OR self-report, BRFSS from
self-report only, so the external divergence could be the outcome mismatch.
This script re-derives the 01_partition_data.py cohort row-for-row (asserted
against the saved node files) and writes a second copy of every training file in
which DIABETES is the self-report label only (DIQ010 == 1; lab-only cases become
non-cases, exactly as BRFSS DIABETE4 treats undiagnosed diabetes).

It also writes the 3,744 NHANES respondents outside the node caps — never seen
by the federated models or the composition-matched control — as an internal
hold-out evaluated under the label the models were trained on.

Outputs: data/nhanes_holdout.csv, data_sr/{node_*,centralised_full,nhanes_holdout}.csv
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import DATA_DIR, PROJECT_ROOT, SEED

FEATS = ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'BMXBMI', 'SMOKING', 'PHYS_ACTIVITY',
         'HEART_ATTACK', 'STROKE']
NODE_FILES = ['node_a_young_urban.csv', 'node_b_elderly_rural.csv', 'node_c_mixed_metro.csv']
SR_DIR = PROJECT_ROOT / 'data_sr'


def rebuild():
    """Faithful copy of 01_partition_data.py, keeping DIQ010 alongside."""
    df = pd.read_csv(DATA_DIR / 'nhanes_merged.csv')
    df['DIABETES'] = np.nan
    df.loc[df['LBXGLU'] < 126, 'DIABETES'] = 0
    df.loc[df['LBXGH'] < 6.5, 'DIABETES'] = 0
    df.loc[df['DIQ010'] == 2, 'DIABETES'] = 0
    df.loc[df['LBXGLU'] >= 126, 'DIABETES'] = 1
    df.loc[df['LBXGH'] >= 6.5, 'DIABETES'] = 1
    df.loc[df['DIQ010'] == 1, 'DIABETES'] = 1
    df = df[df['RIDAGEYR'] >= 18].copy()
    df = df.dropna(subset=['DIABETES'])
    df['DIABETES'] = df['DIABETES'].astype(int)
    df['SMOKING'] = 0
    df.loc[df['SMQ020'] == 1, 'SMOKING'] = 1
    df.loc[(df['SMQ020'] == 1) & (df['SMQ040'].isin([1, 2])), 'SMOKING'] = 2
    pa = ['PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665']
    df['PHYS_ACTIVITY'] = (df[pa] == 1).any(axis=1).astype(int)
    df['HEART_ATTACK'] = df['MCQ160E'].map({1: 1, 2: 0}).fillna(0).astype(int)
    df['STROKE'] = df['MCQ160F'].map({1: 1, 2: 0}).fillna(0).astype(int)
    c = df[FEATS + ['DIABETES', 'DIQ010']].copy()
    c['BMXBMI'] = c['BMXBMI'].fillna(c['BMXBMI'].median())
    c['RIAGENDR'] = c['RIAGENDR'].fillna(c['RIAGENDR'].mode()[0])
    c['RIDRETH3'] = c['RIDRETH3'].fillna(c['RIDRETH3'].mode()[0])
    c = c.reset_index(drop=True)

    pool_a = c[(c['RIDAGEYR'] < 45) | c['RIDRETH3'].isin([2.0, 4.0])]
    node_a = pool_a.sample(n=min(4500, len(pool_a)), random_state=SEED)
    pool_b = c[(c['RIDAGEYR'] >= 55) & c['RIDRETH3'].isin([1.0, 3.0])]
    node_b = pool_b.sample(n=min(3500, len(pool_b)), random_state=SEED)
    used = set(node_a.index) | set(node_b.index)
    pool_c = c[~c.index.isin(used)]
    node_c = pool_c.sample(n=min(4000, len(pool_c)), random_state=SEED)
    return c, [node_a, node_b, node_c]


def main():
    c, nodes = rebuild()
    cols = FEATS + ['DIABETES']
    for node, fn in zip(nodes, NODE_FILES):
        saved = pd.read_csv(DATA_DIR / fn)
        assert np.allclose(saved[cols].values, node[cols].values), fn
    assert np.allclose(pd.read_csv(DATA_DIR / 'centralised_full.csv')[cols].values, c[cols].values)
    print('  rebuilt cohort matches saved node + pool files row-for-row')

    used = set().union(*[set(n.index) for n in nodes])
    hold = c[~c.index.isin(used)]
    print(f'  hold-out n={len(hold):,}  race counts {hold.RIDRETH3.value_counts().sort_index().to_dict()}'
          f'  prevalence {hold.DIABETES.mean():.3f}')

    sr = lambda d: d.assign(DIABETES=(d['DIQ010'] == 1).astype(int))[cols]
    hold[cols].to_csv(DATA_DIR / 'nhanes_holdout.csv', index=False)
    os.makedirs(SR_DIR, exist_ok=True)
    for node, fn in zip(nodes, NODE_FILES):
        sr(node).to_csv(SR_DIR / fn, index=False)
    sr(c).to_csv(SR_DIR / 'centralised_full.csv', index=False)
    sr(hold).to_csv(SR_DIR / 'nhanes_holdout.csv', index=False)
    fed = pd.concat(nodes)
    print(f'  federated cohort prevalence: composite {fed.DIABETES.mean():.3f} -> '
          f'self-report {(fed.DIQ010 == 1).mean():.3f}; lab-only cases relabelled: '
          f'{int(((fed.DIABETES == 1) & (fed.DIQ010 != 1)).sum())}')
    print(f'  wrote {DATA_DIR / "nhanes_holdout.csv"} and {SR_DIR}')


if __name__ == '__main__':
    main()
