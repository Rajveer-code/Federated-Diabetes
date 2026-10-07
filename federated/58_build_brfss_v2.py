"""
58_build_brfss_v2.py — rebuild the BRFSS external cohort from the raw XPT files
==============================================================================
Three defects were found in brfss_final.csv / load_brfss_clean during the
ML4H rebuttal audit (2026-10-07), each verified against the raw LLCP files:

  1. Sex swapped. The processing notebook maps SEXVAR {1:Male, 2:Female} to
     Gender {0, 1}; load_brfss_clean then mapped {1 -> 1, 0 -> 2}, i.e. every
     BRFSS woman entered the NHANES-trained model as RIAGENDR=1 (male).
     Fixed in data_utils (0 -> 1, 1 -> 2); this file keeps the 0/1 Gender coding.
  2. Smoking scale. NHANES SMOKING is 0=never, 1=former, 2=current. The notebook
     collapsed _SMOKER3 to ever/never, so BRFSS current smokers entered as
     1 (former). Rebuilt here on the NHANES scale from _SMOKER3.
  3. Race lost for 2022. BRFSS 2022 renamed _RACE to _RACE1; the notebook kept
     only _RACE, so all 445k 2022 respondents had race missing. That, not
     non-response, was the "36.1% unknown race" in the manuscript.

Row order is identical to brfss_final.csv (2020, 2021, 2022 concatenated, no
filtering), which is asserted below on every column the fix does not touch.
Survey-design columns are carried along for the weighted sensitivity analysis.

Output: data/brfss_final_v2.csv
Usage:  python 58_build_brfss_v2.py
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import DATA_DIR

RAW_DIR = os.environ.get('BRFSS_RAW_DIR', r'C:\diabetes_prediction_project\data\01_raw\BRFSS')
OLD_PATH = os.environ.get('BRFSS_OLD_PATH', r'C:\diabetes_prediction_project\data\03_processed\brfss_final.csv')
OUT_PATH = os.path.join(DATA_DIR, 'brfss_final_v2.csv')
YEARS = ['2020', '2021', '2022']
RACE_VAR = {'2020': '_RACE', '2021': '_RACE', '2022': '_RACE1'}
BASE = ['DIABETE4', '_AGEG5YR', 'SEXVAR', '_BMI5', 'GENHLTH', '_SMOKER3',
        '_TOTINDA', 'CVDINFR4', 'CVDSTRK3']
DESIGN = ['_LLCPWT', '_STSTR', '_PSU', '_STATE', '_URBSTAT', '_METSTAT']
AGE_MAP = {1: 21.5, 2: 27.5, 3: 32.5, 4: 37.5, 5: 42.5, 6: 47.5, 7: 52.5,
           8: 57.5, 9: 62.5, 10: 67.5, 11: 72.5, 12: 77.5, 13: 80.0}


def read_year(year):
    path = os.path.join(RAW_DIR, year, f'LLCP{year}.XPT')
    keep = BASE + [RACE_VAR[year]] + DESIGN
    parts = []
    for chunk in pd.read_sas(path, format='xport', chunksize=100_000, encoding='latin1'):
        parts.append(chunk[keep].copy())
    df = pd.concat(parts, ignore_index=True).rename(columns={RACE_VAR[year]: 'RACE_SRC'})
    df['YEAR'] = int(year)
    print(f'  {year}: {len(df):,} rows')
    return df


def harmonise(raw):
    out = pd.DataFrame({
        'Diabetes_Outcome': raw['DIABETE4'].replace({1: 1, 2: 0, 3: 0, 4: 0, 7: np.nan, 9: np.nan}),
        'Age': raw['_AGEG5YR'].map(AGE_MAP),
        'Gender': raw['SEXVAR'].replace({1: 0, 2: 1}),          # 0=male, 1=female
        'Race_Ethnicity': raw['RACE_SRC'],
        'BMI': raw['_BMI5'] / 100.0,
        'General_Health': raw['GENHLTH'].replace({7: np.nan, 9: np.nan}),
        # NHANES scale: 0=never, 1=former, 2=current
        'Smoking_Status': raw['_SMOKER3'].map({1: 2, 2: 2, 3: 1, 4: 0}),
    })
    for src, dst in [('_TOTINDA', 'Physical_Activity'), ('CVDINFR4', 'History_Heart_Attack'),
                     ('CVDSTRK3', 'History_Stroke')]:
        out[dst] = raw[src].replace({1: 1, 2: 0, 7: np.nan, 9: np.nan})
    for c in DESIGN + ['YEAR']:
        out[c] = raw[c].values
    return out


def main():
    raw = pd.concat([read_year(y) for y in YEARS], ignore_index=True)
    new = harmonise(raw)

    old = pd.read_csv(OLD_PATH)
    assert len(old) == len(new), (len(old), len(new))
    same = ['Diabetes_Outcome', 'Age', 'Gender', 'BMI', 'General_Health',
            'Physical_Activity', 'History_Heart_Attack', 'History_Stroke']
    for c in same:
        a, b = old[c].to_numpy(float), new[c].to_numpy(float)
        ok = np.isclose(a, b, equal_nan=True).all()
        print(f'  alignment {c:22s} {"OK" if ok else "MISMATCH"}')
        assert ok, c
    pre22 = (new['YEAR'] < 2022).to_numpy()
    assert np.array_equal(old['Race_Ethnicity'].to_numpy(float)[pre22],
                          new['Race_Ethnicity'].to_numpy(float)[pre22], equal_nan=True)
    ever_old = old['Smoking_Status'].to_numpy(float)
    ever_new = (new['Smoking_Status'] > 0).astype(float).where(new['Smoking_Status'].notna())
    assert np.isclose(ever_old, ever_new.to_numpy(float), equal_nan=True).all()
    print('  alignment race 2020-21 / smoking ever-never  OK')

    print('\n  Race missing (NaN or 9): old %.3f -> new %.3f' % (
        (old['Race_Ethnicity'].isna() | (old['Race_Ethnicity'] == 9)).mean(),
        (new['Race_Ethnicity'].isna() | (new['Race_Ethnicity'] == 9)).mean()))
    print('  Race distribution by year (share):')
    print(pd.crosstab(new['YEAR'], new['Race_Ethnicity'].fillna(-1), normalize='index').round(3))
    print('  Smoking (new):', new['Smoking_Status'].value_counts(dropna=False).to_dict())

    new.to_csv(OUT_PATH, index=False)
    print(f'\n  Saved {OUT_PATH}  ({len(new):,} rows)')


if __name__ == '__main__':
    main()
