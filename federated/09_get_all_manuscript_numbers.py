"""
09_get_all_manuscript_numbers.py
==================================
Run this LAST. Prints every number needed for the manuscript
in copy-paste-ready format. Takes about 30 seconds.

Run order before this script:
  1. 07_external_validation.py  (with pred_xgb_external.npy save added)
  2. 07_statistical_analysis.py (with XGBoost added to external_models)
  3. 05_fairness_analysis.py    (with FedAvg/FedNova EOD block added)
  4. 08_calibration_analysis.py (new script)
  THEN run this script.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

print("\n" + "=" * 70)
print("  MANUSCRIPT NUMBER EXTRACTION")
print("  Copy these into the LaTeX exactly as shown")
print("=" * 70)

def load_json(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING] {fname} — run the prerequisite script first")
        return {}
    with open(path) as f:
        return json.load(f)

ci       = load_json('auc_confidence_intervals.json')
ext_val  = load_json('external_validation.json')
calib    = load_json('calibration_extended.json')
eod_t6   = load_json('table6_eod_values.json')

# ── SECTION 1: CI Table ────────────────────────────────────────────────────
print("\n\n[1] CONFIDENCE INTERVALS — for Tables 2 and 3")
print("-" * 70)

int_ci = ci.get('internal', {})
ext_ci = ci.get('external', {})

print("\n  INTERNAL AUC (Table 2 — stratified bootstrap, N=2000):")
for name in ['FedAvg', 'FedProx', 'FedNova', 'XGBoost']:
    r = int_ci.get(name)
    if r:
        print(f"    {name:<12}: AUC={r['auc']:.3f}  "
              f"95% CI: {r['lower']:.3f}--{r['upper']:.3f}")
    else:
        print(f"    {name:<12}: [NOT FOUND — check script run]")

print("\n  EXTERNAL AUC (Table 3 — DeLong):")
for name in ['FedAvg', 'FedProx', 'FedNova', 'XGBoost']:
    r = ext_ci.get(name)
    if r:
        print(f"    {name:<12}: AUC={r['auc']:.3f}  "
              f"95% CI: {r['lower']:.3f}--{r['upper']:.3f}  "
              f"SE={r['se']:.5f}")
    else:
        print(f"    {name:<12}: [NOT FOUND — check script run]")

pt = ci.get('paired_test_fedprox_vs_fedavg_external', {})
if pt:
    print(f"\n  PAIRED DELONG TEST (FedAvg vs FedProx external):")
    print(f"    delta AUC = {pt.get('delta', 'N/A'):.4f}")
    print(f"    z         = {pt.get('z', 'N/A'):.3f}")
    print(f"    p-value   = {pt.get('p_value', 'N/A'):.6f}")

# ── SECTION 2: Calibration ─────────────────────────────────────────────────
print("\n\n[2] CALIBRATION — for Section 4.4 and Table 4")
print("-" * 70)
if calib:
    print(f"  ECE (equal-width,  10 bins): {calib.get('ece_equal_width_10', 'N/A'):.4f}"
          f"  ← already in paper")
    print(f"  ECE (equal-freq,   10 bins): {calib.get('ece_equal_freq_10',  'N/A'):.4f}"
          f"  ← ADD to §4.4 text")
    print(f"  Calibration slope:           {calib.get('calibration_slope',   'N/A'):.4f}"
          f"  (1.0 = perfect, <1.0 = overconfident)")
    print(f"  Calibration intercept:       {calib.get('calibration_intercept','N/A'):.4f}")
    print(f"  Brier score:                 {calib.get('brier', 'N/A'):.4f}")
else:
    print("  [MISSING] Run 08_calibration_analysis.py first")

# ── SECTION 3: EOD Table 6 ─────────────────────────────────────────────────
print("\n\n[3] EOD VALUES — for Table 6 (FedAvg and FedNova rows)")
print("-" * 70)
print("  These go into the [compute] placeholder rows in Table 6.")
print()

for model_name in ['FedAvg', 'FedNova']:
    data = eod_t6.get(model_name)
    if not data:
        print(f"  {model_name}: [NOT FOUND] — check 05_fairness_analysis.py ran correctly")
        continue

    g = data.get('global', {})
    s = data.get('subgroup_specific', {})

    global_thresh = g.get('threshold', 'N/A')
    young_tpr_g   = g.get(0, {}).get('tpr', 'N/A') if isinstance(g.get(0), dict) else 'N/A'
    young_fpr_g   = g.get(0, {}).get('fpr', 'N/A') if isinstance(g.get(0), dict) else 'N/A'
    eld_tpr_g     = g.get(1, {}).get('tpr', 'N/A') if isinstance(g.get(1), dict) else 'N/A'
    eld_fpr_g     = g.get(1, {}).get('fpr', 'N/A') if isinstance(g.get(1), dict) else 'N/A'
    eod_g         = g.get('eod', 'N/A')

    young_thresh_s = s.get(0, {}).get('threshold', 'N/A') if isinstance(s.get(0), dict) else 'N/A'
    eld_thresh_s   = s.get(1, {}).get('threshold', 'N/A') if isinstance(s.get(1), dict) else 'N/A'
    young_tpr_s    = s.get(0, {}).get('tpr', 'N/A') if isinstance(s.get(0), dict) else 'N/A'
    young_fpr_s    = s.get(0, {}).get('fpr', 'N/A') if isinstance(s.get(0), dict) else 'N/A'
    eld_tpr_s      = s.get(1, {}).get('tpr', 'N/A') if isinstance(s.get(1), dict) else 'N/A'
    eld_fpr_s      = s.get(1, {}).get('fpr', 'N/A') if isinstance(s.get(1), dict) else 'N/A'
    eod_s          = s.get('eod', 'N/A')

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    print(f"  {model_name} — Global Youden (threshold = {fmt(global_thresh)}):")
    print(f"    Young TPR={fmt(young_tpr_g)}  Young FPR={fmt(young_fpr_g)}  "
          f"Elderly TPR={fmt(eld_tpr_g)}  Elderly FPR={fmt(eld_fpr_g)}  "
          f"EOD={fmt(eod_g)}")
    print(f"  {model_name} — Subgroup-specific "
          f"(young:{fmt(young_thresh_s)}, elderly:{fmt(eld_thresh_s)}):")
    print(f"    Young TPR={fmt(young_tpr_s)}  Young FPR={fmt(young_fpr_s)}  "
          f"Elderly TPR={fmt(eld_tpr_s)}  Elderly FPR={fmt(eld_fpr_s)}  "
          f"EOD={fmt(eod_s)}")
    print()

# ── SECTION 4: PPV check ───────────────────────────────────────────────────
print("\n[4] PPV ARITHMETIC CHECK — for Section 5.5")
print("-" * 70)
# FedAvg BRFSS: sensitivity=0.768, specificity=0.607, prevalence=0.133
prev = 0.133
sens = 0.768
spec = 0.607
tp_rate  = sens * prev
fp_rate  = (1 - spec) * (1 - prev)
ppv = tp_rate / (tp_rate + fp_rate)
npv_tp = (1 - prev)
npv_calc = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev)
print(f"  Using: sens={sens}, spec={spec}, prev={prev}")
print(f"  PPV = {ppv:.4f}  ({ppv*100:.1f}%)")
print(f"  NPV = {npv_calc:.4f}  ({npv_calc*100:.1f}%)")
print(f"  → Use ~{round(ppv*100):.0f}% and ~{round(npv_calc*100):.0f}% in manuscript §5.5")

print("\n" + "=" * 70)
print("  Done. Copy all numbers above into the LaTeX.")
print("=" * 70)