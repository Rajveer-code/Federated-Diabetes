"""
compile_all_results.py — Summarise all new experiment results
=============================================================
Reads the 4 JSON output files and prints a formatted manuscript-ready summary.
Run this after all 4 experiment scripts have completed.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import RESULTS_DIR

def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — run the corresponding experiment first.")
        return None
    with open(path) as f:
        return json.load(f)

def fmt(val, decimals=3):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"

def fmt_ci(auc, lo, hi, decimals=3):
    return f"{fmt(auc, decimals)} [{fmt(lo, decimals)}–{fmt(hi, decimals)}]"


print("=" * 72)
print("  COMPILE ALL RESULTS — NEW EXPERIMENTS FOR JBI REVISION")
print("=" * 72)

# ── EXP 1: Centralised NN ────────────────────────────────────────────────────
e1 = load_json('exp1_central_nn.json')
print("\n" + "=" * 72)
print("  EXPERIMENT 1: CENTRALISED DiabetesNet BASELINE")
print("=" * 72)
if e1:
    i = e1['internal']
    x = e1['external']
    print(f"  Internal AUC:     {fmt_ci(i['auc'], i['ci_low'], i['ci_high'])}")
    print(f"  Internal Brier:   {fmt(i['brier'])}  F1={fmt(i['f1'])}")
    print(f"  Internal Sens:    {fmt(i['sensitivity'])}  Spec={fmt(i['specificity'])}")
    print()
    print(f"  External AUC:     {fmt_ci(x['auc'], x['ci_low'], x['ci_high'])}")
    print(f"  External Brier:   {fmt(x['brier'])}  F1={fmt(x['f1'])}")
    print(f"  External Sens:    {fmt(x['sensitivity'])}  Spec={fmt(x['specificity'])}")
    print()
    subs = x.get('subgroup_auc', {})
    print(f"  Subgroup AUC (external BRFSS):")
    print(f"    Age 18-39:      {fmt(subs.get('age_18_39'))}")
    print(f"    Age 40-59:      {fmt(subs.get('age_40_59'))}")
    print(f"    Age 60+:        {fmt(subs.get('age_60_plus'))}")
    print(f"    Elderly gap:    {fmt(x.get('elderly_gap'))}")
    print()
    print(f"  Delta Int→Ext:    {fmt(e1['delta_int_to_ext'], 3)}")
    print()
    print("  ── Table row (paste into Table 2 / Table 3) ──")
    print(f"  | Centralised NN (DiabetesNet) | {fmt_ci(i['auc'], i['ci_low'], i['ci_high'])} "
          f"| {fmt(i['brier'])} | {fmt(i['f1'])} | {fmt(i['sensitivity'])} | {fmt(i['specificity'])} |")
    print(f"  | Centralised NN (DiabetesNet) | {fmt_ci(x['auc'], x['ci_low'], x['ci_high'])} "
          f"| {fmt(x['brier'])} | {fmt(x['f1'])} | {fmt(x['sensitivity'])} | {fmt(x['specificity'])} |")


# ── EXP 2: Node B Ablation ───────────────────────────────────────────────────
e2 = load_json('exp2_node_b_ablation.json')
print("\n" + "=" * 72)
print("  EXPERIMENT 2: NODE B ABLATION (FedAvg A+C only)")
print("=" * 72)
if e2:
    i2 = e2['internal']
    x2 = e2['external']
    c2 = e2['comparison']
    print(f"  Internal AUC:     {fmt_ci(i2['auc'], i2['ci_low'], i2['ci_high'])}")
    print(f"  External AUC:     {fmt_ci(x2['auc'], x2['ci_low'], x2['ci_high'])}")
    print()
    subs2 = x2.get('subgroup_auc', {})
    print(f"  Elderly AUC:      {fmt(subs2.get('age_60_plus'))}")
    print(f"  Young AUC:        {fmt(subs2.get('age_18_39'))}")
    print(f"  Elderly gap:      {fmt(x2.get('elderly_gap'))}")
    print()
    print(f"  ── Comparison to full FedAvg (A+B+C) ──")
    print(f"  Full FL external AUC:    {c2['full_fl_external_auc']}")
    print(f"  Ablation external AUC:   {fmt(c2['ablation_external_auc'])}")
    print(f"  Full FL elderly gap:     {c2['full_fl_elderly_gap']}")
    print(f"  Ablation elderly gap:    {fmt(c2['ablation_elderly_gap'])}")
    gap_inc = c2.get('gap_increase_pct')
    if gap_inc is not None:
        print(f"  Gap increase:            {gap_inc:+.1f}%")
    print()
    print("  ── Table 6 rows ──")
    print(f"  | FedAvg (Full: A+B+C)       | 0.757 | 0.669 | 0.722 | 0.054 |")
    print(f"  | FedAvg Ablation (A+C only)  | {fmt(x2['auc'])} | "
          f"{fmt(subs2.get('age_60_plus'))} | "
          f"{fmt(subs2.get('age_18_39'))} | "
          f"{fmt(x2.get('elderly_gap'))} |")
    print(f"  | Centralised XGBoost         | 0.700 | 0.587 | 0.656 | 0.069 |")


# ── EXP 3: FedProx Sensitivity ───────────────────────────────────────────────
e3 = load_json('exp3_fedprox_sensitivity.json')
print("\n" + "=" * 72)
print("  EXPERIMENT 3: FedProx μ SENSITIVITY")
print("=" * 72)
if e3:
    mu_map = [('mu_0_01', '0.01'), ('mu_0_05', '0.05'), ('mu_0_10_existing', '0.10')]
    print(f"  {'μ':<8} {'External AUC [95% CI]':<30} {'Elderly Gap'}")
    print("  " + "─" * 55)
    for key, label in mu_map:
        v = e3.get(key, {})
        if key == 'mu_0_10_existing':
            print(f"  {label:<8} 0.752 [0.751–0.753] (existing)     0.066 (existing)")
        else:
            ci_str = fmt_ci(v.get('external_auc'), v.get('ci_low'), v.get('ci_high'))
            gap    = v.get('elderly_gap')
            print(f"  {label:<8} {ci_str:<30} {fmt(gap)}")
    print()
    print("  ── Table 7 rows ──")
    for key, label in mu_map:
        v = e3.get(key, {})
        if key == 'mu_0_10_existing':
            print(f"  | {label} | 0.752 [0.751–0.753] | 0.066 |")
        else:
            ci_str = fmt_ci(v.get('external_auc'), v.get('ci_low'), v.get('ci_high'))
            print(f"  | {label} | {ci_str} | {fmt(v.get('elderly_gap'))} |")


# ── EXP 4: Calibration ───────────────────────────────────────────────────────
e4 = load_json('exp4_calibration.json')
print("\n" + "=" * 72)
print("  EXPERIMENT 4: CALIBRATION (FedAvg on BRFSS)")
print("=" * 72)
if e4:
    print(f"  ECE (10-bin):    {fmt(e4['ece_10bins'], 4)}")
    print(f"  Brier score:     {fmt(e4['brier_score'], 4)}")
    print(f"  Reliability:     {fmt(e4['brier_reliability'], 4)}")
    print(f"  Resolution:      {fmt(e4['brier_resolution'],  4)}")
    print(f"  Uncertainty:     {fmt(e4['brier_uncertainty'],  4)}")
    print(f"  n_samples:       {e4['n_samples']:,}")
    ece = e4['ece_10bins']
    qual = "good" if ece < 0.05 else ("moderate" if ece < 0.10 else "poor")
    print(f"  Calibration quality: {qual} (ECE {ece:.3f})")


# ── Summary comparison table ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY TABLE: ALL MODELS — EXTERNAL BRFSS AUC")
print("=" * 72)
models = [
    ("FedAvg (A+B+C)",         "0.757", "[0.756–0.758]", "0.054"),
    ("FedProx μ=0.1",           "0.752", "[0.751–0.753]", "0.066"),
    ("FedNova",                 "0.744", "—",             "0.064"),
    ("Centralised XGBoost",     "0.700", "—",             "0.069"),
]
if e1:
    x = e1['external']
    models.insert(3, (
        "Centralised NN",
        fmt(x['auc']),
        f"[{fmt(x['ci_low'])}–{fmt(x['ci_high'])}]",
        fmt(x.get('elderly_gap')),
    ))
if e2:
    x2 = e2['external']
    subs2 = x2.get('subgroup_auc', {})
    models.insert(1, (
        "FedAvg Ablation (A+C)",
        fmt(x2['auc']),
        f"[{fmt(x2['ci_low'])}–{fmt(x2['ci_high'])}]",
        fmt(x2.get('elderly_gap')),
    ))

print(f"\n  {'Model':<28} {'AUC':>7} {'95% CI':>16} {'Elderly Gap':>12}")
print("  " + "─" * 65)
for name, auc, ci, gap in models:
    print(f"  {name:<28} {auc:>7} {ci:>16} {gap:>12}")

print("\n" + "=" * 72)
print("  All results compiled. Update manuscript with numbers above.")
print("=" * 72)
