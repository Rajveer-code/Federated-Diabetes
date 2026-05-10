"""
exp4_calibration.py — Calibration analysis for FedAvg on BRFSS
================================================================
Computes ECE (10 equal-width bins), Brier score decomposition,
and generates a publication-quality reliability diagram.

Produces:
  results/exp4_calibration.json
  results/figures/reliability_diagram_fedavg.png
"""

import os, sys, json, random, warnings
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    RESULTS_DIR, MODELS_DIR, GLOBAL_SCALER_PATH,
    FEATURE_COLS, TARGET_COL, RANDOM_SEED,
)
from nn_model import DiabetesNet, get_device

BRFSS_PATH = r"C:\diabetes_prediction_project\data\03_processed\brfss_final.csv"
BRFSS_COL_MAP = {
    'Age': 'RIDAGEYR', 'Gender': 'RIAGENDR', 'Race_Ethnicity': 'RIDRETH3',
    'BMI': 'BMXBMI', 'Smoking_Status': 'SMOKING', 'Physical_Activity': 'PHYS_ACTIVITY',
    'History_Heart_Attack': 'HEART_ATTACK', 'History_Stroke': 'STROKE',
    'Diabetes_Outcome': 'DIABETES',
}

N_BINS  = 10
DEVICE  = get_device()
FIGURES = os.path.join(RESULTS_DIR, 'figures')

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("=" * 65)
print("  EXP 4 — CALIBRATION ANALYSIS (FedAvg on BRFSS)")
print(f"  ECE bins={N_BINS}")
print("=" * 65)

os.makedirs(FIGURES, exist_ok=True)


# ── Load FedAvg model ─────────────────────────────────────────────────────────
print("\n[1/5] Loading FedAvg model...")
fedavg_path = os.path.join(MODELS_DIR, 'fedavg_weights.pt')
if not os.path.exists(fedavg_path):
    print(f"  ERROR: {fedavg_path} not found. Run 03_federated_simulation.py first.")
    sys.exit(1)
model = DiabetesNet().to(DEVICE)
model.load_state_dict(torch.load(fedavg_path, map_location=DEVICE))
model.eval()
print(f"  Loaded: {fedavg_path}")


# ── Load and preprocess BRFSS ─────────────────────────────────────────────────
print("\n[2/5] Loading BRFSS data...")
df_brfss = pd.read_csv(BRFSS_PATH)
df_brfss = df_brfss.rename(columns=BRFSS_COL_MAP)
if df_brfss['RIAGENDR'].max() <= 1.0:
    df_brfss['RIAGENDR'] = df_brfss['RIAGENDR'].map({1.0: 1.0, 0.0: 2.0})
df_brfss = df_brfss.dropna(subset=['DIABETES'])
df_brfss['DIABETES'] = df_brfss['DIABETES'].astype(int)
for col in ['BMXBMI', 'RIDAGEYR']:
    df_brfss[col] = df_brfss[col].fillna(df_brfss[col].median())
for col in ['RIAGENDR', 'RIDRETH3', 'SMOKING', 'PHYS_ACTIVITY', 'HEART_ATTACK', 'STROKE']:
    df_brfss[col] = df_brfss[col].fillna(df_brfss[col].mode()[0])
df_brfss = df_brfss.reset_index(drop=True)

X_brfss   = df_brfss[FEATURE_COLS].values.astype(np.float32)
y_brfss   = df_brfss['DIABETES'].values.astype(np.float32)
scaler    = joblib.load(GLOBAL_SCALER_PATH)
X_brfss_sc = scaler.transform(X_brfss).astype(np.float32)
N_SAMPLES  = len(df_brfss)
print(f"  n={N_SAMPLES:,}  prevalence={y_brfss.mean():.1%}")


# ── Batched inference ─────────────────────────────────────────────────────────
print("\n[3/5] Running batched inference...")
chunk = 50_000
probs_list = []
with torch.no_grad():
    for i in range(0, len(X_brfss_sc), chunk):
        X_c = torch.FloatTensor(X_brfss_sc[i:i+chunk]).to(DEVICE)
        probs_list.append(torch.sigmoid(model(X_c)).cpu().numpy())
        if (i // chunk) % 5 == 0:
            print(f"  Processed {min(i+chunk, N_SAMPLES):,}/{N_SAMPLES:,}")
y_prob = np.concatenate(probs_list)
print(f"  Inference complete: {len(y_prob):,} predictions")


# ── ECE and bin data ──────────────────────────────────────────────────────────
print("\n[4/5] Computing calibration metrics...")
bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
bin_lowers = bin_edges[:-1]
bin_uppers = bin_edges[1:]
bin_data   = []
ece_num    = 0.0

for b, (lo, hi) in enumerate(zip(bin_lowers, bin_uppers)):
    if b == N_BINS - 1:
        mask = (y_prob >= lo) & (y_prob <= hi)
    else:
        mask = (y_prob >= lo) & (y_prob < hi)
    n_k     = int(mask.sum())
    if n_k == 0:
        bin_data.append({'bin': b+1, 'mean_pred': float((lo+hi)/2),
                         'frac_pos': 0.0, 'count': 0})
        continue
    mean_pred = float(y_prob[mask].mean())
    frac_pos  = float(y_brfss[mask].mean())
    ece_num  += n_k * abs(mean_pred - frac_pos)
    bin_data.append({
        'bin':       b + 1,
        'mean_pred': round(mean_pred, 6),
        'frac_pos':  round(frac_pos,  6),
        'count':     n_k,
    })

ece = float(ece_num / N_SAMPLES)
print(f"  ECE (10-bin): {ece:.4f}")

# Brier score (overall)
brier = float(np.mean((y_prob - y_brfss) ** 2))
print(f"  Brier score:  {brier:.4f}")

# Brier decomposition (Murphy 1973)
# BS = Reliability - Resolution + Uncertainty
y_bar = float(y_brfss.mean())
rel, res = 0.0, 0.0
for bd in bin_data:
    n_k = bd['count']
    if n_k == 0:
        continue
    p_k = bd['mean_pred']
    o_k = bd['frac_pos']
    rel += (n_k / N_SAMPLES) * (p_k - o_k) ** 2
    res += (n_k / N_SAMPLES) * (o_k - y_bar) ** 2
unc = y_bar * (1.0 - y_bar)
print(f"  Brier Reliability={rel:.4f}  Resolution={res:.4f}  Uncertainty={unc:.4f}")
print(f"  Brier check (R-Res+Unc): {rel-res+unc:.4f}  (actual: {brier:.4f})")


# ── Reliability diagram ───────────────────────────────────────────────────────
print("\n[5/5] Generating reliability diagram...")
fig, ax = plt.subplots(figsize=(7, 7))

# Diagonal reference
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration', zorder=1)

# Calibration curve with error bars (proportional to bin size)
mean_preds = [bd['mean_pred'] for bd in bin_data if bd['count'] > 0]
frac_poss  = [bd['frac_pos']  for bd in bin_data if bd['count'] > 0]
counts     = [bd['count']     for bd in bin_data if bd['count'] > 0]

# Error bars: 95% Wilson interval width ≈ 1/sqrt(n_k) scaled visually
yerr = [1.96 * np.sqrt(fp*(1-fp)/max(n,1)) for fp, n in zip(frac_poss, counts)]

ax.errorbar(mean_preds, frac_poss, yerr=yerr,
            fmt='o-', color='#2563EB', lw=2, ms=6, capsize=4,
            ecolor='#93C5FD', elinewidth=1.5,
            label=f'FedAvg (ECE={ece:.3f})', zorder=2)

# Shade between curve and diagonal
ax.fill_between(mean_preds, mean_preds, frac_poss, alpha=0.12, color='#2563EB')

# Histogram of predicted probabilities (inset)
ax_ins = ax.inset_axes([0.62, 0.08, 0.35, 0.25])
ax_ins.hist(y_prob, bins=20, color='#93C5FD', edgecolor='white', linewidth=0.5)
ax_ins.set_xlabel('Predicted prob.', fontsize=7)
ax_ins.set_ylabel('Count', fontsize=7)
ax_ins.tick_params(labelsize=7)
ax_ins.set_xlim(0, 1)

ax.set_xlabel('Mean Predicted Probability', fontsize=13)
ax.set_ylabel('Fraction of Positives', fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.25)
plt.tight_layout()

fig_path = os.path.join(FIGURES, 'reliability_diagram_fedavg.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")


# ── Save JSON results ─────────────────────────────────────────────────────────
out = {
    'ece_10bins':         round(ece, 6),
    'brier_score':        round(brier, 6),
    'brier_reliability':  round(rel, 6),
    'brier_resolution':   round(res, 6),
    'brier_uncertainty':  round(unc, 6),
    'n_samples':          N_SAMPLES,
    'n_bins':             N_BINS,
    'bin_data':           bin_data,
}
out_path = os.path.join(RESULTS_DIR, 'exp4_calibration.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*65}")
print(f"  ECE (10-bin):    {ece:.4f}")
print(f"  Brier:           {brier:.4f}")
print(f"  Reliability:     {rel:.4f}  (0 = perfect)")
print(f"  Resolution:      {res:.4f}  (higher = better)")
print(f"  Uncertainty:     {unc:.4f}  (base rate)")
print(f"{'='*65}")
print(f"\n✓ DONE: exp4_calibration.py — results saved to {out_path}")
print(f"  Figure: {fig_path}")
