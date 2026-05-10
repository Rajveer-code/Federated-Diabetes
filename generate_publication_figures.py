"""
generate_publication_figures.py
================================
Generates all 8 publication-quality figures for the FL diabetes manuscript.
Outputs to federated/results/figures/ at 300 dpi, colorblind-safe palette.
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc as sklearn_auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Colorblind-safe palette (Wong 2011, Nature Methods)
C = {
    'FedAvg'     : '#0072B2',  # blue
    'FedProx'    : '#E69F00',  # orange
    'FedNova'    : '#009E73',  # green
    'SCAFFOLD'   : '#CC79A7',  # pink
    'Centralised': '#D55E00',  # vermillion
    'FedAvg_ext' : '#56B4E9',  # sky blue
    'FedProx_ext': '#F0E442',  # yellow
    'FedNova_ext': '#009E73',
}
LS = {'FedAvg': '-', 'FedProx': '--', 'FedNova': '-.', 'SCAFFOLD': ':'}

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['Times New Roman', 'DejaVu Serif'],
    'font.size'       : 10,
    'axes.titlesize'  : 11,
    'axes.labelsize'  : 10,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'legend.fontsize' : 8.5,
    'figure.dpi'      : 150,
    'savefig.dpi'     : 300,
    'axes.spines.top' : False,
    'axes.spines.right': False,
})

# ── Load results ──────────────────────────────────────────────────────────────
def load(fname):
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        return json.load(f)

conv   = load('federated_convergence.json')
ext    = load('external_validation.json')
ci     = load('auc_confidence_intervals.json')
dp     = load('dp_results.json')
fair   = load('fairness_comparison.json')
calib  = load('calibration_results.json')
scaff  = load('scaffold_results.json')


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Federated Learning System Architecture
# ═══════════════════════════════════════════════════════════════════════
print("  Generating Figure 1: Architecture diagram...")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# --- Server box (centre top) ---
srv_x, srv_y = 5.0, 4.6
srv_w, srv_h = 2.4, 0.9
ax.add_patch(FancyBboxPatch((srv_x - srv_w/2, srv_y - srv_h/2),
    srv_w, srv_h, boxstyle='round,pad=0.1',
    facecolor='#DBEAFE', edgecolor='#1D4ED8', lw=1.8))
ax.text(srv_x, srv_y + 0.05, 'Aggregation Server\n(FedAvg / FedProx / FedNova / SCAFFOLD)',
        ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E3A8A')

# --- Hospital nodes ---
node_info = [
    (1.5, 2.4, 'Node A\nYoung Urban', '#ECFDF5', '#065F46', '4,500\nsamples'),
    (5.0, 2.4, 'Node B\nElderly Rural', '#FFF7ED', '#7C2D12', '3,405\nsamples'),
    (8.5, 2.4, 'Node C\nMixed Metro',  '#F5F3FF', '#4C1D95', '4,000\nsamples'),
]
for nx, ny, lbl, fc, ec, n in node_info:
    ax.add_patch(FancyBboxPatch((nx-1.1, ny-0.65), 2.2, 1.3,
        boxstyle='round,pad=0.1', facecolor=fc, edgecolor=ec, lw=1.5))
    ax.text(nx, ny + 0.15, lbl, ha='center', va='center',
            fontsize=8, fontweight='bold', color=ec)
    ax.text(nx, ny - 0.30, n, ha='center', va='center', fontsize=7.5, color='#374151')

    # Arrows: node → server (upload) and server → node (download)
    ax.annotate('', xy=(srv_x + (nx - srv_x)*0.42, srv_y - srv_h/2 + 0.02),
                xytext=(nx, ny + 0.7),
                arrowprops=dict(arrowstyle='->', color='#1D4ED8', lw=1.3,
                                connectionstyle='arc3,rad=0.15'))
    ax.annotate('', xy=(nx, ny + 0.7),
                xytext=(srv_x + (nx - srv_x)*0.42, srv_y - srv_h/2 + 0.02),
                arrowprops=dict(arrowstyle='->', color='#D97706', lw=1.3,
                                connectionstyle='arc3,rad=-0.15'))

# Upload / download labels
ax.text(2.8, 3.65, 'local\ngradients', ha='center', fontsize=7, color='#1D4ED8', style='italic')
ax.text(3.8, 3.35, 'global\nweights',  ha='center', fontsize=7, color='#D97706', style='italic')

# --- Privacy shield ---
ax.add_patch(mpatches.FancyArrowPatch((5.0, 2.4+0.65), (5.0, srv_y-srv_h/2),
    arrowstyle='-', color='gray', lw=0.8, linestyle='dashed'))
ax.text(5.35, 3.55, 'DP noise\n(ε ≤ 5)', ha='left', fontsize=7.5,
        color='#374151', bbox=dict(fc='#F9FAFB', ec='#D1D5DB', pad=2))

# --- Neural network diagram (bottom right) ---
nn_ox, nn_oy = 7.5, 1.0
layer_cfg = [('Input\n8 feat', 8, 0.0), ('64', 4, 1.1), ('32', 3, 2.2), ('16', 2, 3.3), ('Output', 1, 4.4)]
for li, (lname, nodes, lx) in enumerate(layer_cfg):
    ys = np.linspace(0.3, 1.7, nodes) if nodes > 1 else [1.0]
    for yn in ys:
        col = '#BFDBFE' if li == 0 else ('#A7F3D0' if li == len(layer_cfg)-1 else '#E9D5FF')
        c = plt.Circle((nn_ox + lx*0.6, nn_oy + yn), 0.09,
                        color=col, ec='#374151', lw=0.7, zorder=3)
        ax.add_patch(c)
    ax.text(nn_ox + lx*0.6, nn_oy - 0.12, lname,
            ha='center', va='top', fontsize=6.5, color='#374151')

ax.text(nn_ox + 2.2*0.6, nn_oy + 2.0, 'DiabetesNet Architecture',
        ha='center', fontsize=8, style='italic', color='#374151')

# Legend arrows
ax.annotate('', xy=(0.9, 0.45), xytext=(0.3, 0.45),
            arrowprops=dict(arrowstyle='->', color='#1D4ED8', lw=1.3))
ax.text(1.05, 0.45, 'Model upload (local gradients)', va='center', fontsize=7.5, color='#1D4ED8')
ax.annotate('', xy=(0.3, 0.15), xytext=(0.9, 0.15),
            arrowprops=dict(arrowstyle='->', color='#D97706', lw=1.3))
ax.text(1.05, 0.15, 'Model download (global weights)', va='center', fontsize=7.5, color='#D97706')

ax.set_title('Figure 1. Privacy-Preserving Federated Learning Framework for Diabetes Risk Prediction\n'
             'Three demographically distinct hospital nodes train a shared DiabetesNet without sharing raw data.',
             fontsize=9, pad=10)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_architecture.png'), bbox_inches='tight')
plt.close(fig)
print("    Saved fig1_architecture.png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Convergence Curves (all 4 FL strategies)
# ═══════════════════════════════════════════════════════════════════════
print("  Generating Figure 2: Convergence curves...")

fig, ax = plt.subplots(figsize=(7, 4.2))

methods_conv = {
    'FedAvg' : (conv['FedAvg']['rounds'],   conv['FedAvg']['aucs']),
    'FedProx': (conv['FedProx']['rounds'],  conv['FedProx']['aucs']),
    'FedNova': (conv['FedNova']['rounds'],  conv['FedNova']['aucs']),
    'SCAFFOLD': (scaff['rounds'], scaff['aucs']),
}

for name, (rnds, aucs) in methods_conv.items():
    ax.plot(rnds, aucs, color=C[name], ls=LS[name],
            lw=2.0, label=f"{name} (final={aucs[-1]:.3f})", zorder=3)

ax.set_xlabel('Communication Round')
ax.set_ylabel('Validation AUC (NHANES internal test set)')
ax.set_xlim(1, 50)
ax.set_ylim(0.58, 0.81)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.legend(framealpha=0.85, loc='lower right')
ax.grid(axis='y', alpha=0.3, ls=':')
ax.axhline(0.5, color='gray', lw=0.8, ls=':', label='Random (AUC=0.5)')

ax.annotate('SCAFFOLD converges slower\nwith SGD vs AdamW',
            xy=(50, 0.642), xytext=(32, 0.606),
            arrowprops=dict(arrowstyle='->', color='#CC79A7', lw=1.0),
            fontsize=7.5, color='#CC79A7')

ax.set_title('Figure 2. Convergence of federated strategies over 50 communication rounds\n'
             '(NHANES internal validation AUC; three hospital nodes)',
             fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_convergence.png'), bbox_inches='tight')
plt.close(fig)
print("    Saved fig2_convergence.png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — ROC Curves (internal + external)
# ═══════════════════════════════════════════════════════════════════════
print("  Generating Figure 3: ROC curves...")

# Load pred arrays
def load_npy(fname):
    return np.load(os.path.join(RESULTS_DIR, fname))

y_int  = load_npy('y_true_internal.npy')
y_ext  = load_npy('y_true_brfss.npy')

pred_files_int = {
    'FedAvg'    : 'pred_fedavg_internal.npy',
    'FedProx'   : 'pred_fedprox_internal.npy',
    'FedNova'   : 'pred_fednova_internal.npy',
    'SCAFFOLD'  : 'pred_scaffold_internal.npy',
    'Centralised': 'pred_xgb_internal.npy',
}
pred_files_ext = {
    'FedAvg'    : 'pred_fedavg_external.npy',
    'FedProx'   : 'pred_fedprox_external.npy',
    'FedNova'   : 'pred_fednova_external.npy',
    'Centralised': 'pred_xgb_external.npy',
}

col_ext = {
    'FedAvg': '#56B4E9', 'FedProx': '#F0E442',
    'FedNova': '#009E73', 'Centralised': '#F4A261',
}

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (y_true, pfiles, title) in zip(axes, [
    (y_int, pred_files_int, 'Internal Validation (NHANES, n=15,650)'),
    (y_ext, pred_files_ext, 'External Validation (BRFSS 2020-22, n=1,282,897)'),
]):
    ax.plot([0,1],[0,1], color='gray', lw=0.8, ls=':', label='Random')
    for name, fname in pfiles.items():
        try:
            probs = load_npy(fname)
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc_val = sklearn_auc(fpr, tpr)
            col = C.get(name, '#888')
            ls  = LS.get(name, '-')
            ax.plot(fpr, tpr, color=col, lw=1.8, ls=ls,
                    label=f'{name} (AUC={roc_auc_val:.3f})')
        except FileNotFoundError:
            pass
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=9.5)
    ax.legend(framealpha=0.85, fontsize=8)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.2, ls=':')

fig.suptitle('Figure 3. Receiver Operating Characteristic (ROC) curves for all models on internal and external test sets',
             fontsize=9.5, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_roc_curves.png'), bbox_inches='tight')
plt.close(fig)
print("    Saved fig3_roc_curves.png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Fairness: Subgroup AUC across methods (BRFSS external)
# ═══════════════════════════════════════════════════════════════════════
print("  Generating Figure 4: Fairness analysis...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Age subgroup AUCs
models_fair = ['Centralised', 'FedAvg', 'FedProx', 'FedNova']
age_18  = [
    ext['centralised']['fairness']['age_18-39'],
    ext['federated']['FedAvg']['fairness']['age_18-39'],
    ext['federated']['FedProx']['fairness']['age_18-39'],
    ext['federated']['FedNova']['fairness']['age_18-39'],
]
age_60  = [
    ext['centralised']['fairness']['age_60+'],
    ext['federated']['FedAvg']['fairness']['age_60+'],
    ext['federated']['FedProx']['fairness']['age_60+'],
    ext['federated']['FedNova']['fairness']['age_60+'],
]
gaps    = [a - b for a, b in zip(age_18, age_60)]

x = np.arange(len(models_fair))
w = 0.3

ax = axes[0]
b1 = ax.bar(x - w/2, age_18, w, label='Young (18–39)', color='#56B4E9', edgecolor='white')
b2 = ax.bar(x + w/2, age_60, w, label='Elderly (≥60)',  color='#E69F00', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(models_fair, rotation=15, ha='right')
ax.set_ylabel('AUC')
ax.set_ylim(0.55, 0.80)
ax.legend()
ax.set_title('(A) Age Subgroup AUC — External Validation (BRFSS)', fontsize=9.5)
ax.grid(axis='y', alpha=0.3, ls=':')
for xi, (y18, y60) in enumerate(zip(age_18, age_60)):
    ax.text(xi, max(y18, y60)+0.005, f'Δ={y18-y60:.3f}', ha='center', fontsize=7.5, color='#374151')

# Panel B: Elderly gap comparison
ax2 = axes[1]
colors_bar = [C.get(m, '#888') for m in models_fair]
gap_pct = [g * 100 for g in gaps]
bars = ax2.bar(models_fair, gap_pct, color=colors_bar, edgecolor='white', width=0.5)
ax2.axhline(0, color='gray', lw=0.8, ls=':')
ax2.set_ylabel('Fairness Gap (Young AUC − Elderly AUC) ×100')
ax2.set_title('(B) Fairness Gap by Strategy\n(Lower gap = more equitable)', fontsize=9.5)
ax2.set_ylim(-1, 12)
ax2.grid(axis='y', alpha=0.3, ls=':')
for bar, gp in zip(bars, gap_pct):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{gp:.1f}pp', ha='center', fontsize=8.5, fontweight='bold')

# Reference line — published benchmark
ax2.axhline(13.5, color='red', lw=1.2, ls='--', alpha=0.7)
ax2.text(3.45, 13.8, 'Published\nbenchmark\n(Δ=0.135)', ha='right', fontsize=7.5, color='red')

fig.suptitle('Figure 4. Fairness analysis across demographic subgroups — BRFSS external validation (n=1,282,897)',
             fontsize=9.5, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig4_fairness.png'), bbox_inches='tight')
plt.close(fig)
print("    Saved fig4_fairness.png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5 — DP Privacy-Utility Tradeoff
# ═══════════════════════════════════════════════════════════════════════
print("  Generating Figure 5: DP privacy-utility tradeoff...")

dp_eps    = [0.5, 1.0, 2.0, 5.0, float('inf')]
dp_auc    = dp['auc']
dp_young  = dp['young_auc']
dp_eld    = dp['elderly_auc']
dp_x_lbl  = ['0.5', '1.0', '2.0', '5.0', '∞']

fig, ax = plt.subplots(figsize=(7, 4.2))
x_pos = np.arange(len(dp_x_lbl))

ax.bar(x_pos[:-1], dp_auc[:-1], 0.5, color='#BFDBFE', edgecolor='#1D4ED8', label='DP-SGD AUC', zorder=2)
ax.bar(x_pos[-1], dp_auc[-1],   0.5, color='#A7F3D0', edgecolor='#065F46', label='No DP (ε=∞)', zorder=2)
ax.axhline(0.5, color='gray', lw=1.0, ls=':', label='Random classifier')
ax.axhline(dp_auc[-1], color='#065F46', lw=1.2, ls='--', alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'ε={l}' for l in dp_x_lbl], fontsize=9)
ax.set_xlabel('Privacy Budget ε (smaller = more private)')
ax.set_ylabel('AUC')
ax.set_ylim(0.45, 0.82)
ax.legend(framealpha=0.85)
ax.grid(axis='y', alpha=0.3, ls=':')

for xi, (yv, lbl) in enumerate(zip(dp_auc, dp_x_lbl)):
    ax.text(xi, yv + 0.01, f'{yv:.3f}', ha='center', fontsize=8.5, fontweight='bold')

ax.annotate('Model collapses at\ntight ε — fundamental\nprivacy-utility tension',
            xy=(1.5, 0.51), xytext=(1.8, 0.60),
            arrowprops=dict(arrowstyle='->', color='#991B1B', lw=1.0),
            fontsize=7.5, color='#991B1B')

ax.set_title('Figure 5. Privacy-utility tradeoff under differential privacy\n'
             'DP-SGD (δ=10⁻⁵, batch=512, 5 local epochs); FedAvg without DP: AUC=0.766',
             fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig5_dp_tradeoff.png'), bbox_inches='tight')
plt.close(fig)
print("    Saved fig5_dp_tradeoff.png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Calibration (4-panel reliability diagram)
# ═══════════════════════════════════════════════════════════════════════
print("  Generating Figure 6: Calibration reliability diagram...")

# Compute calibration curves from BRFSS predictions
probs_raw  = load_npy('pred_fedprox_external.npy')
y_ext_true = load_npy('y_true_brfss.npy')

# Apply Platt and Isotonic from saved parameters
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Platt: logistic regression on logit space
coef = calib['platt']['coef']
intercept = calib['platt']['intercept']
logit_raw = np.log(np.clip(probs_raw, 1e-7, 1-1e-7) / (1 - np.clip(probs_raw, 1e-7, 1-1e-7)))
probs_platt = 1 / (1 + np.exp(-(coef * logit_raw + intercept)))

# Temperature scaling
T_opt = calib['temperature']['T_optimal']
probs_temp = 1 / (1 + np.exp(-logit_raw / T_opt))

fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
axes = axes.flatten()

calib_sets = [
    ('Uncalibrated', probs_raw, f"ECE={calib['uncalibrated']['ece']:.3f}", '#D55E00'),
    ('Platt Scaling', probs_platt, f"ECE={calib['platt']['ece']:.4f}", '#0072B2'),
    ('Temperature Scaling (T={:.2f})'.format(T_opt), probs_temp,
     f"ECE={calib['temperature']['ece']:.3f}", '#009E73'),
    ('Isotonic Regression', None, f"ECE={calib['isotonic']['ece']:.4f}", '#CC79A7'),
]

for i, (title, probs, ece_lbl, col) in enumerate(calib_sets):
    ax = axes[i]
    if probs is None:
        ax.text(0.5, 0.5, f'Isotonic\n{ece_lbl}\n(fitted on 20% cal split)',
                ha='center', va='center', fontsize=11, transform=ax.transAxes,
                color=col)
        ax.plot([0,1],[0,1], 'k--', lw=1.2, label='Perfect calibration')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
    else:
        n_bins = 10
        try:
            frac_pos, mean_pred = calibration_curve(y_ext_true, probs,
                                                     n_bins=n_bins, strategy='uniform')
            ax.plot(mean_pred, frac_pos, 's-', color=col, lw=1.8, ms=5, label='Model')
        except Exception:
            pass
        ax.plot([0,1],[0,1], 'k--', lw=1.2, label='Perfect calibration')
        ax.fill_between([0,1],[0,1],[0,1], alpha=0.05, color='gray')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(fontsize=8)
    ax.set_xlabel('Mean Predicted Probability', fontsize=9)
    ax.set_ylabel('Fraction of Positives', fontsize=9)
    ax.set_title(f'{title}\n{ece_lbl}', fontsize=9.5, color=col)
    ax.grid(alpha=0.2, ls=':')

fig.suptitle('Figure 6. Calibration reliability diagrams for FedProx on BRFSS external test set\n'
             '(20% calibration / 80% test split; 10 equal-width bins)',
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(FIG_DIR, 'fig6_calibration.png'), bbox_inches='tight')
plt.close(fig)
print("    Saved fig6_calibration.png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Generalisation Gap (internal vs external AUC)
# ═══════════════════════════════════════════════════════════════════════
print("  Generating Figure 7: Generalisation gap...")

models_gap = ['Centralised\n(XGBoost)', 'FedAvg', 'FedProx', 'FedNova']
auc_int = [
    ci['internal']['XGBoost']['auc'],
    ci['internal']['FedAvg']['auc'],
    ci['internal']['FedProx']['auc'],
    ci['internal']['FedNova']['auc'],
]
auc_ext = [
    ci['external']['XGBoost']['auc'],
    ci['external']['FedAvg']['auc'],
    ci['external']['FedProx']['auc'],
    ci['external']['FedNova']['auc'],
]
auc_int_lo = [ci['internal'][k]['lower'] for k in ['XGBoost','FedAvg','FedProx','FedNova']]
auc_int_hi = [ci['internal'][k]['upper'] for k in ['XGBoost','FedAvg','FedProx','FedNova']]
auc_ext_lo = [ci['external'][k]['lower'] for k in ['XGBoost','FedAvg','FedProx','FedNova']]
auc_ext_hi = [ci['external'][k]['upper'] for k in ['XGBoost','FedAvg','FedProx','FedNova']]

x = np.arange(len(models_gap))
w = 0.32

fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w/2, auc_int, w, label='Internal (NHANES)',
            color='#93C5FD', edgecolor='#1D4ED8', lw=0.8)
b2 = ax.bar(x + w/2, auc_ext, w, label='External (BRFSS)',
            color='#6EE7B7', edgecolor='#065F46', lw=0.8)

# Error bars
ax.errorbar(x - w/2, auc_int,
            yerr=[np.array(auc_int)-np.array(auc_int_lo),
                  np.array(auc_int_hi)-np.array(auc_int)],
            fmt='none', color='#1D4ED8', capsize=3, lw=1.0)
ax.errorbar(x + w/2, auc_ext,
            yerr=[np.array(auc_ext)-np.array(auc_ext_lo),
                  np.array(auc_ext_hi)-np.array(auc_ext)],
            fmt='none', color='#065F46', capsize=3, lw=1.0)

ax.set_xticks(x); ax.set_xticklabels(models_gap)
ax.set_ylabel('AUC (95% CI)')
ax.set_ylim(0.65, 0.82)
ax.legend(framealpha=0.85)
ax.grid(axis='y', alpha=0.3, ls=':')

for xi, (vi, ve) in enumerate(zip(auc_int, auc_ext)):
    gap = vi - ve
    ax.text(xi, max(vi, ve)+0.007, f'Δ={gap:.3f}',
            ha='center', fontsize=8, color='#374151')

ax.set_title('Figure 7. Generalisation gap: internal NHANES vs. external BRFSS AUC\n'
             '(Error bars = 95% CI; internal: stratified bootstrap; external: DeLong)',
             fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig7_generalisation_gap.png'), bbox_inches='tight')
plt.close(fig)
print("    Saved fig7_generalisation_gap.png")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8 — AUC Summary Table (all methods, all metrics)
# ═══════════════════════════════════════════════════════════════════════
print("  Generating Figure 8: Summary comparison table...")

methods_all = ['Centralised\n(XGBoost)', 'FedAvg', 'FedProx\n(μ=0.1)', 'FedNova', 'SCAFFOLD\n(Opt. II)']
internal_aucs = [
    ci['internal']['XGBoost']['auc'],
    ci['internal']['FedAvg']['auc'],
    ci['internal']['FedProx']['auc'],
    ci['internal']['FedNova']['auc'],
    scaff['final_auc'],
]
external_aucs = [
    ci['external']['XGBoost']['auc'],
    ci['external']['FedAvg']['auc'],
    ci['external']['FedProx']['auc'],
    ci['external']['FedNova']['auc'],
    None,  # SCAFFOLD external not run
]
elderly_gaps = [
    ext['centralised']['fairness']['elderly_gap'],
    ext['federated']['FedAvg']['fairness']['elderly_gap'],
    ext['federated']['FedProx']['fairness']['elderly_gap'],
    ext['federated']['FedNova']['fairness']['elderly_gap'],
    None,
]

x = np.arange(len(methods_all))
w = 0.28

fig, ax = plt.subplots(figsize=(10, 4.8))

int_vals = np.array(internal_aucs)
ext_vals = np.array([v if v is not None else np.nan for v in external_aucs])
gap_vals = np.array([v if v is not None else np.nan for v in elderly_gaps])

b1 = ax.bar(x - w, int_vals, w, label='Internal AUC', color='#93C5FD', edgecolor='#1D4ED8', lw=0.8)
b2 = ax.bar(x,     ext_vals, w, label='External AUC', color='#6EE7B7', edgecolor='#065F46', lw=0.8)
b3 = ax.bar(x + w, gap_vals * 3, w, label='Fairness Gap (×3 scaled)',
            color='#FCA5A5', edgecolor='#DC2626', lw=0.8)

ax.set_xticks(x); ax.set_xticklabels(methods_all, fontsize=9)
ax.set_ylabel('AUC / Fairness Gap (×3)')
ax.set_ylim(0.0, 0.90)
ax.legend(framealpha=0.85, fontsize=8.5)
ax.grid(axis='y', alpha=0.3, ls=':')
ax.axhline(0.5, color='gray', lw=0.8, ls=':')

for xi, (iv, ev, gv) in enumerate(zip(int_vals, ext_vals, gap_vals)):
    ax.text(xi - w, iv + 0.012, f'{iv:.3f}', ha='center', fontsize=7.5, fontweight='bold', color='#1D4ED8')
    if not np.isnan(ev):
        ax.text(xi, ev + 0.012, f'{ev:.3f}', ha='center', fontsize=7.5, fontweight='bold', color='#065F46')
    if not np.isnan(gv):
        ax.text(xi + w, gv*3 + 0.012, f'{gv:.3f}', ha='center', fontsize=7.5, fontweight='bold', color='#DC2626')

ax.set_title('Figure 8. Performance comparison across all federated and centralised strategies\n'
             'Internal: NHANES (n=15,650); External: BRFSS 2020–22 (n=1,282,897)',
             fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig8_summary_comparison.png'), bbox_inches='tight')
plt.close(fig)
print("    Saved fig8_summary_comparison.png")


print()
print("=" * 55)
print("  All 8 figures saved to results/figures/")
print("=" * 55)
for fn in sorted(os.listdir(FIG_DIR)):
    if fn.endswith('.png'):
        size = os.path.getsize(os.path.join(FIG_DIR, fn)) // 1024
        print(f"    {fn}  ({size} KB)")
