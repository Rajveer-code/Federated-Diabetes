"""
32_multiseed_matched_control.py — seed-robustness for the sample-matched control
================================================================================
Why this exists
---------------
The sample-matched centralised control anchors the paper's central claim about
whether federation's generalisation advantage survives a composition-matched
comparison. Its previously reported gap (0.0239) came from a script with two
data-handling defects (complete-case BRFSS deletion -> 720,161 rows instead of
1,282,897; no BRFSS 0/1 -> NHANES 1/2 sex remap). After fixing both, the gap at
seed 42 is 0.0476 — a different conclusion.

But that single number is one seed, and no training script in this project set a
seed at all before now. So the direction of the corrected result cannot be
asserted from one run. This script runs the corrected pipeline across N seeds
and reports the distribution, so the manuscript can state a mean and spread
rather than a point estimate that may be initialisation noise.

Output: results/matched_control_multiseed.json

Usage:
  python 32_multiseed_matched_control.py            # default 5 seeds
  python 32_multiseed_matched_control.py --seeds 10
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
import joblib
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_PATHS, RESULTS_DIR, FEATURE_COLS, TARGET_COL,
    GLOBAL_SCALER_PATH, NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY,
    FL_NUM_ROUNDS, SEED,
)
from nn_model   import DiabetesNet, train_one_epoch, get_device
from data_utils import DiabetesDataset, compute_class_weight, load_brfss_clean

BASE_SEED = SEED   # 42; seeds run as BASE_SEED + i


def run_one_seed(seed, X_all, y_all, X_brfss, y_brfss, device):
    """Train the matched-subset centralised model at one seed, return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=seed
    )

    pos_w     = compute_class_weight(y_tr)
    model     = DiabetesNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=NN_LR,
                                  weight_decay=NN_WEIGHT_DECAY)
    loader = DataLoader(DiabetesDataset(X_tr, y_tr),
                        batch_size=NN_BATCH_SIZE, shuffle=True)
    amp = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    for _ in range(FL_NUM_ROUNDS):
        train_one_epoch(model, loader, optimizer, criterion, device,
                        proximal_mu=0.0, global_params=None, scaler=amp)

    model.eval()
    with torch.no_grad():
        p_int = torch.sigmoid(model(torch.FloatTensor(X_te).to(device))).cpu().numpy()

    # BRFSS is 1.28M rows — score in chunks to stay inside 8 GB VRAM.
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X_brfss), 65536):
            batch = torch.FloatTensor(X_brfss[i:i + 65536]).to(device)
            chunks.append(torch.sigmoid(model(batch)).cpu().numpy())
    p_ext = np.concatenate(chunks)

    auc_int = float(roc_auc_score(y_te, p_int))
    auc_ext = float(roc_auc_score(y_brfss, p_ext))
    return {
        'seed': int(seed),
        'auc_internal': round(auc_int, 4),
        'auc_external': round(auc_ext, 4),
        'generalisation_gap': round(auc_int - auc_ext, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5, help='number of seeds')
    args = ap.parse_args()

    device = get_device()
    scaler = joblib.load(GLOBAL_SCALER_PATH)

    df_fed = pd.concat([pd.read_csv(p) for p in NODE_PATHS], ignore_index=True)
    X_all  = scaler.transform(df_fed[FEATURE_COLS].values.astype(np.float32))
    y_all  = df_fed[TARGET_COL].values.astype(np.float32)
    print(f"Federated subset: n={len(df_fed):,}")

    _, X_brfss_raw, y_brfss = load_brfss_clean()
    X_brfss = scaler.transform(X_brfss_raw)

    runs = []
    for i in range(args.seeds):
        seed = BASE_SEED + i
        r = run_one_seed(seed, X_all, y_all, X_brfss, y_brfss, device)
        runs.append(r)
        print(f"  seed {seed}: internal={r['auc_internal']:.4f}  "
              f"external={r['auc_external']:.4f}  gap={r['generalisation_gap']:.4f}")

    gaps = np.array([r['generalisation_gap'] for r in runs])
    ext  = np.array([r['auc_external']       for r in runs])
    intl = np.array([r['auc_internal']       for r in runs])

    summary = {
        'description': 'Sample-matched centralised DiabetesNet (n=11,906), '
                       'corrected BRFSS pipeline, multi-seed',
        'n_seeds': len(runs),
        'n_brfss': int(len(y_brfss)),
        'runs': runs,
        'auc_internal': {'mean': round(float(intl.mean()), 4),
                         'sd': round(float(intl.std(ddof=1)), 4)},
        'auc_external': {'mean': round(float(ext.mean()), 4),
                         'sd': round(float(ext.std(ddof=1)), 4)},
        'generalisation_gap': {'mean': round(float(gaps.mean()), 4),
                               'sd': round(float(gaps.std(ddof=1)), 4),
                               'min': round(float(gaps.min()), 4),
                               'max': round(float(gaps.max()), 4)},
        'superseded_buggy_result': {
            'generalisation_gap': 0.0239,
            'n_brfss': 720161,
            'defects': ['complete-case BRFSS deletion (43.9% of rows dropped)',
                        'no BRFSS 0/1 -> NHANES 1/2 sex remap'],
        },
    }

    print("\n" + "=" * 62)
    print(f"  gap  mean={gaps.mean():.4f}  sd={gaps.std(ddof=1):.4f}  "
          f"range=[{gaps.min():.4f}, {gaps.max():.4f}]")
    print(f"  ext  mean={ext.mean():.4f}  sd={ext.std(ddof=1):.4f}")
    print("=" * 62)

    out = os.path.join(RESULTS_DIR, 'matched_control_multiseed.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
