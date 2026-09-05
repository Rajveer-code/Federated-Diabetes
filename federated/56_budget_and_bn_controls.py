"""
56_budget_and_bn_controls.py -- the two controls a reviewer will demand
=======================================================================
Two alternative explanations for the AUC/allocation divergence are not yet ruled
out by anything in the pipeline:

  (1) TRAINING BUDGET. The composition-matched centralised control runs
      FL_NUM_ROUNDS (50) epochs, while each federated client runs
      FL_NUM_ROUNDS x NN_LOCAL_EPOCHS (250) local epochs. Matched on rounds,
      not on example passes. If the control is simply undertrained, the
      divergence could be a capacity artefact rather than a paradigm effect.
      -> Arm A trains the same centralised control for 250 epochs.

  (2) BATCHNORM. DiabetesNet uses nn.BatchNorm1d, whose running statistics are
      pooled across non-IID clients (fl_core.aggregate_buffers). Correcting a
      defect in that pooling moved FedAvg external AUC by 0.005 -- the same
      order as the effect under test. So the divergence could be an interaction
      between non-IID federation and BN rather than a property of federated
      averaging.
      -> Arm B reruns FedAvg and its matched control with BN removed entirely.

Both arms cache prediction vectors into results/preds_multiseed/ under new model
names, so 57_budget_bn_readout.py scores them with exactly the same code path as
the headline models. Nothing here retrains or overwrites an existing model.

Arms (10 seeds each, seeds 42..51):
  Centralised-matched-250ep   centralised, 250 epochs, BN as usual
  Centralised-matched-noBN    centralised,  50 epochs, BN removed
  FedAvg-noBN                 FedAvg, 50 rounds x 5 local epochs, BN removed

Output: results/budget_bn_controls.json  +  results/preds_multiseed/*.npy
Usage:  python 56_budget_and_bn_controls.py --seeds 10
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
warnings.filterwarnings('ignore')
import joblib
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_paths import (
    NODE_PATHS, RESULTS_DIR, TARGET_COL,
    FEATURE_COLS_NORACE as FEATURE_COLS,
    GLOBAL_SCALER_NORACE_PATH as GLOBAL_SCALER_PATH, N_FEATURES_NORACE,
    NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY,
    FL_NUM_ROUNDS, NN_LOCAL_EPOCHS, SEED,
)
from nn_model   import train_one_epoch
from data_utils import (DiabetesDataset, compute_class_weight, load_brfss_clean,
                        set_params_from_numpy, get_params_as_numpy)
import fl_core
from fl_core import DEVICE, seed_everything, build_node_loaders, run_federated, fedavg_aggregate

CHUNK = 65536
PRED_DIR = os.path.join(RESULTS_DIR, 'preds_multiseed')

# Matched example passes: each federated client sees NN_LOCAL_EPOCHS epochs of its
# own shard per round, for FL_NUM_ROUNDS rounds.
MATCHED_EPOCHS = FL_NUM_ROUNDS * NN_LOCAL_EPOCHS   # 250


class DiabetesNetNoBN(nn.Module):
    """DiabetesNet with nn.BatchNorm1d removed and nothing else changed.

    Same widths, same ReLU, same dropout schedule (the last hidden block keeps
    the 0.6x dropout), same Kaiming init. Removing BN removes all buffers, so
    fl_core.aggregate_buffers receives empty dicts and is a no-op -- which is
    the point: this arm contains no cross-client statistic pooling at all.
    """

    def __init__(self, input_dim: int = N_FEATURES_NORACE,
                 hidden_dims: List[int] = [64, 32, 16], dropout: float = 0.3):
        super().__init__()
        layers, in_dim = [], input_dim
        for i, h in enumerate(hidden_dims):
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.6),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cache(name, arr):
    np.save(os.path.join(PRED_DIR, f'{name}.npy'), np.asarray(arr, dtype=np.float32))


@torch.no_grad()
def score(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), CHUNK):
        out.append(torch.sigmoid(
            model(torch.FloatTensor(X[i:i + CHUNK]).to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def train_centralised(X_tr, y_tr, seed, epochs, net_factory):
    """Centralised DiabetesNet for `epochs` passes. Identical to
    33_multiseed_all_models.train_centralised except epochs and architecture
    are explicit rather than fixed at FL_NUM_ROUNDS with BN."""
    seed_everything(seed)
    model     = net_factory().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([compute_class_weight(y_tr)], device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY)
    dl  = DataLoader(DiabetesDataset(X_tr, y_tr), batch_size=NN_BATCH_SIZE, shuffle=True)
    amp = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    for _ in range(epochs):
        train_one_epoch(model, dl, opt, criterion, DEVICE,
                        proximal_mu=0.0, global_params=None, scaler=amp)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()
    seed_list = [SEED + i for i in range(args.seeds)]

    os.makedirs(PRED_DIR, exist_ok=True)
    scaler = joblib.load(GLOBAL_SCALER_PATH)

    _, X_b_raw, y_b = load_brfss_clean(feature_cols=FEATURE_COLS)
    X_b = scaler.transform(X_b_raw)
    print(f"Common BRFSS test set: n={len(y_b):,}")

    df_fed = pd.concat([pd.read_csv(p) for p in NODE_PATHS], ignore_index=True)
    X_fed  = scaler.transform(df_fed[FEATURE_COLS].values.astype(np.float32))
    y_fed  = df_fed[TARGET_COL].values.astype(np.float32)
    print(f"Matched federated cohort: n={len(y_fed):,}   "
          f"budget arm = {MATCHED_EPOCHS} epochs "
          f"({FL_NUM_ROUNDS} rounds x {NN_LOCAL_EPOCHS} local epochs)")

    records = []
    for seed in seed_list:
        print(f"\n{'='*66}\n  SEED {seed}\n{'='*66}")

        # ── Arm A: matched control at matched example passes ──────────────────
        Xtr, Xte, ytr, yte = train_test_split(
            X_fed, y_fed, test_size=0.20, stratify=y_fed, random_state=seed)
        m = train_centralised(Xtr, ytr, seed, MATCHED_EPOCHS, fl_core.new_net)
        p_int, p_ext = score(m, Xte), score(m, X_b)
        cache(f'Centralised-matched-250ep_seed{seed}_external', p_ext)
        cache(f'Centralised-matched-250ep_seed{seed}_internal', p_int)
        cache(f'_y_internal_Centralised-matched-250ep_seed{seed}', yte)
        records.append(dict(model='Centralised-matched-250ep', seed=seed,
                            epochs=MATCHED_EPOCHS, batchnorm=True,
                            internal_val=float(roc_auc_score(yte, p_int)),
                            external=float(roc_auc_score(y_b, p_ext))))
        print(f"  {'Centralised-matched-250ep':<26} "
              f"val={records[-1]['internal_val']:.4f} ext={records[-1]['external']:.4f}")

        # ── Arm B: BN removed, control and FedAvg ─────────────────────────────
        m = train_centralised(Xtr, ytr, seed, FL_NUM_ROUNDS, DiabetesNetNoBN)
        p_int, p_ext = score(m, Xte), score(m, X_b)
        cache(f'Centralised-matched-noBN_seed{seed}_external', p_ext)
        cache(f'Centralised-matched-noBN_seed{seed}_internal', p_int)
        cache(f'_y_internal_Centralised-matched-noBN_seed{seed}', yte)
        records.append(dict(model='Centralised-matched-noBN', seed=seed,
                            epochs=FL_NUM_ROUNDS, batchnorm=False,
                            internal_val=float(roc_auc_score(yte, p_int)),
                            external=float(roc_auc_score(y_b, p_ext))))
        print(f"  {'Centralised-matched-noBN':<26} "
              f"val={records[-1]['internal_val']:.4f} ext={records[-1]['external']:.4f}")

        # FedAvg with the BN-free net. Monkeypatching fl_core.new_net is what
        # makes local_train, run_federated and eval_global all build the same
        # architecture; restored immediately after so nothing else is affected.
        seed_everything(seed)
        loaders, n_samples = build_node_loaders(seed=seed)
        X_val = np.concatenate([dl_v.dataset.X.numpy() for (_, dl_v, _, _) in loaders])
        y_val = np.concatenate([dl_v.dataset.y.numpy() for (_, dl_v, _, _) in loaders])

        _orig_new_net = fl_core.new_net
        fl_core.new_net = lambda: DiabetesNetNoBN()
        try:
            r = run_federated('FedAvg-noBN', fedavg_aggregate, loaders, n_samples,
                              proximal_mu=0.0, seed=seed, save_weights=False,
                              verbose=False)
            m = DiabetesNetNoBN().to(DEVICE)
            set_params_from_numpy(m, r['global_params'])
        finally:
            fl_core.new_net = _orig_new_net

        p_val, p_ext = score(m, X_val), score(m, X_b)
        cache(f'FedAvg-noBN_seed{seed}_external', p_ext)
        cache(f'FedAvg-noBN_seed{seed}_internal', p_val)
        cache(f'_y_internal_FedAvg-noBN_seed{seed}', y_val)
        records.append(dict(model='FedAvg-noBN', seed=seed,
                            epochs=FL_NUM_ROUNDS * NN_LOCAL_EPOCHS, batchnorm=False,
                            internal_val=float(roc_auc_score(y_val, p_val)),
                            external=float(roc_auc_score(y_b, p_ext))))
        print(f"  {'FedAvg-noBN':<26} "
              f"val={records[-1]['internal_val']:.4f} ext={records[-1]['external']:.4f}")

    df = pd.DataFrame(records)
    summary = {
        model: {k: {'mean': round(float(g[k].mean()), 4),
                    'sd': round(float(g[k].std(ddof=1)), 4)}
                for k in ['internal_val', 'external']}
        for model, g in df.groupby('model')
    }
    print(f"\n{'='*70}\n  {'model':<28}{'internal':>14}{'external':>18}\n{'='*70}")
    for model, s in summary.items():
        print(f"  {model:<28}{s['internal_val']['mean']:.4f}+-{s['internal_val']['sd']:.4f}  "
              f"{s['external']['mean']:.4f}+-{s['external']['sd']:.4f}")
    print('=' * 70)

    out = os.path.join(RESULTS_DIR, 'budget_bn_controls.json')
    with open(out, 'w') as f:
        json.dump({'n_seeds': args.seeds, 'seeds': seed_list,
                   'matched_epochs': MATCHED_EPOCHS,
                   'n_brfss': int(len(y_b)), 'runs': records, 'summary': summary},
                  f, indent=2)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
