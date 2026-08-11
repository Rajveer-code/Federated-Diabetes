"""
33_multiseed_all_models.py — seed-robustness for every model, common test set
=============================================================================
Answers the question the corrected matched control raised: with the BRFSS
pipeline unified and initialisation seeded, is FedAvg's generalisation gap
actually smaller than the sample-matched centralised control's, or was the
apparent ordering an artefact of single uncontrolled runs?

Every model here is trained fresh per seed and scored on the SAME BRFSS
population (1,282,897 rows, via data_utils.load_brfss_clean).

Internal AUC is reported two ways, because the generalisation gap depends on it:
  * internal_val   — held-out node validation splits (FL) / model's own 20%
                     holdout (centralised). Clean, no training contamination.
  * internal_pool  — full centralised NHANES pool (15,650). This is what
                     fl_core.eval_global tracks, but for federated models that
                     pool CONTAINS the nodes' training rows, so it is optimistic.
                     Reported only for continuity with earlier runs.

Output: results/multiseed_all_models.json

Usage:
  python 33_multiseed_all_models.py               # 5 seeds
  python 33_multiseed_all_models.py --seeds 3     # quick check
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
    NODE_PATHS, CENTRALISED_PATH, RESULTS_DIR, TARGET_COL,
    FEATURE_COLS_NORACE as FEATURE_COLS,
    GLOBAL_SCALER_NORACE_PATH as GLOBAL_SCALER_PATH,
    NN_BATCH_SIZE, NN_LR, NN_WEIGHT_DECAY,
    FL_NUM_ROUNDS, FEDPROX_MU, NN_LOCAL_EPOCHS, SEED, XGB_PARAMS,
)
from nn_model   import DiabetesNet, train_one_epoch
from data_utils import (DiabetesDataset, compute_class_weight, load_brfss_clean,
                        set_params_from_numpy)
import fl_core
from fl_core import (DEVICE, seed_everything, build_node_loaders, run_federated,
                     fedavg_aggregate, fedprox_aggregate, fednova_aggregate,
                     equal_client_aggregate, run_scaffold)

CHUNK = 65536

# Every per-(model, seed) prediction vector is cached here so that downstream
# analyses — paired bootstrap deltas, subgroup decomposition, race-stratified
# fairness, AUPRC — run on CPU without retraining anything.
PRED_DIR = os.path.join(RESULTS_DIR, 'preds_multiseed')


def cache(name, arr):
    np.save(os.path.join(PRED_DIR, f'{name}.npy'), np.asarray(arr, dtype=np.float32))


def scaffold_eta_local():
    """
    SCAFFOLD's local SGD learning rate, as selected by 37_scaffold_lr_sweep.py on
    held-out node validation AUC. Read from the sweep artifact rather than
    hardcoded so the number in the paper always traces to the run that chose it.
    """
    path = os.path.join(RESULTS_DIR, 'scaffold_lr_sweep.json')
    if not os.path.exists(path):
        sys.exit("Run 37_scaffold_lr_sweep.py first — SCAFFOLD's eta_local is not chosen yet.")
    with open(path) as f:
        return float(json.load(f)['selected_eta_local'])


@torch.no_grad()
def score(model, X):
    """Chunked sigmoid scoring — BRFSS is 1.28M rows, keep inside 8 GB VRAM."""
    model.eval()
    out = []
    for i in range(0, len(X), CHUNK):
        out.append(torch.sigmoid(
            model(torch.FloatTensor(X[i:i + CHUNK]).to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def params_to_model(params, buffers=None):
    """Rebuild a scoreable model. Buffers matter: without the aggregated BN
    running statistics the model scores in eval() with untrained ones, which
    cost FedAvg 0.005 external AUC (results/bn_buffer_check.json)."""
    m = fl_core.new_net().to(DEVICE)
    set_params_from_numpy(m, params)
    fl_core.set_buffers_from_numpy(m, buffers)
    return m


def fairness_gap(y, p, age):
    """young(18-39) AUC - elderly(60+) AUC. Positive = worse for elderly."""
    yo = (age >= 18) & (age <= 39)
    el = age >= 60
    a_y = roc_auc_score(y[yo], p[yo])
    a_e = roc_auc_score(y[el], p[el])
    return float(a_y - a_e), float(a_y), float(a_e)


def train_centralised(X_tr, y_tr, seed):
    """Plain centralised DiabetesNet, same budget as the FL runs see locally."""
    seed_everything(seed)
    pos_w     = compute_class_weight(y_tr)
    model     = fl_core.new_net().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
    opt       = torch.optim.AdamW(model.parameters(), lr=NN_LR,
                                  weight_decay=NN_WEIGHT_DECAY)
    dl  = DataLoader(DiabetesDataset(X_tr, y_tr), batch_size=NN_BATCH_SIZE, shuffle=True)
    amp = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    for _ in range(FL_NUM_ROUNDS):
        train_one_epoch(model, dl, opt, criterion, DEVICE,
                        proximal_mu=0.0, global_params=None, scaler=amp)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    # Lets a later run extend the seed set without recomputing the ones already
    # cached: --seed-start 5 --seeds 5 adds seeds 47..51 alongside 42..46.
    ap.add_argument('--seed-start', type=int, default=0,
                    help='offset from SEED for the first seed in this run')
    args = ap.parse_args()
    seed_list = [SEED + args.seed_start + i for i in range(args.seeds)]

    scaler = joblib.load(GLOBAL_SCALER_PATH)
    os.makedirs(PRED_DIR, exist_ok=True)
    ETA_SCAFFOLD = scaffold_eta_local()
    print(f"SCAFFOLD eta_local = {ETA_SCAFFOLD} (from scaffold_lr_sweep.json)")

    # ── Common external test set ──────────────────────────────────────────────
    df_b, X_b_raw, y_b = load_brfss_clean(feature_cols=FEATURE_COLS)
    X_b    = scaler.transform(X_b_raw)
    age_b  = df_b['RIDAGEYR'].values
    # Pre-imputation race codes, for evaluation strata only (NaN = Unknown).
    race_b = df_b['RIDRETH3_RAW'].values
    print(f"Common BRFSS test set: n={len(y_b):,}")

    # Shared across every model and seed — written once.
    cache('_y_brfss', y_b)
    cache('_age_brfss', age_b)
    cache('_race_brfss', race_b)

    # ── Training pools ────────────────────────────────────────────────────────
    df_fed  = pd.concat([pd.read_csv(p) for p in NODE_PATHS], ignore_index=True)
    X_fed   = scaler.transform(df_fed[FEATURE_COLS].values.astype(np.float32))
    y_fed   = df_fed[TARGET_COL].values.astype(np.float32)

    df_pool = pd.read_csv(CENTRALISED_PATH)
    X_pool  = scaler.transform(df_pool[FEATURE_COLS].values.astype(np.float32))
    y_pool  = df_pool[TARGET_COL].values.astype(np.float32)
    print(f"Federated cohort: n={len(y_fed):,}   Full NHANES pool: n={len(y_pool):,}")

    records = []

    for seed in seed_list:
        print(f"\n{'='*66}\n  SEED {seed}\n{'='*66}")
        seed_everything(seed)

        loaders, n_samples = build_node_loaders(seed=seed)
        # Clean internal set for FL = concatenated held-out node val splits
        X_val = np.concatenate([dl_v.dataset.X.numpy() for (_, dl_v, _, _) in loaders])
        y_val = np.concatenate([dl_v.dataset.y.numpy() for (_, dl_v, _, _) in loaders])
        # One held-out internal set shared by all four FL strategies at this seed.
        cache(f'_y_internal_fl_seed{seed}', y_val)

        fl_specs = [
            ('FedAvg',        fedavg_aggregate,        0.0),
            ('FedProx',       fedprox_aggregate,       FEDPROX_MU),
            ('FedNova',       fednova_aggregate,       0.0),
            ('FedAvg-EqualW', equal_client_aggregate,  0.0),   # T1.3 attribution
            ('SCAFFOLD',      None,                    0.0),   # local SGD, see fl_core
        ]

        for name, agg, mu in fl_specs:
            if name == 'SCAFFOLD':
                r = run_scaffold(loaders, n_samples, eta_local=ETA_SCAFFOLD,
                                 seed=seed, verbose=False)
            else:
                r = run_federated(name, agg, loaders, n_samples, proximal_mu=mu,
                                  seed=seed, save_weights=False, verbose=False)
            m = params_to_model(r['global_params'], r.get('global_buffers'))
            p_val  = score(m, X_val)
            p_pool = score(m, X_pool)
            p_ext  = score(m, X_b)
            a_val  = float(roc_auc_score(y_val,  p_val))
            a_pool = float(roc_auc_score(y_pool, p_pool))
            a_ext  = float(roc_auc_score(y_b,    p_ext))
            fg, ay, ae = fairness_gap(y_b, p_ext, age_b)
            cache(f'{name}_seed{seed}_external', p_ext)
            cache(f'{name}_seed{seed}_internal', p_val)
            records.append(dict(model=name, seed=seed,
                                internal_val=a_val, internal_pool=a_pool,
                                external=a_ext,
                                gap_val=a_val - a_ext, gap_pool=a_pool - a_ext,
                                fair_gap=fg, auc_young=ay, auc_elderly=ae))
            print(f"  {name:<15} val={a_val:.4f} pool={a_pool:.4f} ext={a_ext:.4f} "
                  f"gap_val={a_val-a_ext:+.4f} fair={fg:+.4f}")

        # ── Centralised controls ──────────────────────────────────────────────
        for name, Xc, yc in [('Centralised-matched', X_fed,  y_fed),
                             ('Centralised-full',    X_pool, y_pool)]:
            Xtr, Xte, ytr, yte = train_test_split(
                Xc, yc, test_size=0.20, stratify=yc, random_state=seed)
            m = train_centralised(Xtr, ytr, seed)
            p_int = score(m, Xte)
            a_val = float(roc_auc_score(yte, p_int))
            p_ext = score(m, X_b)
            a_ext = float(roc_auc_score(y_b, p_ext))
            fg, ay, ae = fairness_gap(y_b, p_ext, age_b)
            cache(f'{name}_seed{seed}_external', p_ext)
            cache(f'{name}_seed{seed}_internal', p_int)
            cache(f'_y_internal_{name}_seed{seed}', yte)
            records.append(dict(model=name, seed=seed,
                                internal_val=a_val, internal_pool=None,
                                external=a_ext,
                                gap_val=a_val - a_ext, gap_pool=None,
                                fair_gap=fg, auc_young=ay, auc_elderly=ae))
            print(f"  {name:<15} val={a_val:.4f} ext={a_ext:.4f} "
                  f"gap_val={a_val-a_ext:+.4f} fair={fg:+.4f}")

        # ── XGBoost ───────────────────────────────────────────────────────────
        import xgboost as xgb
        Xtr, Xte, ytr, yte = train_test_split(
            X_pool, y_pool, test_size=0.20, stratify=y_pool, random_state=seed)
        params = dict(XGB_PARAMS); params['random_state'] = seed
        clf = xgb.XGBClassifier(**params)
        clf.fit(Xtr, ytr)
        p_int = clf.predict_proba(Xte)[:, 1]
        a_val = float(roc_auc_score(yte, p_int))
        p_ext = clf.predict_proba(X_b)[:, 1]
        a_ext = float(roc_auc_score(y_b, p_ext))
        fg, ay, ae = fairness_gap(y_b, p_ext, age_b)
        cache(f'XGBoost_seed{seed}_external', p_ext)
        cache(f'XGBoost_seed{seed}_internal', p_int)
        cache(f'_y_internal_XGBoost_seed{seed}', yte)
        records.append(dict(model='XGBoost', seed=seed,
                            internal_val=a_val, internal_pool=None,
                            external=a_ext, gap_val=a_val - a_ext, gap_pool=None,
                            fair_gap=fg, auc_young=ay, auc_elderly=ae))
        print(f"  {'XGBoost':<15} val={a_val:.4f} ext={a_ext:.4f} "
              f"gap_val={a_val-a_ext:+.4f} fair={fg:+.4f}")

    # ── Summarise ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    summary = {}
    for model, g in df.groupby('model'):
        summary[model] = {
            k: {'mean': round(float(g[k].mean()), 4),
                'sd':   round(float(g[k].std(ddof=1)), 4)}
            for k in ['internal_val', 'external', 'gap_val', 'fair_gap',
                      'auc_young', 'auc_elderly']
        }

    print(f"\n{'='*78}")
    print(f"  {'model':<20}{'ext AUC':>16}{'gen gap':>18}{'fairness gap':>18}")
    print(f"{'='*78}")
    for model in ['XGBoost', 'Centralised-full', 'Centralised-matched',
                  'FedNova', 'FedProx', 'FedAvg', 'FedAvg-EqualW', 'SCAFFOLD']:
        if model not in summary:
            continue
        s = summary[model]
        print(f"  {model:<20}"
              f"{s['external']['mean']:.4f}±{s['external']['sd']:.4f}  "
              f"{s['gap_val']['mean']:+.4f}±{s['gap_val']['sd']:.4f}  "
              f"{s['fair_gap']['mean']:+.4f}±{s['fair_gap']['sd']:.4f}")
    print(f"{'='*78}")

    tag = '' if args.seed_start == 0 else f'_from{seed_list[0]}'
    out = os.path.join(RESULTS_DIR, f'multiseed_all_models{tag}.json')
    with open(out, 'w') as f:
        json.dump({'n_seeds': args.seeds, 'seeds': seed_list,
                   'n_brfss': int(len(y_b)), 'runs': records, 'summary': summary},
                  f, indent=2)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()
