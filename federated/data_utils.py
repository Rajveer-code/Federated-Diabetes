"""
data_utils.py — Shared Data Loading & PyTorch Utilities
=========================================================
Imported by ALL scripts. Handles:
  - Loading node CSVs
  - Train/val splitting + scaling
  - PyTorch Dataset + DataLoaders
  - Helper functions for FL parameter passing
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List

FEATURE_COLS = [
    'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'BMXBMI',
    'SMOKING', 'PHYS_ACTIVITY', 'HEART_ATTACK', 'STROKE'
]
TARGET_COL = 'DIABETES'

# BRFSS _RACE -> coarse evaluation strata. Used ONLY to group rows when reporting
# subgroup performance, never as a model input, so a mislabel would rename a
# stratum rather than corrupt a prediction.
# Codes follow the BRFSS _RACE codebook (1=White NH, 2=Black NH, 3=AI/AN NH,
# 4=Asian NH, 5=NHPI NH, 6=Other NH, 7=Multiracial NH, 8=Hispanic, 9=Refused);
# the observed diabetes prevalences are consistent with it (Black 19.5%,
# AI/AN 19.9%, White 12.4%, Asian 9.6%). VERIFY against the codebook for the
# survey years actually used before this reaches the manuscript.
BRFSS_RACE_GROUPS = {
    'White':    [1.0],
    'Black':    [2.0],
    'Hispanic': [8.0],
    'Asian':    [4.0],
    'Other':    [3.0, 5.0, 6.0, 7.0],
    'Unknown':  [9.0],          # plus NaN, handled at the call site
}

# BRFSS -> NHANES column mapping. Canonical definition; do not redefine locally.
BRFSS_COL_MAP = {
    'Age'                 : 'RIDAGEYR',
    'Gender'              : 'RIAGENDR',
    'Race_Ethnicity'      : 'RIDRETH3',
    'BMI'                 : 'BMXBMI',
    'Smoking_Status'      : 'SMOKING',
    'Physical_Activity'   : 'PHYS_ACTIVITY',
    'History_Heart_Attack': 'HEART_ATTACK',
    'History_Stroke'      : 'STROKE',
    'Diabetes_Outcome'    : 'DIABETES',
}


def load_brfss_clean(
    path: Optional[str] = None,
    verbose: bool = True,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Canonical BRFSS load + clean. THE single BRFSS entry point for every script.

    Why this exists: 07_external_validation.py and run_matched_subset_external.py
    previously implemented this independently and silently diverged on two axes —
    (a) missing-data policy: 07 imputed and kept 1,282,897 rows, while the matched
    -subset script used complete-case deletion and kept 720,161 (43.9% dropped,
    non-randomly); (b) sex encoding: 07 remapped BRFSS 0/1 to NHANES 1/2, the
    matched-subset script did not, feeding a miscoded feature to a model trained
    on 1/2. The two scripts' external metrics were therefore not comparable, which
    invalidated the matched-control comparison the paper's headline claim rests on.

    Behaviour here is a faithful extraction of 07_external_validation.py, so the
    already-reported main-model numbers are unchanged; this only brings the other
    callers into line with them.

    Returns: (cleaned_df, X_unscaled, y)  — X is NOT scaled; apply the global
    NHANES scaler with .transform() at the call site, never fit_transform.
    """
    if path is None:
        from config_paths import BRFSS_PATH as _default_path
        path = _default_path

    if verbose:
        print(f"  Loading BRFSS: {path}")
    df = pd.read_csv(path)
    raw_n = len(df)

    df = df.rename(columns=BRFSS_COL_MAP)

    # Sex encoding: BRFSS 0=Female,1=Male -> NHANES 1=Male,2=Female.
    # Guarded so a file already in NHANES coding passes through untouched.
    if df['RIAGENDR'].max() <= 1.0:
        df['RIAGENDR'] = df['RIAGENDR'].map({1.0: 1.0, 0.0: 2.0})
        if verbose:
            print("  Sex remapped: BRFSS 0/1 -> NHANES 1/2")

    # Rows without the outcome are unusable; features are imputed, not dropped.
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    cols = FEATURE_COLS if feature_cols is None else feature_cols

    # Keep the pre-imputation race codes for evaluation grouping. 34.7% of BRFSS
    # rows have race missing; when race is not a model input those rows must stay
    # identifiable as their own stratum instead of being folded into the mode.
    df['RIDRETH3_RAW'] = df['RIDRETH3']

    # Impute only what the model actually consumes.
    for col in ['BMXBMI', 'RIDAGEYR']:
        if col in cols:
            df[col] = df[col].fillna(df[col].median())
    for col in ['RIAGENDR', 'RIDRETH3', 'SMOKING',
                'PHYS_ACTIVITY', 'HEART_ATTACK', 'STROKE']:
        if col in cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.reset_index(drop=True)

    X = df[cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    if verbose:
        print(f"  BRFSS raw: {raw_n:,} -> after cleaning: {len(df):,} rows "
              f"(dropped {raw_n - len(df):,} missing outcome)")
        print(f"  Diabetes prevalence: {y.mean():.4f}")

    return df, X, y


class DiabetesDataset(Dataset):
    """PyTorch Dataset wrapping numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_node_data(
    path: str,
    val_size: float = 0.2,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
    seed: int = 42,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Load a node CSV, split into train/val, scale features.
    Returns: X_train, y_train, X_val, y_val, fitted_scaler

    feature_cols defaults to FEATURE_COLS; the multi-seed pipeline passes
    FEATURE_COLS_NORACE. The split is taken before scaling and does not depend on
    the column list, so a run stays comparable across feature sets at a given seed.
    """
    cols = FEATURE_COLS if feature_cols is None else feature_cols
    df  = pd.read_csv(path)
    X   = df[cols].values.astype(np.float32)
    y   = df[TARGET_COL].values.astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=seed
    )

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
    else:
        X_tr  = scaler.transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)

    return X_tr, y_tr, X_val, y_val, scaler


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    batch_size:  int  = 64,
    pin_memory:  bool = False,
    num_workers: int  = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wrap numpy arrays into PyTorch DataLoaders.
    pin_memory=True + non_blocking GPU transfers are the key speedup on RTX 40xx.
    num_workers=0 is safest on Windows (avoids subprocess-spawn overhead for small datasets).
    """
    train_dl = DataLoader(
        DiabetesDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=pin_memory, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_dl = DataLoader(
        DiabetesDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, drop_last=False,
        pin_memory=pin_memory, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    return train_dl, val_dl


def compute_class_weight(y: np.ndarray) -> float:
    """pos_weight for BCEWithLogitsLoss — mirrors XGBoost scale_pos_weight."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def get_params_as_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model weights as list of numpy arrays (for Flower)."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_params_from_numpy(model: torch.nn.Module, params: List[np.ndarray]):
    """Set model weights from list of numpy arrays (from Flower server)."""
    with torch.no_grad():
        for p, new_val in zip(model.parameters(), params):
            p.copy_(torch.tensor(new_val, dtype=torch.float32))


def get_device() -> torch.device:
    """Auto-detect GPU (RTX 4060) or fall back to CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True   # auto-tune cuDNN kernels
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024**3
        print(f"  Device: GPU — {props.name}  ({vram:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print("  Device: CPU  (CUDA not available — scripts run on CPU)")
        print("  TIP: Install CUDA PyTorch for ~10x speedup on your RTX 4060:")
        print("       pip uninstall torch torchvision torchaudio -y")
        print("       pip install torch torchvision torchaudio "
              "--index-url https://download.pytorch.org/whl/cu124")
    return device
