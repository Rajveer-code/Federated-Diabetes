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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Load a node CSV, split into train/val, scale features.
    Returns: X_train, y_train, X_val, y_val, fitted_scaler
    """
    df  = pd.read_csv(path)
    X   = df[FEATURE_COLS].values.astype(np.float32)
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
<<<<<<< HEAD
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap numpy arrays into PyTorch DataLoaders."""
    train_dl = DataLoader(
        DiabetesDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_dl = DataLoader(
        DiabetesDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, drop_last=False
=======
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
>>>>>>> 435718c297f04a6b74b12d2ac00504407237e06b
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
<<<<<<< HEAD
        print(f"  Device: GPU — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("  Device: CPU (CUDA not available)")
=======
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
>>>>>>> 435718c297f04a6b74b12d2ac00504407237e06b
    return device
