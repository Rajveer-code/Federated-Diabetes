"""
nn_model.py — 3-Layer PyTorch Neural Network for Diabetes Prediction
=====================================================================
Architecture: Input(8) → Dense(64,BN,ReLU,Drop) → Dense(32,BN,ReLU,Drop)
                       → Dense(16,BN,ReLU,Drop) → Dense(1)

Works standalone AND inside Flower federated clients.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiabetesNet(nn.Module):
    """
    3-layer feedforward network with BatchNorm + Dropout.
    Uses BCEWithLogitsLoss during training (outputs raw logits).
    """

    def __init__(
        self,
        input_dim:   int        = 8,
        hidden_dims: List[int]  = [64, 32, 16],
        dropout:     float      = 0.3,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        in_dim = input_dim
        for i, h in enumerate(hidden_dims):
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout if i < len(hidden_dims)-1 else dropout*0.6),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits. Use BCEWithLogitsLoss during training."""
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in [0,1]. Use during evaluation."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_one_epoch(
    model:         DiabetesNet,
    dataloader,
    optimizer:     torch.optim.Optimizer,
    criterion:     nn.Module,
    device:        torch.device,
    proximal_mu:   float = 0.0,
    global_params: list  = None,
) -> float:
    """
    Train for one epoch. Returns mean loss.
    proximal_mu > 0 → FedProx: adds ||w - w_global||² penalty.
    """
    model.train()
    total_loss, n_batches = 0.0, 0

    for X_b, y_b in dataloader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)

        if proximal_mu > 0.0 and global_params is not None:
            prox = sum(
                (p - g.to(device)).pow(2).sum()
                for p, g in zip(model.parameters(), global_params)
            )
            loss = loss + (proximal_mu / 2.0) * prox

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(
    model:     DiabetesNet,
    dataloader,
    criterion: nn.Module,
    device:    torch.device,
):
    """
    Evaluate model on a DataLoader.
    Returns: (mean_loss, y_true_np, y_prob_np)
    """
    model.eval()
    losses, probs, labels = [], [], []

    for X_b, y_b in dataloader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        losses.append(criterion(logits, y_b).item())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y_b.cpu().numpy())

    return (
        float(np.mean(losses)),
        np.concatenate(labels),
        np.concatenate(probs),
    )
