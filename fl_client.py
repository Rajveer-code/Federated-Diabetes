"""
fl_client.py — Flower Federated Learning Client
=================================================
Implements fl.client.NumPyClient for the DiabetesNet.
Supports FedAvg (mu=0) and FedProx (mu>0) transparently.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import NDArrays, Scalar

from nn_model   import DiabetesNet, train_one_epoch, evaluate_model
from data_utils import (load_node_data, get_dataloaders, compute_class_weight,
                         get_params_as_numpy, set_params_from_numpy, get_device)
from config_paths import FL_NUM_ROUNDS


class DiabetesClient(fl.client.NumPyClient):
    """
    Flower client for federated diabetes prediction.
    One instance per hospital node.
    """

    def __init__(
        self,
        node_id:      int,
        data_path:    str,
        local_epochs: int   = 5,
        batch_size:   int   = 64,
        lr:           float = 1e-3,
        proximal_mu:  float = 0.0,
        seed:         int   = 42,
    ):
        self.node_id      = node_id
        self.local_epochs = local_epochs
        self.proximal_mu  = proximal_mu
        self.device       = get_device()

        # Load + split + scale data
        X_tr, y_tr, X_val, y_val, self.scaler = load_node_data(
            data_path, val_size=0.2, seed=seed
        )
        self.train_dl, self.val_dl = get_dataloaders(
            X_tr, y_tr, X_val, y_val, batch_size=batch_size
        )
        self.n_train = len(X_tr)
        self.n_val   = len(X_val)

        # Model + loss + optimiser
        pos_weight     = compute_class_weight(y_tr)
        self.model     = DiabetesNet().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=FL_NUM_ROUNDS, eta_min=1e-6
        )
        # AMP GradScaler — active on CUDA, disabled on CPU
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.device.type == 'cuda' else None
        )

        print(f"    Client {node_id}: n_train={self.n_train} | "
              f"n_val={self.n_val} | pos_w={pos_weight:.2f} | mu={proximal_mu}")

    # ── Flower interface ────────────────────────────────────────────────────
    def get_parameters(self, config: Dict) -> NDArrays:
        return get_params_as_numpy(self.model)

    def set_parameters(self, params: NDArrays):
        set_params_from_numpy(self.model, params)

    def fit(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[NDArrays, int, Dict]:
        """Receive global weights -> train locally -> return updated weights."""
        self.set_parameters(parameters)

        # Cache global weights for FedProx proximal term
        global_params = None
        if self.proximal_mu > 0.0:
            global_params = [p.detach().clone() for p in self.model.parameters()]

        losses = []
        for _ in range(self.local_epochs):
            loss = train_one_epoch(
                self.model, self.train_dl, self.optimizer,
                self.criterion, self.device,
                self.proximal_mu, global_params,
                scaler=self.scaler,
            )
            losses.append(loss)
        self.scheduler.step()

        return self.get_parameters(config), self.n_train, {
            'train_loss': float(np.mean(losses)),
            'node_id':    self.node_id,
        }

    def evaluate(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[float, int, Dict]:
        """Receive global weights -> evaluate on local val set -> return AUC."""
        self.set_parameters(parameters)
        val_loss, y_true, y_prob = evaluate_model(
            self.model, self.val_dl, self.criterion, self.device
        )
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = 0.0

        return float(val_loss), self.n_val, {
            'auc':     auc,
            'node_id': self.node_id,
        }
