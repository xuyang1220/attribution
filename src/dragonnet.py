# src/dragonnet.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class DragonNet(nn.Module):
    """
    First-pass DragonNet-style model for binary treatment / binary outcome.

    Outputs:
        y0_hat_logit: logit for P(Y=1 | do(T=0), X)
        y1_hat_logit: logit for P(Y=1 | do(T=1), X)
        t_hat_logit:  logit for P(T=1 | X)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_shared_layers: int = 2,
        num_head_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_shared_layers < 1:
            raise ValueError("num_shared_layers must be >= 1.")
        if num_head_layers < 1:
            raise ValueError("num_head_layers must be >= 1.")

        self.shared_trunk = self._make_mlp(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_shared_layers,
            dropout=dropout,
        )

        self.y0_head = self._make_head(
            hidden_dim=hidden_dim,
            num_layers=num_head_layers,
            dropout=dropout,
        )
        self.y1_head = self._make_head(
            hidden_dim=hidden_dim,
            num_layers=num_head_layers,
            dropout=dropout,
        )
        self.t_head = self._make_head(
            hidden_dim=hidden_dim,
            num_layers=num_head_layers,
            dropout=dropout,
        )

        self._init_weights()

    @staticmethod
    def _make_mlp(
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = []
        cur_dim = in_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(cur_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            cur_dim = hidden_dim

        return nn.Sequential(*layers)

    @staticmethod
    def _make_head(
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = []
        cur_dim = hidden_dim

        for _ in range(max(0, num_layers - 1)):
            layers.extend(
                [
                    nn.Linear(cur_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            cur_dim = hidden_dim

        layers.append(nn.Linear(cur_dim, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.shared_trunk(x)

        y0_logit = self.y0_head(z).squeeze(-1)
        y1_logit = self.y1_head(z).squeeze(-1)
        t_logit = self.t_head(z).squeeze(-1)

        return {
            "y0_logit": y0_logit,
            "y1_logit": y1_logit,
            "t_logit": t_logit,
        }

    @torch.no_grad()
    def predict_uplift(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(x)

        p0 = torch.sigmoid(out["y0_logit"])
        p1 = torch.sigmoid(out["y1_logit"])
        e = torch.sigmoid(out["t_logit"])
        tau = p1 - p0

        return {
            "p0_hat": p0,
            "p1_hat": p1,
            "tau_hat": tau,
            "propensity_hat": e,
        }


def dragonnet_loss(
    outputs: Dict[str, torch.Tensor],
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    alpha_propensity: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    First-pass loss:
      - factual outcome BCE
      - treatment propensity BCE

    We do NOT add targeted regularization yet.
    """
    y0_logit = outputs["y0_logit"]
    y1_logit = outputs["y1_logit"]
    t_logit = outputs["t_logit"]

    treatment = treatment.float().view(-1)
    outcome = outcome.float().view(-1)

    factual_logit = treatment * y1_logit + (1.0 - treatment) * y0_logit

    outcome_loss = F.binary_cross_entropy_with_logits(factual_logit, outcome)
    propensity_loss = F.binary_cross_entropy_with_logits(t_logit, treatment)

    total_loss = outcome_loss + alpha_propensity * propensity_loss

    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "outcome_loss": float(outcome_loss.detach().cpu().item()),
        "propensity_loss": float(propensity_loss.detach().cpu().item()),
    }
    return total_loss, metrics


@dataclass
class TrainConfig:
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    alpha_propensity: float = 1.0
    grad_clip_norm: float | None = 5.0
    device: str = "cpu"
    verbose: bool = True


def make_tensor_dataset(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    device: str | None = None,
) -> TensorDataset:
    """
    Converts numpy arrays into a TensorDataset.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    t_t = torch.tensor(t, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    if device is not None:
        X_t = X_t.to(device)
        t_t = t_t.to(device)
        y_t = y_t.to(device)

    return TensorDataset(X_t, t_t, y_t)


def train_one_epoch(
    model: DragonNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> Dict[str, float]:
    model.train()

    running = {
        "loss": 0.0,
        "outcome_loss": 0.0,
        "propensity_loss": 0.0,
    }
    n_batches = 0

    for xb, tb, yb in loader:
        optimizer.zero_grad(set_to_none=True)

        outputs = model(xb)
        loss, batch_metrics = dragonnet_loss(
            outputs=outputs,
            treatment=tb,
            outcome=yb,
            alpha_propensity=config.alpha_propensity,
        )

        loss.backward()

        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

        optimizer.step()

        for k in running:
            running[k] += batch_metrics[k]
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Training loader has no batches.")

    return {k: v / n_batches for k, v in running.items()}


@torch.no_grad()
def evaluate_one_epoch(
    model: DragonNet,
    loader: DataLoader,
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()

    running = {
        "loss": 0.0,
        "outcome_loss": 0.0,
        "propensity_loss": 0.0,
    }
    n_batches = 0

    for xb, tb, yb in loader:
        outputs = model(xb)
        _, batch_metrics = dragonnet_loss(
            outputs=outputs,
            treatment=tb,
            outcome=yb,
            alpha_propensity=config.alpha_propensity,
        )

        for k in running:
            running[k] += batch_metrics[k]
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Validation loader has no batches.")

    return {k: v / n_batches for k, v in running.items()}


def fit_dragonnet(
    X_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    t_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = 128,
    num_shared_layers: int = 2,
    num_head_layers: int = 1,
    dropout: float = 0.1,
    config: TrainConfig | None = None,
) -> Tuple[DragonNet, list[Dict[str, float]]]:
    if config is None:
        config = TrainConfig()

    device = torch.device(config.device)

    model = DragonNet(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        num_shared_layers=num_shared_layers,
        num_head_layers=num_head_layers,
        dropout=dropout,
    ).to(device)

    train_ds = make_tensor_dataset(X_train, t_train, y_train)
    val_ds = make_tensor_dataset(X_val, t_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: list[Dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, config)
        val_metrics = evaluate_one_epoch(model, val_loader, config)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_outcome_loss": train_metrics["outcome_loss"],
            "train_propensity_loss": train_metrics["propensity_loss"],
            "val_loss": val_metrics["loss"],
            "val_outcome_loss": val_metrics["outcome_loss"],
            "val_propensity_loss": val_metrics["propensity_loss"],
        }
        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if config.verbose:
            print(
                f"[Epoch {epoch:02d}] "
                f"train_loss={row['train_loss']:.5f} "
                f"val_loss={row['val_loss']:.5f} "
                f"train_outcome={row['train_outcome_loss']:.5f} "
                f"val_outcome={row['val_outcome_loss']:.5f} "
                f"train_prop={row['train_propensity_loss']:.5f} "
                f"val_prop={row['val_propensity_loss']:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def predict_dragonnet(
    model: DragonNet,
    X: np.ndarray,
    batch_size: int = 4096,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    model.eval()
    model = model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(X_t, batch_size=batch_size, shuffle=False)

    p0_list = []
    p1_list = []
    tau_list = []
    e_list = []

    for xb in loader:
        xb = xb.to(device)
        pred = model.predict_uplift(xb)

        p0_list.append(pred["p0_hat"].cpu().numpy())
        p1_list.append(pred["p1_hat"].cpu().numpy())
        tau_list.append(pred["tau_hat"].cpu().numpy())
        e_list.append(pred["propensity_hat"].cpu().numpy())

    return {
        "p0_hat": np.concatenate(p0_list),
        "p1_hat": np.concatenate(p1_list),
        "tau_hat": np.concatenate(tau_list),
        "propensity_hat": np.concatenate(e_list),
    }