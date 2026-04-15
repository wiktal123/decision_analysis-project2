from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int = 4,
        hidden_sizes: tuple[int, ...] = (64, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


@dataclass
class TrainHistory:
    train_loss: List[float]
    test_loss: List[float]
    train_acc: List[float]
    test_acc: List[float]
    train_f1: List[float]
    test_f1: List[float]
    train_auc: List[float]
    test_auc: List[float]


def _metrics_from_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
) -> tuple[float, float, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    y_true_oh = np.eye(num_classes)[y_true]
    auc = roc_auc_score(y_true_oh, y_proba, multi_class="ovr", average="macro")
    return float(acc), float(f1), float(auc)


def _evaluate(
    model: NeuralNetwork,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    acc, f1, auc = _metrics_from_numpy(y_true, y_pred, y_prob, model.num_classes)
    return float(np.mean(losses)), acc, f1, auc


def train_neural_network(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 300,
    lr: float = 3e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    device: Optional[str] = None,
    early_stopping_patience: Optional[int] = 30,
    monitor: str = "test_auc",
    class_weights: Optional[np.ndarray] = None,
) -> TrainHistory:
    dev = (
        torch.device(device)
        if device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(dev)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
    )

    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=dev)
    else:
        w = None
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=epochs,
    )

    history = TrainHistory([], [], [], [], [], [], [], [])
    monitor_mode = "min" if "loss" in monitor else "max"
    best_metric = float("inf") if monitor_mode == "min" else float("-inf")
    best_epoch = -1
    best_state_dict = None
    no_improvement_epochs = 0

    for epoch in trange(epochs, desc="Training Neural Network"):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss, train_acc, train_f1, train_auc = _evaluate(
            model, train_loader, criterion, dev
        )
        test_loss, test_acc, test_f1, test_auc = _evaluate(
            model, test_loader, criterion, dev
        )
        history.train_loss.append(train_loss)
        history.test_loss.append(test_loss)
        history.train_acc.append(train_acc)
        history.test_acc.append(test_acc)
        history.train_f1.append(train_f1)
        history.test_f1.append(test_f1)
        history.train_auc.append(train_auc)
        history.test_auc.append(test_auc)

        monitored_values = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "train_auc": train_auc,
            "test_auc": test_auc,
        }
        current_metric = monitored_values[monitor]
        if monitor_mode == "min":
            improved = current_metric < best_metric
        else:
            improved = current_metric > best_metric

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            no_improvement_epochs = 0
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            no_improvement_epochs += 1

        if (
            early_stopping_patience is not None
            and no_improvement_epochs >= early_stopping_patience
        ):
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.__dict__["best_epoch_"] = best_epoch
    model.__dict__["best_metric_"] = best_metric
    return history
