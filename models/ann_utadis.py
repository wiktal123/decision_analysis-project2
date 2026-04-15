from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class CriterionLayerSpread(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        input_range: tuple[float, float] = (0.0, 1.0),
        normalize_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_criteria = num_criteria
        neg_range = (-input_range[0], -input_range[1])
        self.max_bias = max(neg_range)
        self.min_bias = min(neg_range)
        self.normalize_bias = normalize_bias
        self.bias = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.weight = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.min_w = 0.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for i in range(self.num_criteria):
                self.bias[:, i] = torch.linspace(
                    self.min_bias, self.max_bias, self.bias.shape[0]
                )
            self.weight.fill_(50.0)

    def compute_bias(self) -> torch.Tensor:
        if self.normalize_bias:
            return torch.clamp(self.bias, self.min_bias, self.max_bias)
        return self.bias

    def compute_weight(self) -> torch.Tensor:
        return F.softplus(self.weight) + self.min_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, self.num_criteria)
        return (x + self.compute_bias()) * self.compute_weight()


class CriterionLayerCombine(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        min_weight: float = 0.001,
    ) -> None:
        super().__init__()
        self.min_weight = min_weight
        self.weight = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -3.0, -2.0)

    def compute_weight(self) -> torch.Tensor:
        return F.softplus(self.weight) + self.min_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.compute_weight()).sum(1)


class MonotonicLayer(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
    ) -> None:
        super().__init__()
        self.criterion_layer_spread = CriterionLayerSpread(
            num_criteria=num_criteria,
            num_hidden_components=num_hidden_components,
        )
        self.criterion_layer_combine = CriterionLayerCombine(
            num_criteria=num_criteria,
            num_hidden_components=num_hidden_components,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.criterion_layer_spread(x)
        x = torch.sigmoid(x)
        x = self.criterion_layer_combine(x)
        return x


class Uta(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
    ) -> None:
        super().__init__()
        self.monotonic_layer = MonotonicLayer(
            num_criteria=num_criteria,
            num_hidden_components=num_hidden_components,
        )
        self.bn = nn.BatchNorm1d(1, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.monotonic_layer(x).sum(1).unsqueeze(1)
        if u.size(0) > 1:
            u = self.bn(u)
        return u.squeeze(1)


class OrderedThresholds(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = num_classes
        self.base = nn.Parameter(torch.tensor(0.0))
        self.deltas = nn.Parameter(torch.ones(num_classes - 2))

    def forward(self) -> torch.Tensor:
        if self.num_classes == 2:
            return self.base.unsqueeze(0)
        gaps = F.softplus(self.deltas)
        thresholds = torch.cat(
            [self.base.unsqueeze(0), self.base + torch.cumsum(gaps, dim=0)]
        )
        return thresholds


class ANNUTADIS(nn.Module):
    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int = 30,
        num_classes: int = 4,
        prob_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_criteria = num_criteria
        self.num_classes = num_classes
        self.prob_scale_logit = nn.Parameter(torch.tensor(prob_scale).log())
        self.uta = Uta(num_criteria, num_hidden_components)
        self.thresholds = OrderedThresholds(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        utility = self.uta(x)
        taus = self.thresholds()
        return {"utility": utility, "thresholds": taus}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        utility = out["utility"]
        taus = out["thresholds"]
        return torch.bucketize(utility, taus)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        utility = out["utility"].unsqueeze(1)
        taus = out["thresholds"].unsqueeze(0)
        scale = torch.exp(self.prob_scale_logit)
        p_gt = torch.sigmoid(scale * (utility - taus))

        probs = torch.zeros(
            utility.size(0), self.num_classes, device=x.device, dtype=x.dtype
        )
        probs[:, 0] = 1.0 - p_gt[:, 0]
        for idx in range(1, self.num_classes - 1):
            probs[:, idx] = p_gt[:, idx - 1] - p_gt[:, idx]
        probs[:, -1] = p_gt[:, -1]
        probs = torch.clamp(probs, 1e-7, 1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


def boundary_regret_loss(
    utility: torch.Tensor,
    y: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    utility = utility.view(-1)
    loss = torch.tensor(0.0, device=utility.device, dtype=utility.dtype)
    for boundary_idx, tau in enumerate(thresholds):
        positive = (y > boundary_idx).float()
        negative = (y <= boundary_idx).float()
        loss_pos = F.relu(tau - utility) * positive
        loss_neg = F.relu(utility - tau) * negative
        loss = loss + (loss_pos + loss_neg).mean()
    return loss / len(thresholds)


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
    model: ANNUTADIS,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    losses = []
    preds: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            prob_tensor = model.predict_proba(xb)
            loss = F.nll_loss(torch.log(prob_tensor + 1e-8), yb)
            losses.append(loss.item())

            pred = model.predict(xb).detach().cpu().numpy()
            prob = prob_tensor.detach().cpu().numpy()
            label = yb.detach().cpu().numpy()
            preds.append(pred)
            probs.append(prob)
            labels.append(label)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    acc, f1, auc = _metrics_from_numpy(y_true, y_pred, y_prob, model.num_classes)
    return float(np.mean(losses)), acc, f1, auc


def train_ann_utadis(
    model: ANNUTADIS,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 500,
    lr: float = 5e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    device: Optional[str] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    monitor: str = "test_loss",
    checkpoint_path: Optional[str] = None,
    save_each_epoch: bool = False,
    load_best_at_end: bool = True,
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99)
    )
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
    ckpt_dir = Path(checkpoint_path).parent if checkpoint_path else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in trange(epochs, desc="Training ANN-UTADIS"):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad()
            prob_tensor = model.predict_proba(xb)
            loss = F.nll_loss(torch.log(prob_tensor + 1e-8), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss, train_acc, train_f1, train_auc = _evaluate(model, train_loader, dev)
        test_loss, test_acc, test_f1, test_auc = _evaluate(model, test_loader, dev)
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
        if monitor not in monitored_values:
            raise ValueError(
                f"Unknown monitor '{monitor}'. Valid values: {list(monitored_values.keys())}"
            )

        current_metric = monitored_values[monitor]
        if monitor_mode == "min":
            improved = current_metric < (best_metric - early_stopping_min_delta)
        else:
            improved = current_metric > (best_metric + early_stopping_min_delta)

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            no_improvement_epochs = 0
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            if checkpoint_path:
                torch.save(
                    {
                        "epoch": epoch,
                        "monitor": monitor,
                        "best_metric": best_metric,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
        else:
            no_improvement_epochs += 1

        if save_each_epoch and checkpoint_path:
            epoch_path = Path(checkpoint_path).with_name(
                f"{Path(checkpoint_path).stem}_epoch_{epoch + 1}{Path(checkpoint_path).suffix}"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "monitor": monitor,
                    "metric": current_metric,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                epoch_path,
            )

        if (
            early_stopping_patience is not None
            and no_improvement_epochs >= early_stopping_patience
        ):
            break

    if load_best_at_end and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.__dict__["best_epoch_"] = best_epoch
    model.__dict__["best_metric_"] = best_metric
    return history


def marginal_value_curves(
    model: ANNUTADIS,
    num_criteria: int,
    points: int = 201,
    base_value: float = 0.5,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    dev = torch.device(device) if device else next(model.parameters()).device
    model.eval()
    xs = np.linspace(0.0, 1.0, points)
    curves = np.zeros((points, num_criteria), dtype=np.float64)

    with torch.no_grad():
        mono = model.uta.monotonic_layer
        for ci in range(num_criteria):
            for i, x in enumerate(xs):
                sample = torch.full(
                    (1, num_criteria), base_value, dtype=torch.float32, device=dev
                )
                sample[0, ci] = float(x)
                contributions = mono(sample)[0].detach().cpu().numpy()
                curves[i, ci] = contributions[ci]
            curves[:, ci] = curves[:, ci] - curves[0, ci]
    return xs, curves
