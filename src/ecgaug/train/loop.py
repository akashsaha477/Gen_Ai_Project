"""Seeding and the classifier training loop with early stopping on a validation metric."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ecgaug.eval.metrics import evaluate
from ecgaug.eval.selection import selection_value


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, and request deterministic algorithms where available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    """'auto' picks CUDA if present, else CPU. MPS is used only when asked for by name."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    max_epochs: int = 40
    patience: int = 5
    batch_size: int = 256
    weight_decay: float = 0.01


@dataclass
class FitResult:
    best_epoch: int
    best_value: float
    best_metrics: dict
    val_probs: np.ndarray
    state_dict: dict
    epochs_run: int
    stopped_early: bool
    wall_clock_s: float
    history: list[dict] = field(default_factory=list)


@torch.no_grad()
def predict_proba(
    model: nn.Module, x: torch.Tensor, side: torch.Tensor | None, device: torch.device, batch_size: int = 2048
) -> np.ndarray:
    """Class probabilities ``[n, C]`` in evaluation mode."""
    model.eval()
    chunks = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        sb = side[start : start + batch_size].to(device) if side is not None else None
        chunks.append(torch.softmax(model(xb, sb), dim=1).cpu())
    return torch.cat(chunks).numpy()


def fit_classifier(
    model: nn.Module,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    cfg: TrainConfig,
    selection: str,
    seed: int,
    device: torch.device | None = None,
) -> FitResult:
    """Train with unweighted cross-entropy and AdamW; stop when the validation metric stalls.

    ``train`` and ``val`` hold ``X`` ``[n, L]``, ``y`` and optionally ``side`` ``[n, k]``. After every
    epoch the model is evaluated on validation; the checkpoint with the highest selection value is
    kept, and training stops after ``cfg.patience`` epochs without improvement.
    """
    device = device or torch.device("cpu")
    set_seed(seed)
    model.to(device)
    x_tr = torch.from_numpy(train["X"])
    y_tr = torch.from_numpy(train["y"].astype(np.int64))
    s_tr = torch.from_numpy(train["side"]) if train.get("side") is not None else None
    x_va = torch.from_numpy(val["X"])
    s_va = torch.from_numpy(val["side"]) if val.get("side") is not None else None
    y_va = val["y"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    shuffle = torch.Generator().manual_seed(seed)
    n = x_tr.shape[0]

    best_value, best_epoch, stale = -np.inf, 0, 0
    best_metrics: dict = {}
    best_probs = np.empty((0, 0))
    best_state: dict = {}
    history: list[dict] = []
    started = time.perf_counter()
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=shuffle)
        total_loss, seen = 0.0, 0
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            if idx.numel() < 2:  # batch norm needs at least two samples
                continue
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device)
            sb = s_tr[idx].to(device) if s_tr is not None else None
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.numel()
            seen += idx.numel()

        probs = predict_proba(model, x_va, s_va, device)
        metrics = evaluate(probs, y_va)
        value = selection_value(metrics, selection)
        if np.isnan(value):
            raise ValueError(f"Selection metric {selection!r} is undefined on the validation split")
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "val_selection": value,
                "val_rare_macro_f1": metrics["rare_macro_f1"],
                "val_f1_S": metrics["per_class"]["S"]["f1"],
                "val_f1_V": metrics["per_class"]["V"]["f1"],
                "val_f1_F": metrics["per_class"]["F"]["f1"],
                "val_n_recall": metrics["n_recall"],
                "elapsed_s": time.perf_counter() - started,
            }
        )
        if value > best_value:
            best_value, best_epoch, stale = value, epoch, 0
            best_metrics, best_probs = metrics, probs
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    return FitResult(
        best_epoch=best_epoch,
        best_value=float(best_value),
        best_metrics=best_metrics,
        val_probs=best_probs,
        state_dict=best_state,
        epochs_run=epoch,
        stopped_early=epoch < cfg.max_epochs,
        wall_clock_s=time.perf_counter() - started,
        history=history,
    )
