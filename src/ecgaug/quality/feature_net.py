"""The frozen feature network of the FID-analog metric.

PLAN Phase 2 names the frozen ``cnn`` as the feature extractor. A separate, small network is used
instead: the phase1_input_representation probe CNN on the waveform alone, trained once with a fixed
recipe on the training beats of the dataset and frozen. It is small enough (about 80 KB) to be
committed under ``results/phase2_quality_metrics/``, so every machine scores synthetic beats with the
same weights (verified by SHA-256), and it does not depend on the classifiers being compared. The RR
features are left out because synthetic beats copy theirs from real beats. Recorded in DECISIONS.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ecgaug import paths
from ecgaug.models import build_model
from ecgaug.names import PHASE2_QUALITY_METRICS
from ecgaug.train.loop import TrainConfig, fit_classifier, resolve_device, set_seed


def feature_net_path(dataset_sha256: str) -> Path:
    """One feature network per dataset: ``results/phase2_quality_metrics/feature_net_<hash12>.pt``."""
    return paths.results_for(PHASE2_QUALITY_METRICS) / f"feature_net_{dataset_sha256[:12]}.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_feature_net(train: dict[str, np.ndarray], val: dict[str, np.ndarray], settings: dict, selection: str, device: str = "cpu"):
    """Train the probe CNN on waveforms only (no RR side input) with the fixed recipe in ``settings``."""
    set_seed(int(settings["seed"]))
    model = build_model(settings["model"], 4, 0)
    cfg = TrainConfig(lr=float(settings["lr"]), max_epochs=int(settings["max_epochs"]), patience=int(settings["patience"]),
                      batch_size=int(settings["batch_size"]), weight_decay=float(settings["weight_decay"]))  # fmt: skip
    strip = lambda part: {**part, "side": None}  # noqa: E731
    fit = fit_classifier(model, strip(train), strip(val), cfg, selection, seed=int(settings["seed"]), device=resolve_device(device))
    model.load_state_dict(fit.state_dict)
    return model.cpu().eval(), {"best_epoch": fit.best_epoch, "selection_value": fit.best_value, "epochs_run": fit.epochs_run}


def ensure_feature_net(train: dict, val: dict, settings: dict, selection: str, dataset_sha256: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Load the dataset's feature network if it exists (hash checked), else train and save it."""
    path = feature_net_path(dataset_sha256)
    meta_path = path.with_suffix(".json")
    if path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        return load_feature_net(path, meta["sha256"], settings), meta
    model, fit = train_feature_net(train, val, settings, selection, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta = {"path": str(path.relative_to(paths.repo_root())), "sha256": _sha256(path), "dataset_sha256": dataset_sha256,
            "settings": settings, "fit": fit}  # fmt: skip
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return model, meta


def load_feature_net(path: Path, expected_sha256: str, settings: dict) -> torch.nn.Module:
    """The frozen feature network, after checking the file's SHA-256."""
    path = Path(path)
    if not path.is_absolute():
        path = paths.repo_root() / path
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {actual[:12]}... does not match the recorded {expected_sha256[:12]}...")
    model = build_model(settings["model"], 4, 0)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
