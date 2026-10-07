"""Experiment logging (docs/RULES.md §8): run IDs, environment capture, run directories, runs index, ingest.

This module must not import from ``ecgaug.data``, which imports it.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from ecgaug import paths

RUN_ID_RE = re.compile(
    r"^p(?P<phase>\d+)-m(?P<module>\d+)-(?P<variant>[A-Za-z0-9]+?)(?:\.(?P<tag>.+?))?"
    r"-s(?P<seed>\d+)-(?P<sha>[0-9a-f]{7})-(?P<stamp>\d{12})$"
)
REQUIRED_RUN_FILES = ("config.yaml", "env.json", "train_log.csv", "metrics_val.json")
RUNS_INDEX_COLUMNS = [
    "run_id", "phase", "module", "variant", "tag", "seed", "status",
    "selection_metric", "selection_value", "rare_macro_f1", "wall_clock_s", "git_sha",
]  # fmt: skip
_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "torch", "wfdb", "imbalanced-learn", "captum", "dtaidistance")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_sha(short: bool = False) -> str:
    """Commit SHA of the code checkout, or zeros when git is unavailable."""
    sha = _git("rev-parse", "HEAD") or "0" * 40
    return sha[:7] if short else sha


OUTPUT_DIRS = ("runs/", "results/")


def git_dirty() -> bool:
    """True if code, configs or docs differ from the commit (or git is unavailable).

    Files under ``runs/`` and ``results/`` are outputs of earlier runs and do not count.
    """
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status is None:
        return True
    changed = [line[3:].split(" -> ")[-1].strip('"') for line in status.splitlines() if line.strip()]
    return any(not path.startswith(OUTPUT_DIRS) for path in changed)


def make_run_id(
    phase: int, module: int, variant: str, seed: int, tag: str | None = None, now: _dt.datetime | None = None
) -> str:
    """``p<phase>-m<module>-<variant>[.<tag>]-s<seed>-<gitsha7>-<yyyymmddHHMM>``."""
    stamp = (now or _dt.datetime.now()).strftime("%Y%m%d%H%M")
    label = f"{variant}.{tag}" if tag else variant
    run_id = f"p{phase}-m{module}-{label}-s{seed}-{git_sha(short=True)}-{stamp}"
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"Malformed run id {run_id!r}")
    return run_id


def parse_run_id(run_id: str) -> dict:
    match = RUN_ID_RE.match(run_id)
    if not match:
        raise ValueError(f"Malformed run id {run_id!r}")
    parts = match.groupdict()
    return {**parts, "phase": int(parts["phase"]), "module": int(parts["module"]), "seed": int(parts["seed"])}


def capture_env(device: str) -> dict:
    """Software and hardware of the current process, for ``env.json``."""
    import torch

    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    lock = paths.repo_root() / "uv.lock"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "lockfile_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
    }


def write_run(
    run_id: str,
    config: dict,
    env: dict,
    history: list[dict],
    metrics: dict,
    preds: dict[str, np.ndarray],
    state_dict: dict,
) -> Path:
    """Create ``runs/<run_id>/`` with the files docs/RULES.md §8 lists. Never overwrites a run."""
    import torch

    run_dir = paths.runs_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    with (run_dir / "train_log.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]) if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    (run_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(run_dir / "preds_val.npz", **preds)
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir()
    torch.save(state_dict, checkpoint)
    env = {**env, "checkpoint": {"path": "checkpoints/best.pt", "sha256": _sha256(checkpoint)}}
    (run_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    return run_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upsert_runs_index(row: dict) -> Path:
    """Insert or replace the row for ``row['run_id']`` in ``results/runs_index.csv``."""
    index = paths.results_dir() / "runs_index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if index.is_file():
        with index.open(newline="") as fh:
            rows = {r["run_id"]: r for r in csv.DictReader(fh)}
    rows[row["run_id"]] = {col: row.get(col, "") for col in RUNS_INDEX_COLUMNS}
    with index.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUNS_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))
    return index


def index_row(run_id: str, metrics: dict, wall_clock_s: float, status: str = "done") -> dict:
    """Build a runs-index row from a run's validation metrics."""
    parts = parse_run_id(run_id)
    return {
        "run_id": run_id,
        "phase": parts["phase"],
        "module": parts["module"],
        "variant": parts["variant"],
        "tag": parts["tag"] or "",
        "seed": parts["seed"],
        "status": status,
        "selection_metric": metrics.get("selection_metric", ""),
        "selection_value": metrics.get("selection_value", ""),
        "rare_macro_f1": metrics.get("rare_macro_f1", ""),
        "wall_clock_s": round(float(wall_clock_s), 1),
        "git_sha": parts["sha"],
    }


def _find_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("config.yaml") if RUN_ID_RE.match(p.parent.name))


def ingest(sources: list[Path]) -> dict:
    """Bring run directories (or zips of them) into ``runs/`` and update the runs index.

    A run that carries test predictions is refused: before ``/final-eval`` no such file may exist,
    and afterwards test predictions are produced locally, not ingested. An existing run is never
    overwritten; an identical copy is skipped and a differing one is reported as a conflict.
    """
    report: dict = {
        "ingested": [], "skipped_identical": [], "conflicts": [], "refused": [], "incomplete": [],
        "result_files": [], "result_conflicts": [],
    }  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        candidates: list[Path] = []
        roots: list[Path] = []
        for i, source in enumerate(sources):
            source = Path(source)
            if source.is_file() and zipfile.is_zipfile(source):
                target = Path(tmp) / f"zip{i}"
                with zipfile.ZipFile(source) as zf:
                    zf.extractall(target)
                roots.append(target)
            elif source.is_dir():
                roots.append(source)
            else:
                raise FileNotFoundError(f"{source} is neither a zip archive nor a directory")
        for root in roots:
            candidates += _find_run_dirs(root)

        for run_dir in candidates:
            run_id = run_dir.name
            if any(run_dir.rglob("preds_test*")) or any(run_dir.rglob("metrics_test*")):
                report["refused"].append(run_id)
                continue
            if any(not (run_dir / name).is_file() for name in REQUIRED_RUN_FILES):
                report["incomplete"].append(run_id)
                continue
            destination = paths.runs_dir() / run_id
            if destination.exists():
                same = (destination / "metrics_val.json").read_bytes() == (run_dir / "metrics_val.json").read_bytes()
                report["skipped_identical" if same else "conflicts"].append(run_id)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, destination)
            metrics = json.loads((destination / "metrics_val.json").read_text())
            env = json.loads((destination / "env.json").read_text())
            upsert_runs_index(index_row(run_id, metrics, env.get("wall_clock_s", 0.0)))
            report["ingested"].append(run_id)

        # Variant result files (results/validation/m<M>/<X>.json and notes) travel with the runs.
        for root in roots:
            for validation in root.rglob("validation"):
                if validation.parent.name != "results":
                    continue
                for src in sorted(p for p in validation.rglob("*") if p.is_file()):
                    relative = src.relative_to(validation)
                    dst = paths.results_dir() / "validation" / relative
                    if dst.exists():
                        if dst.read_bytes() != src.read_bytes():
                            report["result_conflicts"].append(str(relative))
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    report["result_files"].append(str(relative))
    return report
