"""What every generator shares: the train-only rule, class-balanced batches, the step loop with
checkpoints at each quarter of the step budget, and the divergence checks (PLAN Phase 3, items 1 and 6).

A generator learns beat waveforms ``X`` ``[n, L]`` of the frozen representation (per-beat z-scored
windows) conditioned on the class. It never models RR features: synthetic beats get theirs from the
class-conditional empirical distribution of training beats (PREREGISTRATION §5), in ``bank.py``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ecgaug.data.records import CLASS_INDEX, CLASSES
from ecgaug.train.loop import set_seed

GENERATED_CLASSES: tuple[str, ...] = ("S", "V", "F")  # the classes a synthetic bank holds
CHECKPOINT_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
MAX_ABS_LOSS = 1e6  # a loss beyond this, or a non-finite one, counts as divergence

# Called at each checkpoint with (checkpoint number from 1, step, generator); returns what to log.
CheckpointFn = Callable[[int, int, "BeatGenerator"], dict]


class TrainOnlyError(ValueError):
    """A generator or oversampler was given beats from outside the training split."""


def assert_train_only(part: dict[str, np.ndarray], val_records: tuple[int, ...] | list[int] = ()) -> None:
    """Raise unless every beat comes from the training split (CLAUDE.md rule 1, PLAN Phase 3 item 6)."""
    if "split" not in part or "record" not in part:
        raise TrainOnlyError("Generator data must carry the 'split' and 'record' fields so the train-only rule can be checked")
    outside = np.asarray(part["split"]) != "train"
    if outside.any():
        raise TrainOnlyError(
            f"{int(outside.sum())} of {outside.size} beats are not from the training split; "
            "generators and oversamplers see training records only"
        )
    records = np.asarray(part["record"])
    leaked = np.isin(records, np.asarray(list(val_records), dtype=np.int64))
    if leaked.any():
        raise TrainOnlyError(f"Beats from validation records {sorted(set(records[leaked].tolist()))} were passed as training beats")


def training_beats(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every contract field of the training split: the only beats a generator may learn from."""
    mask = np.asarray(arrays["split"]) == "train"
    return {name: np.asarray(values)[mask] for name, values in arrays.items()}


class ClassBatches:
    """Batches of training beats with the same number of beats of each requested class.

    Beats are drawn with replacement within a class, so a rare class is seen as often as a common one.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> None:
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = np.asarray(y, dtype=np.int64)
        self.pools = {int(c): np.flatnonzero(self.y == c) for c in np.unique(self.y)}
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def draw(self, class_ids: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [CLASSES[c] for c in class_ids if c not in self.pools]
        if missing:
            raise ValueError(f"No training beats of class {missing}")
        per_class = max(1, self.batch_size // len(class_ids))
        idx = np.concatenate([self.rng.choice(self.pools[c], size=per_class, replace=True) for c in class_ids])
        return self.x[idx], torch.from_numpy(self.y[idx])


@dataclass
class GeneratorFit:
    """What happened during ``fit``: steps, divergence, the loss log and one entry per checkpoint."""

    steps: int
    steps_done: int = 0
    diverged: bool = False
    divergence: str | None = None
    history: list[dict] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)
    generator_seconds: float = 0.0
    wall_clock_s: float = 0.0


class BeatGenerator(ABC):
    """Interface of every generator: ``fit`` on training beats, then ``sample(cls, n, seed)``.

    Subclasses build their networks in ``__init__`` and implement ``modules``, ``setup_optimizers``,
    ``train_step`` (one optimisation step, returning its losses) and ``generate``.
    """

    family = ""

    def __init__(self, settings: dict, n_samples: int) -> None:
        self.settings = dict(settings)
        self.n_samples = int(n_samples)
        self.device = torch.device("cpu")
        names = settings.get("classes", list(CLASSES))
        self.class_ids: tuple[int, ...] = tuple(CLASS_INDEX[c] for c in names)

    @abstractmethod
    def modules(self) -> dict[str, nn.Module]:
        """The networks, by name; what a checkpoint holds."""

    @abstractmethod
    def setup_optimizers(self) -> None:
        """Create the optimisers (called by ``fit`` after the networks are on the device)."""

    @abstractmethod
    def train_step(self, draw: Callable[[], tuple[torch.Tensor, torch.Tensor]], step: int, total: int) -> dict[str, float]:
        """One optimisation step. ``draw()`` returns a fresh batch ``(x, labels)`` on the device."""

    @abstractmethod
    def generate(self, labels: torch.Tensor, noise: torch.Generator) -> torch.Tensor:
        """Beats ``[n, L]`` for class ``labels``; random numbers come from ``noise`` (a CPU generator)."""

    def batch_classes(self, step: int) -> tuple[int, ...]:
        """The classes of the training batch at ``step`` (all learned classes, balanced, by default)."""
        return self.class_ids

    def sample_classes(self) -> tuple[str, ...]:
        """The classes this generator can sample."""
        return tuple(CLASSES[c] for c in self.class_ids)

    def n_parameters(self) -> int:
        return sum(p.numel() for m in self.modules().values() for p in m.parameters() if p.requires_grad)

    def state_dict(self) -> dict:
        return {name: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for name, m in self.modules().items()}

    def load_state_dict(self, state: dict) -> None:
        for name, module in self.modules().items():
            module.load_state_dict(state[name])

    def to(self, device: torch.device) -> BeatGenerator:
        for module in self.modules().values():
            module.to(device)
        self.device = torch.device(device)
        return self

    def fit(
        self,
        part: dict[str, np.ndarray],
        steps: int,
        seed: int,
        device: torch.device | str = "cpu",
        val_records: tuple[int, ...] | list[int] = (),
        on_checkpoint: CheckpointFn | None = None,
        fractions: tuple[float, ...] | list[float] = CHECKPOINT_FRACTIONS,
        log_every: int = 50,
    ) -> GeneratorFit:
        """Train for ``steps`` steps on training beats only; call ``on_checkpoint`` at each fraction.

        Training stops at the first non-finite or exploding loss, or if a checkpoint produces
        non-finite beats; the fit is then marked diverged and later checkpoints are not reached.
        """
        assert_train_only(part, val_records)
        set_seed(seed)
        self.to(torch.device(device))
        self.setup_optimizers()
        batches = ClassBatches(part["X"], part["y"], self.settings["batch_size"], seed)
        marks = sorted({max(1, round(f * steps)) for f in fractions})
        fit = GeneratorFit(steps=steps)
        pending: dict[str, list[float]] = {}
        started = time.perf_counter()
        callback_seconds = 0.0
        for step in range(1, steps + 1):
            for module in self.modules().values():
                module.train()

            def draw(step: int = step) -> tuple[torch.Tensor, torch.Tensor]:
                x, y = batches.draw(self.batch_classes(step))
                return x.to(self.device), y.to(self.device)

            losses = self.train_step(draw, step, steps)
            bad = {k: v for k, v in losses.items() if not np.isfinite(v) or abs(v) > MAX_ABS_LOSS}
            if bad:
                fit.diverged, fit.divergence = True, f"step {step}: loss out of range {bad}"
                break
            fit.steps_done = step
            for key, value in losses.items():
                pending.setdefault(key, []).append(float(value))
            if step % log_every == 0 or step in marks:
                row = {"step": step, **{k: float(np.mean(v)) for k, v in pending.items()}}
                fit.history.append({**row, "elapsed_s": time.perf_counter() - started - callback_seconds})
                pending = {}
            if step in marks:
                probe = {c: self.sample(c, 8, seed=step) for c in self.sample_classes()}
                if not all(np.isfinite(x).all() for x in probe.values()):
                    fit.diverged, fit.divergence = True, f"step {step}: the generator produced non-finite beats"
                    break
                entry = {"checkpoint": marks.index(step) + 1, "step": step, "fraction": step / steps}
                if on_checkpoint is not None:
                    before = time.perf_counter()
                    entry.update(on_checkpoint(entry["checkpoint"], step, self) or {})
                    callback_seconds += time.perf_counter() - before
                fit.checkpoints.append(entry)
        fit.wall_clock_s = time.perf_counter() - started
        fit.generator_seconds = fit.wall_clock_s - callback_seconds
        return fit

    @torch.no_grad()
    def sample(self, cls: str, n: int, seed: int, batch_size: int = 1024) -> np.ndarray:
        """``n`` beats ``[n, L]`` of class ``cls`` (float32); the same seed gives the same beats."""
        if cls not in self.sample_classes():
            raise ValueError(f"This {self.family} generator cannot sample class {cls!r} (it samples {self.sample_classes()})")
        for module in self.modules().values():
            module.eval()
        noise = torch.Generator().manual_seed(int(seed))
        chunks = []
        for start in range(0, int(n), batch_size):
            labels = torch.full((min(batch_size, int(n) - start),), CLASS_INDEX[cls], dtype=torch.long)
            chunks.append(self.generate(labels, noise).float().cpu())
        if not chunks:
            return np.zeros((0, self.n_samples), dtype=np.float32)
        return torch.cat(chunks).numpy().astype(np.float32)
