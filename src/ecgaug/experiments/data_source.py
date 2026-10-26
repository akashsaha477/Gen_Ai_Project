"""Where a run's beats come from: the dataset frozen at the phase 1 gate, or a preview candidate.

Shared by every runner after phase 1 (baseline classifiers, quality metrics, generators, the grid).
After the phase 1 gate a run uses the frozen dataset: ``data/processed/v1`` if it is on disk, or the
same arrays rebuilt from the raw records and checked against the frozen hash (on Kaggle). Before the
gate, or for another representation, only a preview may run, on the candidate dataset of the
representation named by ``--representation`` (default: the winner in the phase 1 gate report).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecgaug import paths, workflow
from ecgaug.data.contract import TRAINVAL_FILE, build_candidate, content_sha256
from ecgaug.data.counts import count_record
from ecgaug.data.fake import write_fake_mitdb
from ecgaug.data.loaders import load_candidate, load_trainval
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split, check_split, selection_metric_for
from ecgaug.experiments import phase1_input_representation as phase1
from ecgaug.experiments.common import PreconditionError, temporary_root
from ecgaug.names import PHASE1_GATE


@dataclass
class DataSource:
    arrays: dict[str, np.ndarray]
    spec: RepresentationSpec
    info: dict
    provisional: str | None = None  # why this is not the frozen dataset


def reported_phase1_winner() -> str | None:
    """The winner in the written (not necessarily approved) phase 1 gate report."""
    report = paths.results_for(PHASE1_GATE) / "report.json"
    return json.loads(report.read_text())["winner"] if report.is_file() else None


def load_data(split: Split, preview: bool, representation: str | None = None) -> DataSource:
    """The training and validation beats a run uses.

    After the phase 1 gate: the frozen dataset, from ``data/processed/v1`` or, where that is not on
    disk (Kaggle), rebuilt from the raw records and checked against the frozen hash. In a preview:
    the candidate dataset of ``representation`` or of the phase 1 gate report's winner.
    """
    winner = workflow.phase1_winner()
    letter = representation or winner or reported_phase1_winner()
    if letter is None:
        raise PreconditionError(
            "No input representation to use: the phase 1 gate is not approved, its report is not written, "
            "and --representation was not given."
        )
    configs = workflow.variant_configs()
    if letter not in configs:
        raise PreconditionError(f"No phase1_input_representation variant {letter!r}")
    spec = RepresentationSpec(**phase1.load_config(configs[letter])["representation"])
    if winner is not None and letter == winner:
        expected = workflow.manifest()[PHASE1_GATE]["files"][TRAINVAL_FILE]["content_sha256"]
        if (paths.processed_dir() / "v1" / TRAINVAL_FILE).is_file():
            arrays, source = load_trainval("v1"), "data/processed/v1 (frozen at the phase 1 gate)"
        else:
            build_candidate(spec, split)
            arrays, source = load_candidate(spec.name), f"rebuilt from the raw records ({spec.name}) and checked against the frozen hash"
        actual = content_sha256(arrays)
        if actual != expected:
            raise PreconditionError(
                f"The training data ({actual[:12]}...) does not match the frozen dataset ({expected[:12]}..., "
                "results/manifest.json). Use the frozen files (data/processed/v1) instead of rebuilding."
            )
        return DataSource(arrays, spec, {"source": source, "representation": spec.name, "content_sha256": actual, "frozen": True})
    if not preview:
        raise PreconditionError(
            "Phase 1 is not finished: the phase 1 gate has not frozen the dataset. Approve the phase 1 gate first "
            "(python -m ecgaug.workflow status), or add --preview."
        )
    build_candidate(spec, split)
    arrays = load_candidate(spec.name)
    reason = "the phase 1 gate has not been approved" if winner is None else f"the phase 1 gate chose {winner}, not {letter}"
    info = {"source": f"candidate dataset {spec.name}", "representation": spec.name, "content_sha256": content_sha256(arrays), "frozen": False}
    return DataSource(arrays, spec, info, f"{reason}: the beats come from the candidate dataset {spec.name}")


SMOKE_TRAIN, SMOKE_VAL = (101, 106, 108, 114), (118, 201)


@contextmanager
def smoke_data(spec: RepresentationSpec | None = None) -> Iterator[tuple[Split, DataSource]]:
    """Fake records in a temporary root, a split of them and their candidate dataset, for smoke runs.

    The default representation is that of phase1_input_representation variant C (0.5 s window plus
    RR features). Nothing is written to the repository.
    """
    spec = spec or RepresentationSpec(name="smoke_short_window_rr", window_pre_s=0.25, window_post_s=0.25, rr_features=True)
    with temporary_root():
        write_fake_mitdb(paths.raw_dir(), SMOKE_TRAIN + SMOKE_VAL, seconds=120.0)
        counts = pd.DataFrame([count_record(r) for r in SMOKE_TRAIN + SMOKE_VAL])
        split = Split(name="smoke", train=SMOKE_TRAIN, val=SMOKE_VAL)
        check_split(split, require_all_ds1=False)
        split = Split(name="smoke", train=SMOKE_TRAIN, val=SMOKE_VAL, selection_metric=selection_metric_for(counts, split))
        build_candidate(spec, split, require_all_ds1=False)
        arrays = load_candidate(spec.name)
        info = {"source": "fake records (smoke run)", "representation": spec.name, "content_sha256": content_sha256(arrays), "frozen": False}
        yield split, DataSource(arrays, spec, info, "smoke run on fake data")
