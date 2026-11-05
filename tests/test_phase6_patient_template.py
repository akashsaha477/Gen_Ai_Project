"""phase6_patient_template: the local template, the relative beat, the bank remap and the adoption rule (hand-made cases)."""

import numpy as np
import pytest

from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments.common import PreconditionError


def test_local_template_is_the_median_of_the_windows_within_the_half_window():
    windows = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [9.0, 90.0]], dtype=np.float32)
    r_all = np.array([0, 360, 720, 3600])  # seconds 0, 1, 2, 10 at 360 Hz
    out = pt.local_templates(windows, r_all, np.array([360, 3600]), fs=360, half_window_s=1.0)
    assert np.allclose(out[0], [1.0, 10.0])  # beats at 0, 1 and 2 s
    assert np.allclose(out[1], [9.0, 90.0])  # only itself
    with pytest.raises(ValueError, match="own window"):
        pt.local_templates(windows, r_all, np.array([361]), fs=360, half_window_s=1.0)


def test_relative_subtracts_the_template():
    arrays = {"X": np.array([[1.0, 2.0]], dtype=np.float32), "y": np.array([0])}
    out = pt.relative(arrays, np.array([[0.5, 0.5]], dtype=np.float32))
    assert np.allclose(out["X"], [[0.5, 1.5]]) and out["y"] is arrays["y"]


def test_bank_remap_takes_ratio_and_template_of_the_matched_real_beat():
    v1 = {"y": np.array([1, 1]), "rr_prev": np.float32([0.5, 0.6]), "rr_next": np.float32([0.9, 0.9]), "rr_ratio": np.float32([0.7, 0.8])}
    v3 = {**v1, "rr_ratio": np.float32([0.4, 0.5])}
    templates = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    bank = {"X": np.array([[3.0, 3.0]], dtype=np.float32), "y": np.array([1]), "rr_prev": np.float32([0.6]), "rr_next": np.float32([0.9]), "rr_ratio": np.float32([0.8])}
    out = pt.remap_bank(bank, v1, v3, templates)
    assert np.allclose(out["X"], [[1.0, 1.0]]) and np.allclose(out["rr_ratio"], [0.5])
    with pytest.raises(PreconditionError):
        pt.remap_bank({**bank, "y": np.array([2])}, v1, v3, templates)


def _rows(values):
    return [{"seed": s, "selection_value": v, "run_id": f"r{s}"} for s, v in enumerate(values)]


def test_adoption_needs_the_mean_gain_to_exceed_the_margin():
    cfg = {"adoption_margin": 0.02}
    old = {"cnn": _rows([0.10, 0.10, 0.10]), "rnn": _rows([0.10, 0.10, 0.10])}
    assert pt.adoption(cfg, {"cnn": _rows([0.15, 0.15, 0.15]), "rnn": _rows([0.12, 0.12, 0.12])}, old)["adopted"]  # mean +0.035
    assert not pt.adoption(cfg, {"cnn": _rows([0.13, 0.13, 0.13]), "rnn": _rows([0.10, 0.10, 0.10])}, old)["adopted"]  # mean +0.015
    noisy = pt.adoption(cfg, {"cnn": _rows([0.05, 0.25, 0.15]), "rnn": _rows([0.05, 0.25, 0.15])}, old)  # +0.05, but seed SD 0.1
    assert not noisy["adopted"] and noisy["margin"] > 0.05
