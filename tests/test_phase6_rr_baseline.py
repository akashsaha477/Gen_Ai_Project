"""PREREGISTRATION v5: the patient-baseline RR ratio and the bank remapping rule."""

import numpy as np
import pytest

from ecgaug.experiments.common import PreconditionError
from ecgaug.experiments.phase6_rr_baseline import baseline_ratio, remap_bank, substitute


def test_baseline_ratio_divides_by_the_median_interval_in_the_window():
    fs = 100
    t = np.array([0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.5])  # intervals 1, 1, 1, 0.5, 0.5, 0.5, 1
    ratio = baseline_ratio((t * fs).astype(int), fs, half_window_s=100.0)
    assert np.isnan(ratio[0])
    assert ratio[1] == pytest.approx(1.0)  # median of all seven intervals is 1
    assert ratio[4] == pytest.approx(0.5)  # an early beat in a run stays early against the baseline
    assert ratio[5] == pytest.approx(0.5)


def test_baseline_ratio_window_is_local():
    fs = 10
    t = np.concatenate([np.arange(0, 1000, 1.0), 1000 + np.arange(1, 1001) * 0.5])  # 1 s rhythm, then 0.5 s
    ratio = baseline_ratio((t * fs).astype(int), fs, half_window_s=150.0)
    assert ratio[10] == pytest.approx(1.0)
    assert ratio[-10] == pytest.approx(1.0)  # far into the faster rhythm its own median is the baseline


def test_remap_takes_the_ratio_of_the_matching_real_beat_and_refuses_unknown_values():
    v1 = {"y": np.array([1, 1, 2], dtype=np.int8), "rr_prev": np.array([0.5, 0.6, 0.5], dtype=np.float32),
          "rr_next": np.array([0.9, 0.9, 0.9], dtype=np.float32), "rr_ratio": np.array([0.8, 0.9, 0.8], dtype=np.float32)}
    v2 = {**v1, "rr_ratio": np.array([0.61, 0.72, 0.55], dtype=np.float32)}
    bank = {"y": np.array([1, 2, 1], dtype=np.int8), "rr_prev": np.array([0.6, 0.5, 0.5], dtype=np.float32),
            "rr_next": np.array([0.9, 0.9, 0.9], dtype=np.float32), "rr_ratio": np.array([0.9, 0.8, 0.8], dtype=np.float32)}
    out = remap_bank(bank, v1, v2)
    assert out["rr_ratio"].tolist() == pytest.approx([0.72, 0.55, 0.61])
    bad = {**bank, "y": np.array([2, 2, 2], dtype=np.int8)}  # class 2 never carried (0.6, 0.9, 0.9)
    with pytest.raises(PreconditionError):
        remap_bank(bad, v1, v2)


def test_substitute_refuses_a_beat_without_a_ratio():
    arrays = {"record": np.array([1, 1]), "r_idx": np.array([10, 20]), "rr_ratio": np.zeros(2, dtype=np.float32)}
    assert substitute(arrays, {(1, 10): 0.5, (1, 20): 1.5})["rr_ratio"].tolist() == [0.5, 1.5]
    with pytest.raises(PreconditionError):
        substitute(arrays, {(1, 10): 0.5})
