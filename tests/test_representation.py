"""Input representations: window sizes, normalisation, filtering, resampling, identical beat sets."""

import numpy as np
import pytest

from conftest import FAKE_DS1, REPO, variant_config
from ecgaug.data.contract import build_arrays
from ecgaug.data.representation import (
    RepresentationSpec,
    bandpass,
    transform_record,
    usable_mask,
    zscore_per_beat,
)
from ecgaug.experiments.phase1_input_representation import load_config

SPECS = {
    letter: RepresentationSpec(**load_config(variant_config(REPO / "configs" / "phase1_input_representation", letter))["representation"])
    for letter in "ABCD"
}


def test_window_sizes_match_the_variant_definitions():
    assert SPECS["A"].window_samples == (90, 90) and SPECS["A"].n_samples == 180
    assert SPECS["B"].window_samples == (144, 216) and SPECS["B"].n_samples == 360
    assert SPECS["C"].window_samples == (90, 90) and SPECS["C"].n_samples == 180
    assert SPECS["D"].window_samples == (31, 32) and SPECS["D"].n_samples == 63
    assert [SPECS[k].n_inputs for k in "ABCD"] == [180, 360, 183, 63]
    assert SPECS["B"].bandpass_hz == (0.5, 40.0) and SPECS["D"].target_fs == 125
    assert SPECS["C"].rr_features and not any(SPECS[k].rr_features for k in "ABD")


def test_zscore_per_beat():
    rng = np.random.default_rng(0)
    z = zscore_per_beat(rng.normal(3.0, 5.0, size=(20, 180)))
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-12)
    flat = zscore_per_beat(np.full((2, 180), 7.0))
    assert np.isfinite(flat).all() and (flat == 0).all()


def _spikes(n=20_000, positions=(3000, 9000, 15000)):
    signal = np.zeros(n)
    for p in positions:
        signal[p - 2 : p + 3] = [0.2, 0.7, 1.0, 0.7, 0.2]
    return signal, np.array(positions)


@pytest.mark.parametrize("letter", "ABCD")
def test_windows_are_centred_on_the_r_peak(letter):
    spec = SPECS[letter]
    signal, r = _spikes()
    x = transform_record(signal, r, spec)
    assert x.shape == (3, spec.n_samples) and x.dtype == np.float32
    pre = spec.window_samples[0]
    assert np.all(np.abs(x.argmax(axis=1) - pre) <= 1)


def test_bandpass_removes_offset_and_drift_without_moving_the_peak():
    time = np.arange(20_000) / 360
    drift = 5.0 + 2.0 * np.sin(2 * np.pi * 0.05 * time)
    filtered_drift = bandpass(drift, (0.5, 40.0), order=4)
    assert np.abs(filtered_drift[3600:-3600]).max() < 0.05
    signal, r = _spikes()
    x = transform_record(signal + drift, r, SPECS["B"])
    clean = transform_record(signal, r, SPECS["B"])
    assert np.corrcoef(x.ravel(), clean.ravel())[0, 1] > 0.999


def test_usable_mask_boundaries():
    n = 10_000
    r = np.array([143, 144, n - 216, n - 215, 5000, 5001])
    ok = np.ones(6)
    rr_prev = ok.copy()
    rr_next = ok.copy()
    rr_prev[4] = np.nan
    np.testing.assert_array_equal(usable_mask(r, n, rr_prev, rr_next), [False, True, True, False, False, True])


def test_transform_refuses_a_window_past_the_record_edge():
    with pytest.raises(ValueError, match="record edge"):
        transform_record(np.zeros(1000), np.array([10]), SPECS["A"])


def test_spec_rejects_windows_wider_than_the_usable_margin():
    with pytest.raises(ValueError, match="usable-beat margins"):
        RepresentationSpec(name="too-wide", window_pre_s=0.5, window_post_s=0.5)
    with pytest.raises(ValueError, match="normalization"):
        RepresentationSpec(name="x", window_pre_s=0.25, window_post_s=0.25, normalization="global")


def test_beat_set_is_identical_across_variants(fake_root):
    split_of = {r: "train" for r in FAKE_DS1}
    built = {k: build_arrays(list(FAKE_DS1), spec, split_of)[0] for k, spec in SPECS.items()}
    reference = built["A"]
    assert len(reference["y"]) > 300
    for letter, arrays in built.items():
        assert arrays["X"].shape == (len(reference["y"]), SPECS[letter].n_samples)
        for name in ("record", "r_idx", "y", "rr_prev", "rr_next", "rr_ratio"):
            np.testing.assert_array_equal(arrays[name], reference[name], err_msg=f"{letter}:{name}")
    np.testing.assert_array_equal(built["A"]["X"], built["C"]["X"])  # C is A plus RR side features
