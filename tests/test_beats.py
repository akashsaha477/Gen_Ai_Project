"""Beat extraction: RR intervals, symbol handling, AAMI labels, lead selection by name."""

import numpy as np
import pytest

from ecgaug.data import raw
from ecgaug.data.beats import extract_beats, rr_intervals
from ecgaug.data.records import SYMBOL_TO_AAMI


def test_rr_intervals_hand_computed():
    # beats at 0, 1.0, 1.8 and 2.8 s
    rr_prev, rr_next, ratio = rr_intervals(np.array([0, 360, 648, 1008]), fs=360)
    np.testing.assert_allclose(rr_prev, [np.nan, 1.0, 0.8, 1.0])
    np.testing.assert_allclose(rr_next, [1.0, 0.8, 1.0, np.nan])
    # beat 1 has no earlier interval -> 1; beat 2: 0.8 / 1.0; beat 3: 1.0 / mean(1.0, 0.8)
    np.testing.assert_allclose(ratio, [np.nan, 1.0, 0.8, 1.0 / 0.9])


def test_rr_ratio_uses_only_the_ten_preceding_intervals():
    gaps = [2.0, 2.0] + [1.0] * 10 + [0.5]
    samples = np.concatenate([[0], np.cumsum(gaps)]) * 360
    rr_prev, _, ratio = rr_intervals(samples.astype(int), fs=360)
    assert rr_prev[13] == pytest.approx(0.5)
    assert ratio[13] == pytest.approx(0.5)  # the ten preceding intervals are all 1.0
    assert ratio[12] == pytest.approx(1.0 / 1.1)  # preceding ten: one of 2.0 and nine of 1.0


def test_single_beat_record_has_no_rr():
    rr_prev, rr_next, ratio = rr_intervals(np.array([500]))
    assert np.isnan(rr_prev).all() and np.isnan(rr_next).all() and np.isnan(ratio).all()


def test_extract_keeps_qrs_annotations_and_drops_non_beats(fake_root):
    _, written = fake_root
    beats = extract_beats(101)
    np.testing.assert_array_equal(beats["r_idx"], written[101]["qrs_samples"])
    assert beats["symbol"].tolist() == written[101]["qrs_symbols"]
    assert not set(beats["symbol"]) & {"+", "~", "|"}


def test_extract_maps_symbols_to_aami_classes(fake_root):
    beats = extract_beats(101)
    for symbol, aami, y in zip(beats["symbol"], beats["aami"], beats["y"]):
        assert aami == SYMBOL_TO_AAMI.get(symbol, "-")
        assert y == {"N": 0, "S": 1, "V": 2, "F": 3}.get(aami, -1)
    assert {"L", "A", "V", "F", "Q", "!"} <= set(beats["symbol"])  # the fake record exercises every case
    assert (beats.loc[beats["symbol"] == "L", "aami"] == "N").all()
    assert (beats.loc[beats["symbol"] == "!", "y"] == -1).all()
    assert (beats.loc[beats["symbol"] == "Q", "y"] == -1).all()


def test_rr_is_measured_across_q_and_unmapped_beats(fake_root):
    beats = extract_beats(101)
    r, symbols = beats["r_idx"].to_numpy(), beats["symbol"].to_numpy()
    after_dropped = [i for i in range(1, len(beats)) if symbols[i - 1] in ("Q", "!") and symbols[i] in ("N", "L")]
    assert after_dropped, "fake record should contain a classified beat right after a Q or flutter annotation"
    for i in after_dropped:
        assert beats["rr_prev"].iloc[i] == pytest.approx((r[i] - r[i - 1]) / 360)


def test_unknown_symbol_raises(monkeypatch):
    monkeypatch.setattr(raw, "read_annotations", lambda record, access=None: (np.array([10, 400]), np.array(["N", "$"])))
    with pytest.raises(ValueError, match="unknown annotation symbols"):
        extract_beats(101)


def test_unordered_qrs_annotations_raise(monkeypatch):
    monkeypatch.setattr(raw, "read_annotations", lambda record, access=None: (np.array([400, 400]), np.array(["N", "V"])))
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_beats(101)


def test_lead_is_selected_by_name_not_position(fake_root):
    _, written = fake_root
    assert raw.read_header(114)["sig_name"] == ["V5", "MLII"]  # swapped, as in the real record 114
    for record in (101, 114):
        signal = raw.read_signal(record)
        np.testing.assert_allclose(signal, written[record]["mlii"], atol=5e-3)
