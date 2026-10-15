"""PTB-XL preprocessing and the R-peak detector."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import REPO
from ecgaug import paths
from ecgaug.data import ptbxl
from ecgaug.data.contract import content_sha256
from ecgaug.data.fake import write_fake_ptbxl
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.rpeaks import detect_r_peaks, match_peaks

A = RepresentationSpec(name="m1-A", window_pre_s=0.25, window_post_s=0.25)
D = RepresentationSpec(name="m1-D", window_pre_s=0.25, window_post_s=0.25, target_fs=125)


@pytest.fixture
def fake_ptbxl(tmp_root):
    root = tmp_root / "ptbxl"
    return root, write_fake_ptbxl(root)


def test_detector_finds_clean_beats_at_the_right_sample():
    fs, truth = 360, np.arange(300, 10_500, 290)
    time = np.arange(10_800)
    x = 0.01 * np.random.default_rng(0).standard_normal(time.size) + 0.3 * np.sin(2 * np.pi * time / 3000)
    for t in truth:
        x += np.exp(-0.5 * ((time - t) / 4.0) ** 2)
    x[truth[5] - 3 : truth[5] + 4] *= -1.8  # one inverted, larger beat
    peaks = detect_r_peaks(x, fs)
    tp, fn, fp, offsets = match_peaks(truth, peaks, tolerance=54)
    assert (tp, fn, fp) == (truth.size, 0, 0)
    assert np.abs(offsets).max() <= 2


def test_detector_returns_nothing_for_flat_or_invalid_input():
    assert detect_r_peaks(np.zeros(3600)).size == 0
    assert detect_r_peaks(np.full(3600, np.nan)).size == 0
    assert detect_r_peaks(np.ones(100)).size == 0


def test_match_peaks_hand_computed():
    tp, fn, fp, offsets = match_peaks(np.array([100, 500, 900]), np.array([103, 480, 700, 2000]), tolerance=30)
    assert (tp, fn, fp) == (2, 1, 2) and offsets.tolist() == [3, -20]
    tp, fn, fp, _ = match_peaks(np.array([100]), np.array([95, 105]), tolerance=30)
    assert (tp, fn, fp) == (1, 0, 1)  # a reference beat is matched once


def test_labels_are_aggregated_to_superclasses():
    scp = pd.DataFrame(
        {"diagnostic": [1, 1, 1, np.nan], "diagnostic_class": ["NORM", "MI", "STTC", np.nan]},
        index=["NORM", "IMI", "NDT", "SR"],
    )
    mapping = ptbxl.superclass_map(scp)
    assert mapping == {"NORM": "NORM", "IMI": "MI", "NDT": "STTC"}
    np.testing.assert_array_equal(ptbxl.record_labels("{'IMI': 50.0, 'NDT': 0.0, 'SR': 0.0}", mapping), [0, 1, 1, 0, 0])
    assert not ptbxl.record_labels("{'SR': 0.0}", mapping).any()


def test_signal_checks():
    good = np.sin(np.arange(5000) / 20.0)
    assert ptbxl.signal_problem(good, 500) is None
    assert ptbxl.signal_problem(good[:4000], 500) == "unexpected_length_or_rate"
    assert ptbxl.signal_problem(good, 100) == "unexpected_length_or_rate"
    assert ptbxl.signal_problem(np.where(np.arange(5000) == 7, np.nan, good), 500) == "non_finite"
    assert ptbxl.signal_problem(np.zeros(5000), 500) == "flat_signal"
    assert ptbxl.signal_problem(good + np.where(np.arange(5000) == 9, 50.0, 0.0), 500) == "implausible_amplitude"
    stuck = good.copy()
    stuck[1000:1700] = stuck[1000]
    assert ptbxl.signal_problem(stuck, 500) == "flat_segment"
    assert ptbxl.longest_flat_run(np.array([1, 1, 1, 2, 2, 3])) == 3
    assert ptbxl.resample_to_fs(good, 500).size == 3600


def test_metadata_exclusions_are_counted_by_reason(fake_ptbxl):
    meta, report = ptbxl.load_metadata(fake_ptbxl[0])
    assert report["in_metadata"] == 9 and report["eligible"] == 6
    assert report["excluded_by_metadata"] == {"no_diagnostic_superclass": 1, "pacemaker": 1, "electrodes_problems": 1}
    assert meta["ecg_id"].tolist() == [1, 2, 3, 4, 8, 9]


def test_pipeline_end_to_end_on_fake_recordings(fake_ptbxl):
    root, truth = fake_ptbxl
    arrays, report = ptbxl.build(root)
    assert arrays["ecg_id"].tolist() == [1, 2, 3, 4]
    assert report["excluded_by_signal"] == {"flat_signal": 1, "low_qrs_contrast": 1}
    assert arrays["signal"].shape == (4, 3600) and arrays["signal"].dtype == np.float32
    assert arrays["split"].tolist() == ["train", "train", "val", "unused"]
    np.testing.assert_array_equal(arrays["labels"], [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 0]])

    for index, ecg_id in enumerate(arrays["ecg_id"]):
        r = arrays["r_idx"][arrays["beat_record"] == index]
        expected = truth[int(ecg_id)] * 360 / 500
        assert r.size >= 6
        for peak in r:  # every kept beat sits on a true R peak, including the lead-swapped recording 3
            assert np.abs(expected - peak).min() <= 2
        assert r.min() >= 144 and r.max() <= 3600 - 216
    assert np.isfinite(arrays["rr_prev"]).all() and (arrays["rr_prev"] >= 0.3).all() and (arrays["rr_next"] <= 2.0).all()
    assert report["beats_kept"] == arrays["r_idx"].size
    assert report["per_split"]["train"]["patients"] == 1 and report["per_split"]["train"]["recordings"] == 2


def test_no_patient_is_in_two_splits(fake_ptbxl):
    arrays, _ = ptbxl.build(fake_ptbxl[0])
    ptbxl.check_patient_split(arrays)
    leaky = {**arrays, "split": np.array(["train", "val", "val", "unused"])}  # patient 10 in train and val
    with pytest.raises(ValueError, match="more than one split"):
        ptbxl.check_patient_split(leaky)
    assert [ptbxl.fold_split(f) for f in (1, 8, 9, 10)] == ["train", "train", "val", "unused"]


@pytest.mark.parametrize("spec", [A, D], ids=["A", "D"])
def test_beat_arrays_use_the_mitbih_representation(fake_ptbxl, spec):
    arrays, _ = ptbxl.build(fake_ptbxl[0])
    train = ptbxl.beat_arrays(arrays, spec, "train")
    n = int(np.isin(arrays["beat_record"], [0, 1]).sum())
    assert train["X"].shape == (n, spec.n_samples) and train["X"].dtype == np.float32
    assert train["y"].shape == (n, 5) and train["side"].shape == (n, 3)
    assert set(train["ecg_id"].tolist()) == {1, 2} and set(train["patient"].tolist()) == {10}
    assert np.all(np.abs(train["X"].argmax(axis=1) - spec.window_samples[0]) <= 1)  # R peak centred like MIT-BIH
    np.testing.assert_array_equal(train["y"][train["ecg_id"] == 2][0], [0, 1, 1, 0, 0])
    val = ptbxl.beat_arrays(arrays, spec, "val")
    assert set(val["ecg_id"].tolist()) == {3}


def test_cli_writes_hashed_output_that_reloads(fake_ptbxl):
    assert ptbxl.main(["--root", str(fake_ptbxl[0]), "--skip-verify"]) == 0
    report = json.loads((ptbxl.out_dir() / "report.json").read_text())
    loaded = ptbxl.load()
    assert report["content_sha256"] == content_sha256(loaded)
    assert not report["partial_run"] and (paths.results_for("phase2_ptbxl_data") / "ptbxl_preprocessing.json").is_file()
    first = report["file_sha256"]
    assert ptbxl.main(["--root", str(fake_ptbxl[0]), "--skip-verify"]) == 0
    assert json.loads((ptbxl.out_dir() / "report.json").read_text())["file_sha256"] == first
    (ptbxl.out_dir() / "records.npz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="does not match"):
        ptbxl.load()


def test_ptbxl_code_cannot_reach_mitbih_records():
    text = (REPO / "src" / "ecgaug" / "data" / "ptbxl.py").read_text()
    assert "read_signal" not in text and "read_annotations" not in text and "load_test" not in text


def test_result_does_not_depend_on_the_number_of_workers(tmp_root):
    root = tmp_root / "ptbxl"
    write_fake_ptbxl(root, n_extra=12)
    one, report_one = ptbxl.build(root, workers=1)
    two, report_two = ptbxl.build(root, workers=2)
    assert report_one["content_sha256"] == report_two["content_sha256"]
    assert content_sha256(one) == content_sha256(two) and report_one["recordings_kept"] == 16


def test_root_is_found_below_an_attached_folder(fake_ptbxl, tmp_root):
    assert ptbxl.find_root(fake_ptbxl[0]) == fake_ptbxl[0]
    assert ptbxl.find_root(tmp_root) == fake_ptbxl[0]
    with pytest.raises(FileNotFoundError, match="no ptbxl_database.csv"):
        ptbxl.find_root(tmp_root / "records500")


def test_report_carries_quality_measures(fake_ptbxl):
    _, report = ptbxl.build(fake_ptbxl[0])
    quality = report["quality_of_kept_recordings"]
    assert 40 < quality["heart_rate_bpm"]["p50"] < 120  # the fake beats are 0.77-0.83 s apart
    assert quality["qrs_contrast"]["p1"] >= ptbxl.MIN_QRS_CONTRAST
    assert quality["powerline_50hz_ratio"]["share_above_10"] == 0.0
    assert report["settings"]["edge_pad_mode"] == "reflect" and "checksums" not in report


def test_powerline_ratio_detects_mains_interference():
    time = np.arange(5000) / 500
    clean = np.random.default_rng(0).standard_normal(5000)
    assert ptbxl.powerline_ratio(clean, 500) < 10
    assert ptbxl.powerline_ratio(clean + 0.5 * np.sin(2 * np.pi * 50 * time), 500) > 50
    assert ptbxl.powerline_ratio(clean + 0.5 * np.sin(2 * np.pi * 60 * time), 500, f0=60.0) > 50


def test_checksum_comparison_reports_matches_and_mismatches(fake_ptbxl):
    root = fake_ptbxl[0]
    arrays, report = ptbxl.build(root, verify=False)
    hea, dat = "records500/00000/00001_hr.hea", "records500/00000/00001_hr.dat"
    from ecgaug.data import raw

    hashes = {hea: raw.file_sha256(root / hea), dat: raw.file_sha256(root / dat)}
    reference = {hea: hashes[hea], dat: "0" * 64, "ptbxl_database.csv": raw.file_sha256(root / "ptbxl_database.csv")}
    out = ptbxl.verify_checksums(root, hashes, reference=reference)
    assert out["files_checked"] == 4 and out["matched"] == 2 and out["mismatched"] == 1 and out["not_in_reference"] == 1
    assert out["examples_mismatched"] == [dat]
    assert out["metadata_matches_reference"] == {"ptbxl_database.csv": True, "scp_statements.csv": False}


def test_verified_build_hashes_the_files_it_read(fake_ptbxl, monkeypatch):
    from ecgaug.data import raw

    monkeypatch.setattr(raw, "fetch_reference_checksums", lambda *a, **k: None)  # offline
    _, report = ptbxl.build(fake_ptbxl[0], verify=True)
    assert report["checksums"]["reference_available"] is False
    assert report["checksums"]["files_checked"] == 2 * report["processed_from_metadata"] + 2


def test_download_extracts_only_what_the_pipeline_needs(tmp_root, fake_ptbxl):
    import shutil

    from ecgaug.data import raw

    staging = tmp_root / "zip_src" / "ptb-xl-1.0.3"
    shutil.copytree(fake_ptbxl[0], staging)
    (staging / "records100" / "00000").mkdir(parents=True)
    (staging / "records100" / "00000" / "00001_lr.dat").write_bytes(b"low rate copy")
    archive = shutil.make_archive(str(tmp_root / "ptbxl_fake"), "zip", tmp_root / "zip_src")
    root = raw.download_ptbxl(tmp_root / "downloaded", url=f"file://{archive}")
    assert root == tmp_root / "downloaded" / "ptb-xl-1.0.3" and (root / "scp_statements.csv").is_file()
    assert (root / "records500" / "00000" / "00001_hr.dat").is_file()
    assert not (root / "records100").exists() and not (tmp_root / "downloaded" / "ptbxl.zip").exists()
    assert ptbxl.build(ptbxl.find_root(tmp_root / "downloaded"))[1]["recordings_kept"] == 4


def test_mirror_padding_reduces_the_edge_error_of_a_band_pass():
    fs = 360
    time = np.arange(30 * fs) / fs
    beats = np.rint(np.arange(0.4, 29.6, 0.8) * fs).astype(int)
    x = 2.0 + 0.5 * np.sin(2 * np.pi * 0.3 * time)  # offset and slow drift
    for b in beats:
        x += np.exp(-0.5 * ((time - b / fs) / 0.012) ** 2)
    segment = x[3600:7200]
    r = beats[(beats >= 3600 + 144) & (beats <= 7200 - 216)] - 3600
    band = RepresentationSpec(name="B", window_pre_s=0.4, window_post_s=0.6, bandpass_hz=(0.5, 40.0))
    from ecgaug.data.representation import transform_record

    inside_long_record = transform_record(x, r + 3600, band)
    padded = np.abs(ptbxl.windows_for_record(segment, r, band) - inside_long_record).max(axis=1)
    unpadded = np.abs(transform_record(segment, r, band) - inside_long_record).max(axis=1)
    # the worse of the two beats nearest the ends improves; on real data the gain is measured by
    # `python -m ecgaug.data.checks` (results/phase2_ptbxl_data/method_checks_ds1.json)
    assert max(padded[0], padded[-1]) < 0.5 * max(unpadded[0], unpadded[-1])
    assert padded[len(r) // 2] < 0.05
    # without a filter, padding changes nothing
    np.testing.assert_array_equal(ptbxl.windows_for_record(segment, r, A), transform_record(segment, r, A))


def test_alignment_report_hand_case():
    centred = np.zeros((4, 180), dtype=np.float32)
    centred[:, 90] = 1.0
    shifted = np.zeros((4, 180), dtype=np.float32)
    shifted[:2, 90] = 1.0
    shifted[2:, 120] = 1.0
    out = ptbxl.alignment_report(shifted, centred, A)
    assert out["ptbxl_peak_at_centre"] == 0.5 and out["mitbih_peak_at_centre"] == 1.0
    assert out["tolerance_samples"] == 2 and out["n_ptbxl_beats"] == 4
    assert ptbxl.alignment_report(centred, centred, A)["mean_beat_correlation"] == pytest.approx(1.0)
