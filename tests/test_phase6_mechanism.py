"""P8 attribution helpers and the P7 config check (hand-made cases)."""

import numpy as np
import pytest

from ecgaug.experiments import phase6_augmentation_mechanism as mech
from ecgaug.experiments import phase6_relative_cnn as p7


def test_nearest_finds_the_closest_row_and_can_exclude_itself():
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    assert mech.nearest(ref, np.array([[0.9, 0.1], [4.0, 4.0]])).tolist() == [1, 2]
    assert mech.nearest(ref, ref, exclude_self=True, batch=2).tolist() == [1, 0, 1]


def test_shares_and_effective_number_of_records():
    shares = mech.record_shares(np.array([209, 209, 209, 207]))
    assert list(shares) == ["209", "207"] and shares["209"] == pytest.approx(0.75)
    assert mech.effective_records({"1": 1.0}) == pytest.approx(1.0)
    assert mech.effective_records({"1": 0.5, "2": 0.5}) == pytest.approx(2.0)
    entry = mech._entry(np.array([1, 1, 2]), own=np.array([1, 2, 2]))
    assert entry["top_record"] == "1" and entry["same_record"] == pytest.approx(2 / 3)


def test_configs_refuse_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment: phase6_relative_cnn\nextra: 1\n")
    with pytest.raises(ValueError):
        p7.load_config(bad)
    bad.write_text("experiment: phase6_augmentation_mechanism\nextra: 1\n")
    with pytest.raises(ValueError):
        mech.load_config(bad)


def test_external_beats_are_clustered_by_patient_when_the_header_names_one(tmp_root):
    from ecgaug.data import raw
    from ecgaug.experiments import phase6_external_validation as ext

    incart = raw.external_dir("incartdb")
    incart.mkdir(parents=True)
    for name, patient in (("I01", 1), ("I02", 1), ("I03", 2)):
        (incart / f"{name}.hea").write_text(f"{name} 12 257 462600\n#<age>: 65 <sex>: F\n# patient {patient}\n")
    svdb = raw.external_dir("svdb")
    svdb.mkdir(parents=True)
    (svdb / "800.hea").write_text("800 2 128 230400\n")
    assert ext.patient_of_record("incartdb") == {1: 1, 2: 1, 3: 2}
    assert ext.patient_of_record("svdb") == {800: 800}  # no patient named: the record is its own patient
    assert ext.clusters("incartdb", {"record": np.array([1, 2, 3, 3])}).tolist() == [1, 1, 2, 2]
