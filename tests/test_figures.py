"""Before-and-after figures: they are drawn from DS1 and PTB-XL data, never from DS2."""

import json
import re
import shutil

import pandas as pd

from conftest import REPO
from ecgaug import figures, paths
from ecgaug.data import ptbxl
from ecgaug.data.counts import count_record
from ecgaug.data.fake import write_fake_ptbxl
from ecgaug.data.records import DS1


def _png_ok(path):
    return path.is_file() and path.stat().st_size > 10_000 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_phase1_figures_on_fake_records(full_fake_root):
    shutil.copytree(REPO / "configs", paths.configs_dir())
    (paths.results_for("phase1_data")).mkdir(parents=True)
    pd.DataFrame([count_record(r) for r in DS1]).to_csv(paths.results_for("phase1_data") / "record_counts.csv", index=False)
    written = figures.phase1_data_figures()
    assert [p.name for p in written] == [
        "01_raw_vs_filtered_signal.png", "02_same_beats_in_each_variant.png",
        "03_class_counts_train_vs_validation.png", "04_class_average_beats_per_variant.png",
    ]  # fmt: skip
    assert all(_png_ok(p) for p in written)


def test_ptbxl_figures_from_raw_files_and_the_report(fake_root, monkeypatch):
    shutil.copytree(REPO / "configs", paths.configs_dir())
    shutil.copytree(REPO / "configs" / "validation_split", paths.configs_dir() / "validation_split", dirs_exist_ok=True)
    root = paths.repo_root() / "ptbxl_raw"
    write_fake_ptbxl(root, n_extra=12)
    ptbxl.prepare(root, verify=False)
    # the MIT-BIH comparison needs the training records; restrict the approved split to the fake ones
    (paths.configs_dir() / "validation_split" / "approved.yaml").write_text("name: t\nval: [118, 201]\nselection_metric: s_f1\napproved_by: [x]\n")
    monkeypatch.setattr(figures, "_train_records", lambda: (101, 106, 108, 114))
    written = figures.phase2_ptbxl_data_figures(root)
    assert [p.name for p in written] == [
        "01_resampling_and_detected_beats.png", "02_ptbxl_vs_mitbih_beat_shape.png",
        "03_recordings_kept_and_excluded.png", "04_label_share_per_split.png",
    ]  # fmt: skip
    assert all(_png_ok(p) for p in written)


def test_a_plotting_error_never_fails_a_run(tmp_root, capsys):
    bad = tmp_root / "results" / "validation" / "phase2_ptbxl_pretraining" / "Z.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps({"variant": "Z"}))
    assert figures.safe_for_result(bad) == []
    assert "WARNING" in capsys.readouterr().out


def test_figure_code_reads_no_test_data():
    text = (REPO / "src" / "ecgaug" / "figures.py").read_text()
    assert "load_test" not in text and "ALLOW_TEST" not in text and not re.search(r'access="', text)
    assert "DS2" not in text.replace("DS2 never is", "")
