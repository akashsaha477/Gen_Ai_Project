"""The architecture inventory: parameter counts come from the configs and match what phase 2 recorded; every classifier
architecture and generator family appears; choices follow the gate records; the flow figure and the report section build."""

import json

from ecgaug import architecture as arch
from ecgaug import paths
from ecgaug.generators import FAMILIES
from ecgaug.models import ARCHITECTURES


def test_parameter_counts_match_phase_2s_records_and_every_model_appears(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    rows = {r["name"]: r for r in arch.inventory()}
    for variant, name in (("A", "Shallow CNN (the cnn)"), ("B", "ResNet1D"), ("C", "BiLSTM (the rnn)"), ("D", "CNN-BiLSTM")):
        recorded = json.loads((paths.results_for("phase2_baseline_classifiers") / f"{variant}.json").read_text())["n_parameters"]
        assert rows[name]["parameters"] == recorded
    structures = " ".join(r["structure"] + r["name"] for r in rows.values()).lower()
    for model in ARCHITECTURES:  # probe_cnn, shallow_cnn, resnet1d, bilstm, cnn_bilstm
        assert model.replace("_", " ").replace("1d", "").split()[0] in structures
    assert {"cvae", "wgan", "diffusion"} <= {w for w in ("cvae", "wgan", "diffusion") if w in structures} and set(FAMILIES) == {"cvae", "wgan_gp", "diffusion"}
    assert {f"cVAE {v}" for v in "ABCD"} | {f"WGAN-GP {v}" for v in "ABCD"} <= set(rows)
    assert rows["cVAE D"]["chosen"] and rows["WGAN-GP A"]["chosen"] and not rows["cVAE A"]["chosen"]
    assert rows["Shallow CNN (the cnn)"]["chosen"] and rows["BiLSTM (the rnn)"]["chosen"] and not rows["ResNet1D"]["chosen"]
    assert sum(r["chosen"] for r in rows.values() if r["group"] == "note") == 1
    assert all(r["group"] in arch.GROUPS for r in rows.values())


def test_the_flow_figure_and_the_report_section_build(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    from ecgaug import figures, paper_text

    monkeypatch.setattr(figures.paths, "figures_for", lambda name, _tmp=paths.repo_root(): _tmp / "results" / ".test_figures" / name)
    out = figures.architecture_flow_figure()
    try:
        assert len(out) == 1 and out[0].is_file()
    finally:
        out[0].unlink()
        out[0].parent.rmdir()
        out[0].parent.parent.rmdir()
    text = paper_text.architecture_section()
    assert text.count("\\begin{table}") == 3 and text.count("{") == text.count("}") and "fig:architecture" in text
    assert "_" not in text.replace("\\_", "") and "%" not in text.replace("\\%", "")
    assert "73,226" in text and "488,362" in text and "\\textbf{cVAE D}" in text
