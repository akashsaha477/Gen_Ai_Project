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


def test_literature_values_and_the_metrics_section(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    from ecgaug import literature, paper_text

    rows = {r["key"]: r for r in literature.rows()}
    assert abs(rows["dechazal2004"]["f1"]["S"] - 2 * 0.759 * 0.385 / (0.759 + 0.385)) < 1e-12 and rows["dechazal2004"]["f1_computed"] == ["S", "V"]
    assert rows["zhu2022"]["f1"]["F"] == 0.4775 and abs(rows["zhu2022"]["rare_macro_f1"] - (0.6861 + 0.4775) / 2) < 1e-12 and rows["zhu2022"]["f1_computed"] == []
    assert rows["kwak2024diffusion"]["protocol"] == "intra-patient" and rows["mousavi2019"]["note"]
    text = paper_text.metrics_section()
    assert text.count("\\begin{table}") == 3 and text.count("{") == text.count("}") and "tab:literature" in text
    assert "\\textbf{cnn, random oversampling (0.1)}" in text and "\\textbf{P18: cVAE, stage and veto (0.8, 0.95)}" in text
    assert "_" not in text.replace("\\_", "").replace("tab:all-final", "").replace("tab:all-posthoc", "") and "%" not in text.replace("\\%", "")
    table = paper_text.posthoc_table()
    assert table.count("\\begin{table}") == 2 and "P22 &" in table and "tab:posthoc" in table
