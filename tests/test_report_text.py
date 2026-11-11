"""The course report's own texts and figures: every number comes from a result file, the example cases follow their
selection rule, the architecture tables can flow across pages, and the medical figure reads training records only."""

import json
import shutil

from conftest import REPO

from ecgaug import paths


def test_report_texts_come_from_the_result_files(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    from ecgaug import paper_text as pt
    from ecgaug import report_text as rt

    texts = rt.build()
    assert set(texts) == {"report_summary", "report_example", "report_architecture", "report_posthoc", "report_discussion", "report_conclusion"}
    for name, text in texts.items():
        assert "\\missing" not in text and text.count("{") == text.count("}"), name
        assert "_" not in text.replace("\\_", ""), name
    final = json.loads((paths.results_for("final_eval") / "summary.json").read_text())
    best = json.loads((paths.results_for("phase6_seed_ensemble") / "summary.json").read_text())["ensemble"]["with_af_veto"]["cnn/cvae"]
    for key in ("report_summary", "report_conclusion"):
        assert pt._n(final["conditions"]["cnn/none"]["rare_macro_f1"]["point"]) in texts[key]
        assert pt._n(best["rare_macro_f1"]["point"]) in texts[key]
    story = texts["report_posthoc"]
    assert f"{rt.posthoc_reads()} reads after the final evaluation" in story and rt.posthoc_reads() >= 18
    assert "\\ref{tab:posthoc}" in story and "\\ref{fig:progress}" in story and story.count("\\paragraph{") == 5
    for label in ("fig:relative", "fig:synthetic-patients", "fig:rr-reference"):
        assert f"\\ref{{{label}}}" in story, label


def test_the_report_discussion_refers_only_to_report_sections(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    import re

    from ecgaug import paper_text as pt
    from ecgaug import report_text as rt

    text = rt.discussion()
    assert set(re.findall(r"\\ref\{([^}]+)\}", text)) <= {"sec:posthoc", "sec:medical"}
    assert "Patient variation, not class imbalance" not in text and "What better results need" not in text
    assert "What the scores mean in the clinic" in text
    best = json.loads((paths.results_for("phase6_seed_ensemble") / "summary.json").read_text())["ensemble"]["with_af_veto"]["cnn/cvae"]
    new = json.loads((paths.results_for("phase6_fresh_patients") / "summary.json").read_text())["ensemble"]["with_af_veto"]["cnn/cvae"]
    for value in (best["se_S"]["point"], best["ppv_S"]["point"], new["se_S"]["point"], new["ppv_S"]["point"]):
        assert pt._n(value, 2) in text
    assert f"about {round(10 * (1 - new['ppv_S']['point']))} of every 10 S flags" in text
    assert pt.discussion() != pt.discussion(brief=True) and "sec:patient-template" in pt.discussion()


def test_the_example_cases_follow_their_rule(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    from ecgaug import report_text as rt

    cases = json.loads((paths.results_for("phase5_clinical_notes") / "cases.json").read_text())["cases"]
    text = rt.example_output()
    for cls in rt.EXAMPLE_CLASSES:
        first = next(c for c in cases if c["payload"]["predicted_class"] == c["true_class"] == cls and not c["payload"]["low_confidence"])
        probs = ", ".join(f"{k} {v:.2f}" for k, v in first["payload"]["probabilities"].items())
        assert probs in text
    assert text.count("\\fbox{") == 2 and "\\label{fig:example}" in text


def test_the_architecture_tables_flow_with_the_same_rows(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    from ecgaug import paper_text as pt

    floating, flowing = pt.architecture_section(), pt.architecture_section(flowing=True)
    assert floating.count("\\begin{table}") == 3 and "longtable" not in floating
    assert flowing.count("\\begin{longtable}") == 3 and "\\begin{table}" not in flowing
    rows = lambda text: sorted(line for line in text.splitlines() if line.endswith("\\\\") and " & " in line and not line.startswith("Model &"))  # noqa: E731
    assert rows(floating) == rows(flowing)


def test_the_progress_figure_builds_from_the_summaries(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")
    from ecgaug import figures

    out_dir = paths.repo_root() / "results" / ".test_figures"
    monkeypatch.setattr(figures.paths, "figures_for", lambda name: out_dir / name)
    try:
        out = figures.report_progress_figure()
        assert len(out) == 1 and out[0].is_file()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_the_medical_figure_reads_training_records_only(full_fake_root, monkeypatch):
    from ecgaug import figures
    from ecgaug.data import raw
    from ecgaug.data.splits import load_approved_split

    shutil.copytree(REPO / "configs", paths.configs_dir(), dirs_exist_ok=True)
    train = set(load_approved_split().train)
    read = []
    real = raw.read_signal
    monkeypatch.setattr(raw, "read_signal", lambda record, *a, **k: (read.append(record), real(record, *a, **k))[1])
    out = figures.medical_background_figure()
    assert len(out) == 1 and out[0].is_file() and out[0].parent.name == "report"
    assert read and set(read) <= train


def test_the_medical_figure_is_skipped_without_raw_records(tmp_root):
    from ecgaug import figures

    shutil.copytree(REPO / "configs", paths.configs_dir(), dirs_exist_ok=True)
    assert figures.medical_background_figure() == []


def test_the_report_inputs_the_report_texts():
    tex = (REPO / "report" / "report.tex").read_text()
    for name in ("report_summary", "report_example", "report_architecture", "report_posthoc", "report_discussion", "report_conclusion"):
        assert f"\\input{{generated/{name}.tex}}" in tex
    assert "report/medical_background.png" in tex and "report/progress.png" in tex
    for label in ("fig:relative", "fig:synthetic-patients", "fig:rr-reference", "sec:posthoc", "sec:medical"):
        assert f"\\label{{{label}}}" in tex, label
