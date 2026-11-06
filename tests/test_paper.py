"""Phase 6: the paper's tables and numbers come from result files; missing results are marked, never invented."""

import json
import shutil

import pytest

from conftest import REPO
from ecgaug import paper, paths


def test_paper_tables_come_from_results_and_mark_what_is_missing(tmp_root):
    shutil.copytree(REPO / "configs", paths.configs_dir())
    shutil.copytree(REPO / "results" / "phase1_input_representation", paths.results_for("phase1_input_representation"))
    shutil.copytree(REPO / "results" / "phase1_data", paths.results_for("phase1_data"))
    report = paper.build()
    generated = paths.repo_root() / "paper" / "generated"
    table = (generated / "table_input_representation.tex").read_text()
    c = json.loads((REPO / "results" / "phase1_input_representation" / "C.json").read_text())
    assert f"{c['mean']:.3f} $\\pm$ {c['sd']:.3f}" in table and "results/phase1_input_representation/C.json" in table
    assert "table_baselines" in report["missing"] and "\\missing{" in (generated / "table_baselines.tex").read_text()
    numbers = (generated / "numbers.tex").read_text()
    assert "\\newcommand{\\CountValidationF}{27}" in numbers and "\\newcommand{\\PhaseOneInputRepresentationCMean}" in numbers
    assert "final_eval_comparisons.tex" in report["missing"] and not report["final_evaluation"]
    assert (paths.repo_root() / "paper" / "claims_vs_evidence.md").is_file()


def test_macro_names_are_letters_only():
    assert paper._macro("count training S") == "CountTrainingS"
    assert paper._macro("phase1_input_representation C mean") == "PhaseOneInputRepresentationCMean"


@pytest.mark.skipif(not (REPO / "paper" / "main.tex").is_file(), reason="the paper draft is kept out of the repository")
def test_the_draft_cites_only_verified_entries_and_inputs_only_generated_files():
    import re

    from conftest import REPO

    from ecgaug import paper

    tex = (REPO / "paper" / "main.tex").read_text() + (REPO / "paper" / "report.tex").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", (REPO / "paper" / "refs.bib").read_text()))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]*)\}", tex) for k in group.split(",")}
    assert cited and cited <= keys, f"cited but not in refs.bib: {cited - keys}"
    report = paper.build()
    inputs = set(re.findall(r"\\input\{generated/([^}]*)\}", tex))
    generated = paths.repo_root() / "paper" / "generated"
    assert all((generated / (i if i.endswith(".tex") else f"{i}.tex")).is_file() for i in inputs)
    for name in report["written"]:
        text = (paths.repo_root() / name).read_text()
        assert text.count("{") == text.count("}"), f"unbalanced braces in {name}"
