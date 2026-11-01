"""The paper's results sentences and hypothesis verdicts, written from the result files (PLAN Phase 6).

Called by ``python -m ecgaug.paper``. Every sentence is built from a result file, and its wording follows
from the numbers by the pre-registered criteria (PAPER_TRACK RQ1-RQ4, PREREGISTRATION §7): a difference is
called an improvement or a reduction only when its Holm-adjusted two-sided bootstrap p-value is below 0.05;
otherwise it "did not detectably change", and the interval is always given. Exploratory results are
labelled exploratory. A result that does not exist yet is a visible ``\\missing{...}`` placeholder, so the
draft states nothing that the results do not.
"""

from __future__ import annotations

import json

from ecgaug import paths, workflow
from ecgaug.names import FINAL_EVAL, PHASE2_GATE, PHASE4_LEAKAGE_AUDIT, PHASE4_QUALITY_VS_GAIN, PHASE5_CLINICAL_NOTES, PHASE5_EXPLAINABILITY, PHASE5_GATE

CONDITION = {"none": "no augmentation", "class_weighted": "class-weighted loss", "random_oversampling": "random oversampling",
             "smote": "SMOTE", "cvae": "cVAE augmentation", "wgan_gp": "WGAN-GP augmentation"}  # fmt: skip
SHORT = {"none": "none", "class_weighted": "class-weighted loss", "random_oversampling": "random oversampling", "smote": "SMOTE",
         "cvae": "cVAE", "wgan_gp": "WGAN-GP"}  # fmt: skip
GENERATIVE = ("cvae", "wgan_gp")
METHOD = {"gradcam": "Grad-CAM", "integrated_gradients": "integrated gradients", "smoothgrad": "SmoothGrad", "occlusion": "occlusion"}


def _missing(text: str) -> str:
    return f"\\missing{{{text}}}"


def _load(path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _n(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _s(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}".replace("-", "$-$")


def _rho(value: float) -> str:
    """Spearman's rho inside math mode, or 'undefined'."""
    return "undefined" if value != value else f"$\\rho={value:+.2f}$"


def _moved(value: float) -> str:
    """'raised ... by' or 'lowered ... by' for a signed change."""
    return f"raised by {_n(value)}" if value >= 0 else f"lowered by {_n(-value)}"


def _ci(e: dict, signed: bool = False) -> str:
    fmt = _s if signed else _n
    return f"{fmt(e['point'])} (95\\% CI {fmt(e['ci_low'])} to {fmt(e['ci_high'])})"


def _p(c: dict) -> str:
    return f"Holm-adjusted $p={c['p_holm']:.3f}$"


def _changed(c: dict) -> str:
    """The verb for a difference against no augmentation, by the pre-registered test."""
    if c["reject_holm"]:
        return "improved" if c["point"] > 0 else "lowered"
    return "did not detectably change"


def _compared(c: dict) -> str:
    """The verb for a difference between two methods."""
    if c["reject_holm"]:
        return "outperformed" if c["point"] > 0 else "underperformed"
    return "was not detectably different from"


def _arch_name(arch: str) -> str:
    chosen = (workflow.manifest().get(PHASE2_GATE) or {}).get(arch)
    return f"the {arch} ({chosen['model']['name'].replace('_', ' ')})" if chosen else f"the {arch}"


# --------------------------------------------------------------------------------------------------
# RQ1: the final evaluation
# --------------------------------------------------------------------------------------------------


def rq1(summary: dict | None) -> tuple[str, dict]:
    """RQ1 paragraphs and the H1, H1b and H1c verdicts."""
    if summary is None:
        return _missing("test results: written after /final-eval") + "\n", {}
    conditions, families = summary["conditions"], summary["families"]
    archs = sorted({key.split("/")[0] for key in conditions})
    lines, h1_hits, h1b, h1c = [], [], [], {}
    for arch in archs:
        base = conditions.get(f"{arch}/none")
        if base is None:
            continue
        text = [f"For {_arch_name(arch)}, training without augmentation reached a test rare-class macro-F1 of {_ci(base['rare_macro_f1'])}."]
        for gen in GENERATIVE:
            c = families["confirmatory"].get(f"{arch}: {gen} vs none")
            entry = conditions.get(f"{arch}/{gen}")
            if c is None or entry is None:
                continue
            strength = f" (strength {entry['strength']:g})" if entry["strength"] is not None else ""
            text.append(f"{CONDITION[gen]}{strength} {_changed(c)} it: difference {_ci(c, signed=True)}, {_p(c)}.")
            if c["reject_holm"] and c["point"] > 0:
                h1_hits.append(f"{SHORT[gen]} with the {arch}")
            d_f = entry["f1_F"]["point"] - base["f1_F"]["point"]
            d_s = entry["f1_S"]["point"] - base["f1_S"]["point"]
            h1b.append((arch, gen, d_f, d_s))
        lines.append(" ".join(text))
        best = (summary.get("phase4_gate") or {}).get("best_generative", {}).get(arch)
        secondary = []
        for name, c in families.get("secondary", {}).items():
            a_arch, rest = name.split(": ")
            if a_arch != arch:
                continue
            a, b = rest.split(" vs ")
            secondary.append(f"{SHORT[a]} {_compared(c)} {SHORT[b]} ({_ci(c, signed=True)}, {_p(c)})")
            if a == best and b != "wgan_gp" and b != "cvae":
                h1c[(arch, b)] = bool(c["reject_holm"] and c["point"] > 0)
        if secondary:
            lines.append(f"In the secondary family for the {arch} (best generative condition on validation: {SHORT.get(best, best)}), " + "; ".join(secondary) + ".")
    verdict_h1 = "supported" if h1_hits else "not supported"
    lines.append(f"H1 (confirmatory) is {verdict_h1}" + (f": significant improvement for {', '.join(h1_hits)}." if h1_hits else
                 ": no generative condition improved the primary metric after Holm correction."))  # fmt: skip
    without = summary["sensitivity"]["without_record_202"]["confirmatory"]
    flipped = [name for name, c in without.items() if c["reject_holm"] != families["confirmatory"][name]["reject_holm"]]
    lines.append("Without record 202 (the DS2 record of the subject of DS1 record 201) the confirmatory conclusions are "
                 + (f"different for {', '.join(flipped).replace('_', ' ')}." if flipped else "unchanged."))  # fmt: skip
    if h1b:
        parts = [f"{SHORT[g]} with the {a}: F {_s(df)}, S {_s(ds)}" for a, g, df, ds in h1b]
        lines.append("Per class (exploratory, H1b; differences of test F1 point estimates against no augmentation): " + "; ".join(parts) + ".")
    larger_f = sum(df > ds for _, _, df, ds in h1b)
    verdicts = {
        "H1": (verdict_h1, ", ".join(h1_hits) or "no confirmatory comparison significant"),
        "H1b": ("consistent" if h1b and larger_f == len(h1b) else "mixed" if larger_f else "not consistent",
                f"F gain larger than S gain in {larger_f} of {len(h1b)} generative conditions"),
        "H1c": ("supported" if h1c and any(all(v for (a, _), v in h1c.items() if a == arch) for arch in {a for a, _ in h1c})
                else "partly supported" if any(h1c.values()) else "not supported",
                f"{sum(h1c.values())} of {len(h1c)} comparisons of the best generative condition with a baseline significant"),
    }  # fmt: skip
    return "\n\n".join(lines).replace("_", "\\_") + "\n", verdicts


# --------------------------------------------------------------------------------------------------
# RQ2: the leakage audit
# --------------------------------------------------------------------------------------------------


def rq2(audit: dict | None) -> tuple[str, dict]:
    if audit is None:
        return _missing("leakage audit: optional, runs after the final evaluation") + "\n", {}
    none = audit["conditions"].get("none", {})
    if not (none.get("intra") and none.get("inter")):
        return _missing("leakage audit: intra- or inter-patient result missing") + "\n", {}
    split = audit.get("split") or {}
    lines = [f"With the same pipeline on a random beat-level split that pools DS1 and DS2 ({split.get('records_in_train_and_test', '?')} of "
             f"{split.get('records', '?')} records have beats in both training and test), the {audit['architecture']} without augmentation reached a "
             f"rare-class macro-F1 of {_n(none['intra']['mean'])} $\\pm$ {_n(none['intra']['sd'])} (mean $\\pm$ SD over seeds), against "
             f"{_ci(none['inter'])} under the inter-patient protocol: evaluated intra-patient, the primary metric was {_moved(audit['inflation_none'])}."]  # fmt: skip
    larger = []
    for gen, g in audit["gains"].items():
        if g["intra"] and g["inter"]:
            lines.append(f"The apparent gain of {SHORT[gen]} over none was {_s(g['intra']['mean'])} intra-patient and {_ci(g['inter'], signed=True)} inter-patient.")
            larger.append(g["intra"]["mean"] > g["inter"]["point"])
    consistent = audit["inflation_none"] > 0 and larger and all(larger)
    verdict = ("consistent" if consistent else "not consistent", f"inflation of the baseline {_s(audit['inflation_none'])}; larger intra-patient gain for {sum(larger)} of {len(larger)} generators")
    return " ".join(lines) + "\n", {"H2": verdict}


# --------------------------------------------------------------------------------------------------
# RQ3: quality metrics against the gain
# --------------------------------------------------------------------------------------------------


def rq3(summary: dict | None) -> tuple[str, dict]:
    if summary is None:
        return _missing("RQ3: python -m ecgaug.eval.rq3 after the phase 3 runs") + "\n", {}
    metrics = summary["metrics"]

    def name(m: str) -> str:
        return m.removeprefix("quality_").removesuffix("_mean").replace("_", " ")

    def show(m: str) -> str:
        e = metrics[m]
        ci = f", 95\\% CI {_s(e['ci_low'], 2)} to {_s(e['ci_high'], 2)}" if e["ci_low"] is not None else ""
        return f"{name(m)} ({_rho(e['rho'])}{ci})"

    main = [m for m, e in metrics.items() if e["group"] != "tstr"]
    found = [m for m in main if metrics[m].get("excludes_zero")]
    lines = [f"Across {summary['n_checkpoints']} checkpoints of {summary['n_runs']} generator runs (exploratory; intervals resample runs), "
             + (f"the interval excluded zero for {', '.join(show(m) for m in found)}." if found else "no fidelity or diversity metric had an interval excluding zero.")
             + " The others: " + ", ".join(show(m) for m in main if m not in found) + "."]  # fmt: skip
    for m in (m for m, e in metrics.items() if e["group"] == "tstr"):
        lines.append(f"TSTR, analysed apart because it and the gain are scores of classifiers trained on the same synthetic beats: {show(m)}.")
    fid, diversity = metrics.get("quality_fid_analog"), metrics.get("diversity_ratio_mean")
    note = ""
    if fid and diversity and fid["rho"] == fid["rho"] and diversity["rho"] == diversity["rho"]:
        weaker = abs(fid["rho"]) < abs(diversity["rho"])
        note = f"; FID-analog $|\\rho|$ {'below' if weaker else 'not below'} diversity $|\\rho|$"
        lines.append(f"The fidelity-only FID-analog correlated {'more weakly' if weaker else 'not more weakly'} than the diversity ratio.")
    if summary.get("provisional"):
        lines.append("These numbers come from provisional (preview) generator runs.")
    verdict = ("supported" if found else "not supported", f"{len(found)} of {len(main)} metrics with an interval excluding zero{note}")
    return " ".join(lines) + "\n", {"H3": verdict}


# --------------------------------------------------------------------------------------------------
# RQ4 and the notes
# --------------------------------------------------------------------------------------------------


def _segment_mean(alignment: dict, segment: str, classes=("S", "V", "F")) -> float | None:
    values = [alignment.get(segment, {}).get(c) for c in classes]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def rq4(gate: dict | None, explain: dict[str, dict]) -> tuple[str, dict]:
    if gate is None or not explain:
        return _missing("RQ4: the phase 5 gate") + "\n", {}
    lines, at_least, total = [], 0, 0
    for arch, method in gate["explainability"].items():
        result = next((r for r in explain.values() if r["method"] == method), None)
        entry = (result or {}).get("per_architecture", {}).get(arch)
        if not entry or not entry.get("applicable"):
            continue
        cond = entry["conditions"]
        gains = {c: v["faithfulness_gain"] for c, v in cond.items()}
        text = [f"For the {arch}, the rule chose {METHOD.get(method, method)}. Its faithfulness gain over random deletion was "
                + ", ".join(f"{_s(g, 4)} for {SHORT[c]}" for c, g in gains.items()) + "."]  # fmt: skip
        for c in GENERATIVE:
            if c in gains and "none" in gains:
                total += 1
                at_least += gains[c] >= gains["none"]
            synthetic = cond.get(c, {}).get("alignment_on_synthetic")
            real = cond.get(c, {}).get("alignment")
            if isinstance(synthetic, dict) and "skipped" not in synthetic and real:
                qrs_syn, qrs_real = _segment_mean(synthetic, "QRS"), _segment_mean(real, "QRS")
                if qrs_syn is not None and qrs_real is not None:
                    text.append(f"The {SHORT[c]} model put {_n(qrs_syn, 2)} of its saliency mass in the QRS window on synthetic S, V and F beats against "
                                f"{_n(qrs_real, 2)} on real ones{' (a shift that may indicate reliance on synthetic artifacts)' if abs(qrs_syn - qrs_real) > 0.1 else ''}.")  # fmt: skip
        lines.append(" ".join(text))
    verdict = ("consistent" if total and at_least == total else "mixed" if at_least else "not consistent",
               f"augmented faithfulness at least the baseline's in {at_least} of {total} model pairs")  # fmt: skip
    return "\n\n".join(lines) + "\n", {"H4": verdict}


def notes(gate: dict | None, results: dict[str, dict]) -> str:
    if gate is None or not results:
        return _missing("note generators: the phase 5 gate") + "\n"
    chosen = results.get(gate["notes_variant"], {})
    ran = [l for l, r in results.items() if not r.get("not_run")]
    skipped = {l: r.get("reason", "") for l, r in results.items() if r.get("not_run")}
    text = (f"The rule chose note generator {gate['notes_variant']} ({chosen.get('description', '').rstrip('.')}). Over {chosen.get('n_cases')} validation "
            f"cases, {chosen.get('n_low_confidence')} of them low-confidence, it produced {chosen.get('unsupported_claims')} unsupported claims, with "
            f"coverage {_n(chosen.get('coverage', float('nan')), 2)} of the required parts.")  # fmt: skip
    if skipped:
        text += f" Variants {', '.join(skipped)} were not run ({next(iter(skipped.values()))})."
    text += f" Variants run: {', '.join(ran)}."
    return text + "\n"


# --------------------------------------------------------------------------------------------------
# the hypothesis table and the abstract sentence
# --------------------------------------------------------------------------------------------------

HYPOTHESES = {
    "H1": "confirmatory: a generative condition improves test rare-class macro-F1 over none (Holm, 4 tests)",
    "H1b": "exploratory: gains larger for F than for S",
    "H1c": "secondary: generative beats SMOTE, random oversampling and class-weighted loss",
    "H2": "exploratory: intra-patient evaluation inflates the baseline and the gains",
    "H3": "exploratory: a quality metric predicts the downstream gain",
    "H4": "exploratory: augmented models are at least as faithful as the baseline",
}


def hypotheses_table(verdicts: dict) -> str:
    rows = [f"{h} & {text.replace('_', ' ')} & {verdicts[h][0] if h in verdicts else _missing('pending')} & "
            f"{verdicts[h][1].replace('_', ' ') if h in verdicts else ''} \\\\" for h, text in HYPOTHESES.items()]  # fmt: skip
    return "\n".join(["% Generated by python -m ecgaug.paper from the result files", "\\begin{tabular}{lp{5.2cm}lp{5.2cm}}", "\\toprule",
                      "Hypothesis & Statement & Verdict & Evidence \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]) + "\n"  # fmt: skip


def abstract_sentence(summary: dict | None, verdicts: dict, audit: dict | None) -> str:
    if summary is None:
        return _missing("results sentence: written after /final-eval")
    families = summary["families"]["confirmatory"]
    best_name, best = max(families.items(), key=lambda kv: kv[1]["point"])
    arch, rest = best_name.split(": ")
    gen = rest.split(" vs ")[0]
    text = (f"Under the pre-registered analysis, generative augmentation {'improved' if verdicts['H1'][0] == 'supported' else 'did not detectably improve'} "
            f"test rare-class macro-F1 over no augmentation (largest difference: {SHORT[gen]} with the {arch}, {_ci(best, signed=True)}, {_p(best)}).")  # fmt: skip
    if audit and audit.get("inflation_none") is not None:
        text += f" Evaluated on a random beat-level split instead, the baseline's primary metric was {_moved(audit['inflation_none'])}."
    return text


def build() -> dict[str, str]:
    """Every generated text file: name -> LaTeX."""
    summary = _load(paths.results_for(FINAL_EVAL) / "summary.json")
    audit = _load(paths.results_for(PHASE4_LEAKAGE_AUDIT) / "summary.json")
    quality = _load(paths.results_for(PHASE4_QUALITY_VS_GAIN) / "summary.json")
    gate = workflow.manifest().get(PHASE5_GATE)
    explain = {k: v for k, v in workflow.experiment_results(PHASE5_EXPLAINABILITY).items() if v}
    note_results = {k: v for k, v in workflow.experiment_results(PHASE5_CLINICAL_NOTES).items() if v}
    verdicts: dict = {}
    files = {}
    for name, (text, found) in {"results_rq1": rq1(summary), "results_rq2": rq2(audit), "results_rq3": rq3(quality), "results_rq4": rq4(gate, explain)}.items():
        files[name] = text
        verdicts.update(found)
    files["results_notes"] = notes(gate, note_results)
    files["table_hypotheses"] = hypotheses_table(verdicts)
    files["abstract_results"] = abstract_sentence(summary, verdicts, audit) + "\n"
    return files
