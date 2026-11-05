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


def concentration(errors: dict | None) -> str:
    """Where the test S and F beats sit and how often each condition finds them (results/error_analysis/test.json, exploratory)."""
    if not errors:
        return _missing("per-record test analysis: python -m ecgaug.report.error_analysis --test") + "\n"
    first = next(iter(errors.values()))
    parts, recalls = [], []
    for cls in ("S", "F"):
        support = {rec: v[cls]["support"] for rec, v in first["per_record"].items() if cls in v}
        top = max(support, key=support.get)
        recall = [1 - e["per_record"][top][cls]["missed_fraction"] for e in errors.values()]
        parts.append(f"record {top} holds {support[top]:,} of the {sum(support.values()):,} test {cls} beats")
        recalls.append(f"{cls} recall on record {top} ranged from {_n(min(recall), 2)} to {_n(max(recall), 2)}")
    text = (f"The test rare beats sit in very few patients: {parts[0]}, and {parts[1]}. Across the {len(errors)} models of Table~\\ref{{tab:test}} "
            f"(seed means), {recalls[0]}, and {recalls[1]}, so these two records set most of the primary metric.")  # fmt: skip
    base = errors.get("cnn/none")
    if base:
        cm = base["confusion_matrix"]
        s, f = 1, 3
        text += (f" The cnn without augmentation labelled {cm[s][0]:,.0f} of {sum(cm[s]):,.0f} S beats and {cm[f][0]:,.0f} of {sum(cm[f]):,.0f} F beats "
                 f"as normal (exploratory).")  # fmt: skip
        rr = base["rr_context"]
        if rr.get("S correct", {}).get("rr_ratio_median") and rr.get("S called N", {}).get("rr_ratio_median"):
            text += (f" With seed 0, S beats it found were more premature than S beats it called normal (median ratio of the previous RR interval to the "
                     f"local mean {_n(rr['S correct']['rr_ratio_median'], 2)} against {_n(rr['S called N']['rr_ratio_median'], 2)}).")  # fmt: skip
    return text + "\n"


# --------------------------------------------------------------------------------------------------
# post-hoc analyses (PREREGISTRATION v4, P1-P4): reported apart, never replacing the primary results
# --------------------------------------------------------------------------------------------------

DATABASE = {"incartdb": "St Petersburg INCART", "svdb": "MIT-BIH Supraventricular Arrhythmia"}
METRIC = {"rare_macro_f1": "rare-class macro-F1", "f1_S": "S F1"}


def _against(diff: dict, other: str) -> str:
    """Reference minus cnn, worded by whether the 95% interval excludes zero (P1 rule)."""
    if diff["ci_low"] > 0:
        return f"the {other} was weaker than this reference"
    if diff["ci_high"] < 0:
        return f"this reference was weaker than the {other}"
    return f"this reference was not detectably different from the {other}"


def posthoc_rr(summary: dict | None, final: dict | None) -> str:
    if summary is None or final is None:
        return _missing("P1 RR reference: python -m ecgaug.experiments.phase6_rr_reference --part test")
    parts = []
    for name, e in summary["references"].items():
        d, tt = e["vs_cnn"]["none"], name.replace("_", "\\_")
        parts.append(f"\\texttt{{{tt}}} reached {_ci(e['rare_macro_f1'])} (S Se {_n(e['se_S']['point'])}, S +P {_n(e['ppv_S']['point'])}); "
                     f"minus the cnn without augmentation, {_ci(d, signed=True)}, so {_against(d, 'cnn without augmentation')}")  # fmt: skip
    weaker = [r for r, e in summary["references"].items() if e["vs_cnn"]["none"]["cnn_weaker"]]
    reading = ("Neither reference beat the cnn, so its low test score is not explained by a weak network that a simple RR rule would beat."
               if not weaker else f"The cnn without augmentation was weaker than {', '.join(weaker)} by the registered rule.")  # fmt: skip
    return (f"\\paragraph{{A fixed RR-interval reference (P1).}} Two linear discriminants with uniform priors and nothing to tune were fitted once on the "
            f"training records: \\texttt{{rr\\_lda}} on the three RR features, \\texttt{{rr\\_morph\\_lda}} on those plus every tenth sample of the beat window. "
            f"On the test records, {'; '.join(parts)} (Figure~\\ref{{fig:rr-reference}}). {reading} Both references are cruder than published feature sets.")  # fmt: skip


def posthoc_external(summaries: dict[str, dict]) -> str:
    if not summaries:
        return _missing("P2 external validation: python -m ecgaug.experiments.phase6_external_validation --part evaluate")
    parts = []
    for db, s in summaries.items():
        metric, beats = s["primary_metric"], s["beats"]
        base = "; ".join(f"the {arch} without augmentation reached {_ci(s['conditions'][f'{arch}/none'][metric])}"
                         for arch in dict.fromkeys(k.split("/")[0] for k in s["conditions"]))  # fmt: skip
        comparisons = "; ".join(f"{name.replace('_', ' ')} {_ci(c, signed=True)}, {_p(c)} ({_changed(c)})" for name, c in s["confirmatory"].items())
        rejected = [k for k, v in s["confirmatory"].items() if v["reject_holm"]]
        context = []
        for arch in dict.fromkeys(k.split(": ")[0] for k in rejected):  # the same architecture's other conditions on this database
            others = {c.split("/")[1]: v[metric]["point"] for c, v in s["conditions"].items() if c.startswith(f"{arch}/")}
            ranked = ", ".join(f"{SHORT[c]} {_n(v)}" for c, v in sorted(others.items(), key=lambda kv: -kv[1]))
            context.append(f"the {arch}'s conditions ranked by {METRIC[metric]}: {ranked}")
        same = all(v["replicated"] for v in s["confirmatory"].values())
        text = (f"On {DATABASE[db]} ({s['role']}; {s['records']} records; {beats['S']:,} S and {beats['F']:,} F beats; primary metric {METRIC[metric]}), "
                f"{base}. Against no augmentation, the confirmatory differences were: {comparisons}. Holm rejected {len(rejected)} of "
                f"{len(s['confirmatory'])}, which {'matches' if same else 'does not match'} the verdicts on the MIT-BIH test records")  # fmt: skip
        if context:
            text += f"; {'; '.join(context)}"
        parts.append(text)
    return (f"\\paragraph{{External databases (P2).}} Every frozen checkpoint was applied unchanged to two databases no model or decision had seen, "
            f"with beats cut exactly as in the frozen dataset. {'. '.join(parts)} (Figure~\\ref{{fig:external}}). The databases differ from "
            f"MIT-BIH in sampling rate, lead and annotation practice (Figure~\\ref{{fig:external-beats}}), so these results measure transfer as well as "
            f"augmentation.")  # fmt: skip


def posthoc_seeds(summary: dict | None) -> str:
    if summary is None:
        return _missing("P3 extra seeds: python -m ecgaug.experiments.phase6_extra_seeds --part test")
    conf = summary["families"]["confirmatory"]
    best_name, best = max(conf.items(), key=lambda kv: kv[1]["point"])
    changed = summary["changed"]
    return (f"\\paragraph{{Five seeds (P3).}} Seeds 3 and 4 were trained with the frozen configurations, strengths and generators. With five seeds, "
            f"the largest confirmatory difference was {best_name.replace('_', ' ')}, {_ci(best, signed=True)}, {_p(best)}; "
            f"{'no Holm verdict changed' if not changed else 'verdicts that changed: ' + ', '.join(c.replace('_', ' ') for c in changed)} "
            f"(Figure~\\ref{{fig:seeds}}).")  # fmt: skip


def posthoc_audit(summary: dict | None) -> str:
    if summary is None:
        return _missing("P4 audit checks: python -m ecgaug.experiments.phase6_audit_check")
    parts = []
    for c in summary["checks"]:
        what = f"no augmentation on random split {c['split_seed']}" if c["condition"] == "none" else f"WGAN-GP at strength {c['strength']} on the audit's split"
        parts.append(f"{what}: {_n(c['intra']['mean'])} $\\pm$ {_n(c['intra']['sd'])}")
    base = summary["audit_split0"]["none"]
    return (f"\\paragraph{{Checks of the audit (P4).}} Intra-patient rare-class macro-F1 of the cnn (mean $\\pm$ SD over seeds): "
            f"{'; '.join(parts)}, against {_n(base['mean'])} $\\pm$ {_n(base['sd'])} without augmentation on the audit's split and "
            f"{_n(summary['inter_patient']['none']['point'])} inter-patient (Figure~\\ref{{fig:audit-checks}}). {_audit_reading(summary)} Exploratory.")  # fmt: skip


def _audit_reading(summary: dict) -> str:
    """What the checks say about the audit, from the numbers: the inflation on every split, and the gain at the chosen strength."""
    inter = summary["inter_patient"]["none"]["point"]
    nones = [summary["audit_split0"]["none"]["mean"], *(c["intra"]["mean"] for c in summary["checks"] if c["condition"] == "none")]
    text = f"Without augmentation, every random split was {_n(min(nones) - inter)} or more above the inter-patient value."
    base = summary["audit_split0"]["none"]["mean"]
    for c in summary["checks"]:
        if c["condition"] != "none":
            gain = c["intra"]["mean"] - base
            inter_gain = summary["inter_patient"][c["condition"]]["point"] - inter
            reading = ("no larger than the inter-patient difference, so the audit's verdict does not depend on its strength" if gain <= inter_gain
                       else "larger than the inter-patient difference, so at this strength the audit's verdict would differ")  # fmt: skip
            text += (f" At the chosen strength, {SHORT[c['condition']]} {'lowered' if gain < 0 else 'raised'} the intra-patient mean by {_n(abs(gain))} "
                     f"({_s(gain)} against {_s(inter_gain)} inter-patient), {reading}.")  # fmt: skip
    return text


def _family_sentence(items: dict) -> str:
    return "; ".join(f"{name.replace('_', ' ')} {_ci(c, signed=True)}, {_p(c)} ({_changed(c)})" for name, c in items.items())


def posthoc_rr_baseline() -> str:
    """P5: the full grid with the patient-baseline RR ratio, from results/phase6_rr_baseline/."""
    from ecgaug.names import PHASE6_RR_BASELINE

    folder = paths.results_for(PHASE6_RR_BASELINE)
    data, selection, summary = (_load(folder / n) for n in ("data.json", "selection.json", "summary.json"))
    if data is None:
        return _missing("P5: python -m ecgaug.experiments.phase6_rr_baseline --part data")
    d = data["diagnostic"]["train"]
    text = (f"\\paragraph{{A patient-baseline RR ratio (P5).}} The v1 ratio divides the previous RR interval by the mean of the ten intervals "
            f"before it, so an early beat inside a run of early beats looks regular. On the training records, {_n(d['S']['below_0.9_v1'], 2)} of "
            f"S beats had a ratio below 0.9 with v1 and {_n(d['S']['below_0.9_v2'], 2)} with a ratio to the median RR within "
            f"{data['half_window_s']:g}~s (N beats: {_n(d['N']['below_0.9_v1'], 2)} and {_n(d['N']['below_0.9_v2'], 2)}; "
            f"Figure~\\ref{{fig:rr-ratio}}). Version 5 of the pre-registration changed only this field (dataset v2), remapped the ratio of "
            f"the frozen synthetic beats, reran the full grid with the frozen classifiers and selected on validation with the phase 4 gate rule.")  # fmt: skip
    if selection is None or summary is None:
        return text + " " + _missing("P5 results: --part grid on Kaggle, then --part select and --part test") + "\n"
    strengths = "; ".join(f"{arch}: " + ", ".join(f"{SHORT[c]} {v['strength']}" for c, v in conds.items() if v["strength"] is not None)
                          for arch, conds in selection["chosen"].items())  # fmt: skip
    nones = "; ".join(f"the {arch} without augmentation reached {_ci(summary['conditions'][f'{arch}/none']['rare_macro_f1'])} "
                      f"(v2 minus v1 {_ci(summary['versus_v1'][f'{arch}/none'], signed=True)})" for arch in selection["chosen"])  # fmt: skip
    best_key = max(summary["conditions"], key=lambda k: summary["conditions"][k]["rare_macro_f1"]["point"])
    best_arch, best_cond = best_key.split("/")
    best_name = "no augmentation" if best_cond == "none" else SHORT[best_cond]
    diffs = summary["versus_v1"]
    up = [k for k, c in diffs.items() if c["ci_low"] > 0]
    down = [k for k, c in diffs.items() if c["ci_high"] < 0]
    span = f"{_s(min(c['point'] for c in diffs.values()))} to {_s(max(c['point'] for c in diffs.values()))}"
    conf = summary["families"]["confirmatory"]
    improved = [k for k, c in conf.items() if c["reject_holm"] and c["point"] > 0]
    secondary = {k: c for k, c in summary["families"]["secondary"].items() if c["reject_holm"]}
    text += (f" The rule chose the strengths {strengths}. On the test records with v2 features (Figure~\\ref{{fig:v2-results}}), {nones}. "
             f"Across the {len(diffs)} conditions the paired difference v2 minus v1 ranged from {span}; "
             + (f"its 95\\% CI was above zero in {len(up)} and below zero in {len(down)}." if up or down else
                "no 95\\% CI excluded zero, so the new ratio did not detectably change the networks' primary metric.")
             + f" The highest score was the {best_arch} with {best_name}, {_ci(summary['conditions'][best_key]['rare_macro_f1'])}. "
             f"Against no augmentation: {_family_sentence(conf)}. "
             f"By the registered criterion, generative augmentation {'improved' if improved else 'did not detectably improve'} the primary metric with v2 "
             f"features. {'In the secondary family Holm rejected ' + _family_sentence(secondary) + '.' if secondary else 'No secondary comparison was rejected.'}")  # fmt: skip
    refs = summary.get("references", {})
    if "rr_lda" in refs:
        from ecgaug.names import PHASE6_RR_REFERENCE

        r = refs["rr_lda"]
        old = _load(paths.results_for(PHASE6_RR_REFERENCE) / "summary.json")
        before = f" (v1: {_n(old['references']['rr_lda']['rare_macro_f1']['point'])}, S Se {_n(old['references']['rr_lda']['se_S']['point'])})" if old else ""
        text += (f" The RR-only reference refitted on v2 reached {_ci(r['rare_macro_f1'])} (S Se {_n(r['se_S']['point'])}, "
                 f"S +P {_n(r['ppv_S']['point'])}){before}.")  # fmt: skip
    ext = []
    for db in DATABASE:
        e = _load(folder / f"{db}_summary.json")
        if e:
            metric = e["primary_metric"]
            verbs = [_changed(c) for c in e["confirmatory"].values()]
            ext.append(f"on {DATABASE[db]} the cnn without augmentation reached {_n(e['conditions']['cnn/none'][metric]['point'])} ({METRIC[metric]}) and "
                       + ", ".join(f"generative augmentation {v} it in {verbs.count(v)}" for v in dict.fromkeys(verbs)) + " of the 4 confirmatory comparisons")
    if ext:
        text += " Externally, " + "; ".join(ext) + " (Figure~\\ref{fig:v2-external})."
    return text + " These results are post hoc and do not replace the primary results of Table~\\ref{tab:test}.\n"


def abstract_posthoc() -> str:
    """Abstract sentences for the post-hoc analyses (P2 primary database, P3, P5), from their summaries."""
    from ecgaug.names import PHASE6_EXTERNAL_VALIDATION, PHASE6_EXTRA_SEEDS

    incart = _load(paths.results_for(PHASE6_EXTERNAL_VALIDATION) / "incartdb_summary.json")
    seeds = _load(paths.results_for(PHASE6_EXTRA_SEEDS) / "summary.json")
    if incart is None or seeds is None:
        return _missing("post-hoc abstract sentence: after P2 and P3")
    conf = incart["confirmatory"]
    verbs = [_changed(c) for c in conf.values()]
    counts = ", ".join(f"{verb} {'the rare-class macro-F1' if i == 0 else 'it'} in {verbs.count(verb)}" for i, verb in enumerate(dict.fromkeys(verbs)))
    rejected = [c["p_holm"] for c in conf.values() if c["reject_holm"]]
    bound = f" (Holm-adjusted $p \\le {max(rejected):.3f}$)" if rejected and len(set(verbs)) == 1 else ""
    changed = "no pre-registered verdict" if not seeds["changed"] else f"{len(seeds['changed'])} pre-registered verdicts"
    text = (f"In analyses added after the final evaluation, the frozen models were applied to the St Petersburg INCART database, where generative "
            f"augmentation {counts} of the {len(conf)} confirmatory comparisons with no augmentation{bound}; two more seeds changed {changed}.")  # fmt: skip
    from ecgaug.names import PHASE6_RR_BASELINE

    v5 = _load(paths.results_for(PHASE6_RR_BASELINE) / "summary.json")
    if v5 is not None:
        diffs = v5["versus_v1"].values()
        moved = sum(c["ci_low"] > 0 or c["ci_high"] < 0 for c in diffs)
        improved = any(c["reject_holm"] and c["point"] > 0 for c in v5["families"]["confirmatory"].values())
        text += (f" Rerunning the full grid with an RR feature measured against each patient's baseline "
                 f"{'changed' if moved else 'did not detectably change'} the networks' scores (v2 minus v1 from "
                 f"{_s(min(c['point'] for c in diffs))} to {_s(max(c['point'] for c in diffs))}) and "
                 f"{'did' if improved else 'did not'} make generative augmentation detectably better than none.")  # fmt: skip
    return text


def posthoc(final: dict | None) -> str:
    from ecgaug.names import PHASE6_AUDIT_CHECK, PHASE6_EXTERNAL_VALIDATION, PHASE6_EXTRA_SEEDS, PHASE6_RR_REFERENCE

    external = {db: s for db in DATABASE if (s := _load(paths.results_for(PHASE6_EXTERNAL_VALIDATION) / f"{db}_summary.json"))}
    return "\n\n".join([posthoc_rr(_load(paths.results_for(PHASE6_RR_REFERENCE) / "summary.json"), final), posthoc_external(external),
                        posthoc_seeds(_load(paths.results_for(PHASE6_EXTRA_SEEDS) / "summary.json")),
                        posthoc_audit(_load(paths.results_for(PHASE6_AUDIT_CHECK) / "summary.json"))]) + "\n"  # fmt: skip


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
    files["results_concentration"] = concentration(_load(paths.results_for("error_analysis") / "test.json"))
    files["results_posthoc"] = posthoc(summary)
    files["abstract_posthoc"] = abstract_posthoc() + "\n"
    files["results_rr_baseline"] = posthoc_rr_baseline()
    files["table_hypotheses"] = hypotheses_table(verdicts)
    files["abstract_results"] = abstract_sentence(summary, verdicts, audit) + "\n"
    return files
