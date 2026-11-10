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

import numpy as np

from ecgaug import paths, workflow
from ecgaug.names import FINAL_EVAL, PHASE2_GATE, PHASE4_LEAKAGE_AUDIT, PHASE4_QUALITY_VS_GAIN, PHASE5_CLINICAL_NOTES, PHASE5_EXPLAINABILITY, PHASE5_GATE

CONDITION = {"none": "no augmentation", "class_weighted": "class-weighted loss", "random_oversampling": "random oversampling",
             "smote": "SMOTE", "cvae": "cVAE augmentation", "wgan_gp": "WGAN-GP augmentation"}  # fmt: skip
SHORT = {"none": "none", "class_weighted": "class-weighted loss", "random_oversampling": "random oversampling", "smote": "SMOTE",
         "cvae": "cVAE", "wgan_gp": "WGAN-GP", "diffusion": "diffusion"}  # fmt: skip
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


PHYSICS_SHORT = {"cvae_physics": "cVAE with the Euler loss", "wgan_gp_physics": "WGAN-GP with the Euler loss", "cvae_fusion": "cVAE with fusion F beats"}
PHYSICS_BASE = {"cvae_physics": "cvae", "wgan_gp_physics": "wgan_gp", "cvae_fusion": "cvae"}


def _physics_family(items: dict) -> str:
    renamed = {}
    for name, v in items.items():
        for c, label in PHYSICS_SHORT.items():
            name = name.replace(c, label)
        renamed[name.replace("cnn: ", "").replace("wgan_gp", "WGAN-GP").replace("cvae", "cVAE")] = v
    return "; ".join(f"{name} {_ci(c, signed=True)}, {_p(c)} ({_changed(c)})" for name, c in renamed.items())


def posthoc_physics_prior() -> str:
    """P21: McSharry et al.'s simulator as a prior for the generators (SimGANs' Euler loss) and physiological fusion beats."""
    from ecgaug.names import PHASE6_PHYSICS_PRIOR

    folder = paths.results_for(PHASE6_PHYSICS_PRIOR)
    physics, s, test, fusion = (_load(folder / n) for n in ("physics.json", "summary.json", "test_summary.json", "fusion.json"))
    fams = {f: d for f in ("cvae", "wgan_gp") if (d := _load(folder / f"{f}.json")) is not None}
    text = ("\\paragraph{Physics in the generators (P21).} SimGANs \\cite{golany2020simgans} add McSharry et al.'s ECG simulator "
            "\\cite{mcsharry2003dynamical}, three differential equations, to a GAN's generator objective: the Euler loss measures how far a generated "
            "beat is from satisfying the simulator's equation for the ECG, for simulator parameters drawn from the beat's class. P21 adds it to "
            "P12's cVAE and WGAN-GP, applied to whole beats (each generated patient-relative beat plus a real training patient's template), with "
            "its weight chosen on validation among 0.1, 1 and 10, and makes a third condition from physiology: P12's cVAE banks with every "
            "synthetic F beat replaced by a fraction of a real training patient's V deviation, since a fusion beat is a ventricular and a "
            "conducted activation meeting. Each condition goes through P18's ten-run pipeline, with the stage and the veto chosen on P14's "
            "patients by P20's rule.")  # fmt: skip
    if physics is None:
        return text + " " + _missing("P21: --part check on Kaggle") + "\n"
    c = physics["classes"]
    fu = physics["fusion"]
    text += (" Fitted to real training beats, the simulator explained a median R$^2$ of "
             + ", ".join(f"{_n(c[k]['median_r2'])} ({k}, {c[k]['beats']:,} beats)" for k in c)
             + f"; the fusion model explained a median R$^2$ of {_n(fu['median_r2'])} of {fu['f_beats']:,} real training F beats in "
             f"{len(fu['recordings'])} recordings, with alpha from {_n(fu['alpha_low'], 2)} to {_n(fu['alpha_high'], 2)} "
             "(Figure~\\ref{fig:p21-fit}). Sixteen numbers fit almost any smooth beat, so a good fit does not show that the physics is right.")  # fmt: skip
    if not physics["physics_passed"]:
        return text + " The simulator fell below the pre-registered stop rule (median R$^2$ 0.5), so the generator conditions were not run.\n"
    if not fu["passed"]:
        text += (" The fusion model fell below its pre-registered stop rule (median R$^2$ 0.5): real F beats here are not a fraction of their "
                 "recording's mean V deviation, so the fusion condition was not run, and each family below keeps Holm's adjustment for three.")  # fmt: skip
    if len(fams) < 2 or (fu["passed"] and fusion is None):
        return text + " " + _missing("P21: --part train (both families)" + (" and --part fusion" if fu["passed"] else "") + " on Kaggle") + "\n"
    parts = []
    for f, d in fams.items():
        t = d["lambda_choice"]["table"]
        lam = f"{d['chosen_lambda']:g}"
        p12, chosen = d["describe"]["p12_seed0"], d["describe"][f"lambda{lam}_seed0"]
        parts.append(f"{SHORT[f]}: weight {lam} (validation S F1 {_n(t[lam]['mean'])}, weight 0 "
                     f"{_n(t['0']['mean'])}); S beats over {_n(chosen['S']['nearest_patients'], 1)} effective training patients "
                     f"(P12's bank {_n(p12['S']['nearest_patients'], 1)}), F beats over {_n(chosen['F']['nearest_patients'], 1)} "
                     f"({_n(p12['F']['nearest_patients'], 1)})")  # fmt: skip
    text += " Chosen on validation: " + "; ".join(parts)
    if fusion is not None:
        nf = fusion["nearest_patients_F"]
        text += f". The fusion F beats spread over {_n(nf['fusion_seed0'], 1)} effective training patients (P12's cVAE F beats {_n(nf['p12_seed0'], 1)})"
    text += " (Figure~\\ref{fig:p21-banks})."
    euler = []
    for f, d in fams.items():
        p12, chosen = d["describe"]["p12_seed0"], d["describe"][f"lambda{d['chosen_lambda']:g}_seed0"]
        euler.append(f"{SHORT[f]} " + ", ".join(f"{k} {_n(chosen[k]['euler'], 2)} ({_n(p12[k]['euler'], 2)})" for k in ("S", "V", "F")))
    text += (" The Euler loss over its scale (real whole training beats average 1) of the chosen seed-0 banks, P12's banks in brackets: "
             + "; ".join(euler) + ". The loss barely moved the generators: they make patient-relative beats and the loss scores them after a "
             "real training template is added, so a beat near zero already scores about as a real one does. More of the loss has to come "
             "from the generated part itself, for example by drawing the simulator's parameters with the generator's noise, before physics "
             "can shape these banks.")  # fmt: skip
    if s is None:
        return text + " " + _missing("P21: --part score on Kaggle") + "\n"
    sc = s["conditions"]["chosen"]
    present = {k: b for k, b in PHYSICS_BASE.items() if f"cnn/{k}" in sc}
    rows = "; ".join(f"{PHYSICS_SHORT[k]} {_n(sc[f'cnn/{k}']['f1_S']['point'])} ({SHORT[b]} {_n(sc[f'cnn/{b}']['f1_S']['point'])})" for k, b in present.items())
    text += (f" On P19's {s['patients']:,} patients, S F1 with the chosen settings: {rows} (Figure~\\ref{{fig:p21}}). Secondary: "
             f"{_physics_family(s['families']['secondary'])}.")  # fmt: skip
    if test is None:
        return text + " " + _missing("P21: --part test on Kaggle") + "\n"
    tc = test["conditions"]["chosen"]
    rows = "; ".join(f"{PHYSICS_SHORT[k]} {_n(tc[f'cnn/{k}']['rare_macro_f1']['point'])} (S F1 {_n(tc[f'cnn/{k}']['f1_S']['point'])}, "
                     f"F F1 {_n(tc[f'cnn/{k}']['f1_F']['point'])}; {SHORT[b]} {_n(tc[f'cnn/{b}']['rare_macro_f1']['point'])})" for k, b in present.items())  # fmt: skip
    text += (f" On the test records (one read), rare-class macro-F1: {rows}. Confirmatory: {_physics_family(test['families']['confirmatory'])}.")
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


PAIR_SHORT = {"cnn/none": "the cnn without augmentation", "cnn/cvae": "the cnn with the cVAE", "cnn/wgan_gp": "the cnn with the WGAN-GP",
              "resnet/none": "the ResNet without augmentation", "resnet/cvae": "the ResNet with the cVAE", "resnet/wgan_gp": "the ResNet with the WGAN-GP",
              "cnn/cvae_fitted": "the cVAE with the fitted loss", "cnn/wgan_gp_fitted": "the WGAN-GP with the fitted loss"}


def _pairs_family(items: dict) -> str:
    out = []
    for name, c in items.items():
        a, b = name.replace(", chosen settings", "").split(" vs ")
        out.append(f"{PAIR_SHORT[a]} against {PAIR_SHORT[b]} {_ci(c, signed=True)}, {_p(c)} ({_changed(c)})")
    return "; ".join(out)


def _pairs_results(s: dict | None, test: dict | None, pairs: dict, label: str, missing: str) -> str:
    """P19's patients and the test read for a P22- or P23-style comparison of new ensembles with P18's."""
    if s is None:
        return " " + _missing(f"{missing}: the scoring session on Kaggle")
    sc = s["conditions"]["chosen"]
    rows = "; ".join(f"{PAIR_SHORT[c]} {_n(sc[c]['f1_S']['point'])} ({PAIR_SHORT[b]} {_n(sc[b]['f1_S']['point'])})" for c, b in pairs.items())
    text = (f" On P19's {s['patients']:,} patients, S F1 with the chosen settings: {rows} (Figure~\\ref{{{label}}}). Secondary: "
            f"{_pairs_family(s['families']['secondary'])}.")  # fmt: skip
    if test is None:
        return text + " " + _missing(f"{missing}: --part test on Kaggle")
    tc = test["conditions"]["chosen"]
    rows = "; ".join(f"{PAIR_SHORT[c]} {_n(tc[c]['rare_macro_f1']['point'])} (S F1 {_n(tc[c]['f1_S']['point'])}, F F1 {_n(tc[c]['f1_F']['point'])}; "
                     f"{PAIR_SHORT[b]} {_n(tc[b]['rare_macro_f1']['point'])})" for c, b in pairs.items())  # fmt: skip
    return text + f" On the test records (one read), rare-class macro-F1: {rows}. Confirmatory: {_pairs_family(test['families']['confirmatory'])}."


def posthoc_bigger_classifier() -> str:
    """P22: phase 2's ResNet1D in P18's pipeline."""
    from ecgaug.experiments.phase6_bigger_classifier import PAIRS, cnn_validation
    from ecgaug.names import PHASE6_BIGGER_CLASSIFIER

    folder = paths.results_for(PHASE6_BIGGER_CLASSIFIER)
    m, s, test = (_load(folder / n) for n in ("members.json", "summary.json", "test_summary.json"))
    text = ("\\paragraph{A larger classifier (P22).} The rare classes might need more capacity than the P7 cnn has. Phase 2's ResNet1D (a stem "
            "convolution and four residual stages, about 6.7 times the cnn's parameters), which lost to the cnn at phase 2 before the pipeline "
            "had patient-relative beats, RR features and the banks, was trained in P18's pipeline with phase 2's learning rate for it and the "
            "budget every phase 2 classifier shared, for no augmentation, the cVAE and the WGAN-GP, ten seeds each; Huang's stage and the veto "
            "were chosen for each ResNet ensemble on P14's patients by P20's rule.")  # fmt: skip
    if m is None:
        return text + " " + _missing("P22: --part run on Kaggle") + "\n"
    cnn = cnn_validation() or {}
    rows = []
    for c in m["validation"]:
        r = m["validation"][c]["f1_S"]
        base = cnn.get(c.replace("resnet/", "cnn/"), {}).get("f1_S")
        rows.append(f"{PAIR_SHORT[c]} {_n(float(np.mean(r)))} (SD {_n(float(np.std(r, ddof=1)))})"
                    + (f" against {_n(float(np.mean(base)))} (SD {_n(float(np.std(base, ddof=1)))}) for the cnn" if base else ""))  # fmt: skip
    text += " Validation S F1 over the ten runs: " + "; ".join(rows) + " (Figure~\\ref{fig:p22-val})."
    text += _pairs_results(s, test, PAIRS, "fig:p22", "P22")
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_physics_fitted() -> str:
    """P23: P21 with the simulator fitted to each generated beat."""
    from ecgaug.experiments.phase6_physics_fitted import PAIRS
    from ecgaug.names import PHASE6_PHYSICS_FITTED

    folder = paths.results_for(PHASE6_PHYSICS_FITTED)
    check, s, test = (_load(folder / n) for n in ("physics.json", "summary.json", "test_summary.json"))
    fams = {f: d for f in ("cvae", "wgan_gp") if (d := _load(folder / f"{f}.json")) is not None}
    text = ("\\paragraph{A fitted physics loss (P23).} P21's loss drew the simulator's parameters at random, so its value was set by the draw. "
            "P23 keeps P21's design and changes only the loss: each generated beat, with a real training template added, is compared with the "
            "simulator's beat that fits it best for the class's mean wave timing (heights and baseline solved exactly per beat), and the loss is "
            "what the simulator cannot explain, scaled so real training beats score 1. In SimGANs' derivative form this loss mostly measured "
            "roughness on 60 training beats per class checked before the pre-registration, so the signal form was registered.")  # fmt: skip
    if check is None:
        return text + " " + _missing("P23: --part check on Kaggle") + "\n"
    c = check["classes"]
    text += (" With the class's wave timing the simulator explained a median R$^2$ of "
             + ", ".join(f"{_n(c[k]['median_r2'], 2)} ({k})" for k in c)
             + " of the real training beats (Figure~\\ref{fig:p23-check}).")  # fmt: skip
    if not check["physics_passed"]:
        return text + " This fell below the pre-registered stop rule (median R$^2$ 0.5), so neither condition was run.\n"
    if len(fams) < 2:
        return text + " " + _missing("P23: --part train (both families) on Kaggle") + "\n"
    parts = []
    for f, d in fams.items():
        t, lam = d["lambda_choice"]["table"], f"{d['chosen_lambda']:g}"
        before, after = d["describe"]["p12_seed0"], d["describe"][f"lambda{lam}_seed0"]
        parts.append(f"{SHORT[f]}: weight {lam} (validation S F1 {_n(t[lam]['mean'])}, weight 0 {_n(t['0']['mean'])}); fitted loss of the "
                     "seed-0 bank " + ", ".join(f"{k} {_n(after[k]['physics'], 2)} ({_n(before[k]['physics'], 2)})" for k in ("S", "V", "F"))
                     + ", roughness " + ", ".join(f"{k} {_n(after[k]['roughness'], 2)} ({_n(before[k]['roughness'], 2)})" for k in ("S", "V", "F")))  # fmt: skip
    text += " Chosen on validation, with P12's banks in brackets: " + "; ".join(parts) + " (Figure~\\ref{fig:p23-banks})."
    text += _pairs_results(s, test, PAIRS, "fig:p23", "P23")
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def _count_word(n: int) -> str:
    return "none" if int(n) == 0 else f"{int(n):,}"


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
        units = (f"{s['records']} records from {s['patients']} patients, intervals resampling patients" if s.get("clustered_by") == "patient" and s.get("patients", s["records"]) != s["records"]
                 else f"{s['records']} records, each taken as one patient")  # fmt: skip
        text = (f"On {DATABASE[db]} ({s['role']}; {units}; {beats['S']:,} S and {beats['F']:,} F beats; primary metric {METRIC[metric]}), "
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
        from ecgaug.names import PHASE6_EXTERNAL_VALIDATION

        higher = []
        for db in DATABASE:
            e2, e1 = _load(folder / f"{db}_summary.json"), _load(paths.results_for(PHASE6_EXTERNAL_VALIDATION) / f"{db}_summary.json")
            if e2 and e1:
                m = e2["primary_metric"]
                higher.append(f"{sum(e2['conditions'][k][m]['point'] > e1['conditions'][k][m]['point'] for k in e2['conditions'])} of the {len(e2['conditions'])} conditions on {DATABASE[db]}")
        text += " Externally, " + "; ".join(ext) + (f"; the v2 point estimate was above v1's in {' and in '.join(higher)}" if higher else "") + " (Figure~\\ref{fig:v2-external})."
    return text + " These results are post hoc and do not replace the primary results of Table~\\ref{tab:test}.\n"


def posthoc_patient_template() -> str:
    """P6: patient-relative beats, from results/phase6_patient_template/ (adoption, diagnostic, test, external)."""
    from ecgaug.names import PHASE6_PATIENT_TEMPLATE

    folder = paths.results_for(PHASE6_PATIENT_TEMPLATE)
    adoption, diag, summary = (_load(folder / n) for n in ("adoption.json", "diagnostic.json", "summary.json"))
    text = ("\\paragraph{Patient-relative beats (P6).} A second follow-up asked whether the networks fit the shapes of the training patients' beats, "
            "so that a beat whose shape is normal for its own patient but unlike the training patients' beats is called abnormal. Version 6 of the pre-registration gives every beat minus the "
            "patient's template, the median of the z-scored beats of the same record within 150~s (no labels), with the v2 RR ratio (dataset v3, "
            "Figure~\\ref{fig:v3-input}); synthetic beats take the template of the real beat whose RR values they carry.")  # fmt: skip
    if diag is not None:
        parts = []
        for arch, versions in diag["networks"].items():
            g = {v: versions[v] for v in versions}
            parts.append(f"the {arch} without augmentation reached S PR-AUC " + ", ".join(
                f"{_n(x['train']['pr_auc']['S'])} on the training and {_n(x['val']['pr_auc']['S'])} on the validation records with {v}" for v, x in g.items()))
        lin = diag["linear"]
        text += (f" On DS1 (Figure~\\ref{{fig:v3-diagnostic}}), {'; '.join(parts)}. A linear discriminant with uniform priors reached validation S F1 "
                 f"{_n(lin['rr']['f1']['S'])} on the RR features, {_n(lin['rr_raw']['f1']['S'])} with samples of the raw beat added and "
                 f"{_n(lin['rr_relative']['f1']['S'])} with samples of the relative beat.")  # fmt: skip
    if adoption is None:
        return text + " " + _missing("P6: --part grid on Kaggle") + "\n"
    per = "; ".join(f"{arch} {_n(v['v3_mean'])} against {_n(v['v2_mean'])}" for arch, v in adoption["per_architecture"].items())
    text += (f" By the pre-registered adoption rule, the cells without augmentation were run first: validation S F1 (seed means) {per}, a mean "
             f"difference of {_s(adoption['mean_difference'])} against a margin of {_n(adoption['margin'])}, so v3 was "
             f"{'adopted and the full grid was run' if adoption['adopted'] else 'not adopted; the other 90 cells were not run and the test records were not read'}.")  # fmt: skip
    if not adoption["adopted"]:
        return text + " Post hoc.\n"
    if summary is None:
        return text + " " + _missing("P6 results: --part select, then --part test") + "\n"
    nones = "; ".join(f"the {arch} without augmentation reached {_ci(summary['conditions'][f'{arch}/none']['rare_macro_f1'])} "
                      f"(v3 minus v1 {_ci(summary['versus_v1'][f'{arch}/none'], signed=True)})" for arch in ("cnn", "rnn"))  # fmt: skip
    diffs = summary["versus_v1"]
    up = [k for k, c in diffs.items() if c["ci_low"] > 0]
    down = [k for k, c in diffs.items() if c["ci_high"] < 0]
    s_se = "; ".join(f"{arch} {_n(summary['conditions'][f'{arch}/none']['se_S']['point'])}" for arch in ("cnn", "rnn"))
    conf = summary["families"]["confirmatory"]
    improved = [k for k, c in conf.items() if c["reject_holm"] and c["point"] > 0]
    secondary = {k: c for k, c in summary["families"]["secondary"].items() if c["reject_holm"]}
    text += (f" On the test records (Figure~\\ref{{fig:v3-results}}), {nones}. Against v1, the paired difference had a 95\\% CI above zero in "
             f"{len(up)} of the {len(diffs)} conditions and below zero in {len(down)}. S sensitivity without augmentation: {s_se}. "
             f"Against no augmentation: {_family_sentence(conf)}. By the registered criterion, generative augmentation "
             f"{'improved' if improved else 'did not detectably improve'} the primary metric with v3 inputs. "
             f"{'In the secondary family Holm rejected ' + _family_sentence(secondary) + '.' if secondary else 'No secondary comparison was rejected.'}")  # fmt: skip
    ext = []
    for db in DATABASE:
        e = _load(folder / f"{db}_summary.json")
        if e:
            metric = e["primary_metric"]
            verbs = [_changed(c) for c in e["confirmatory"].values()]
            ext.append(f"on {DATABASE[db]} the cnn without augmentation reached {_n(e['conditions']['cnn/none'][metric]['point'])} ({METRIC[metric]}) and "
                       + ", ".join(f"generative augmentation {v} it in {verbs.count(v)}" for v in dict.fromkeys(verbs)) + " of the 4 confirmatory comparisons")
    if ext:
        text += " Externally, " + "; ".join(ext) + " (Figure~\\ref{fig:v3-external})."
    return text + " These results are post hoc and do not replace the primary results of Table~\\ref{tab:test}.\n"


def posthoc_relative_cnn() -> str:
    """P7: the cnn alone with patient-relative beats, from results/phase6_relative_cnn/."""
    from ecgaug.names import PHASE6_RELATIVE_CNN

    folder = paths.results_for(PHASE6_RELATIVE_CNN)
    selection, summary = _load(folder / "selection.json"), _load(folder / "summary.json")
    text = ("\\paragraph{The cnn with patient-relative beats (P7).} Version 7 of the pre-registration takes the one baseline that v6 made "
            "clearly stronger on validation, the cnn with v3 inputs, and repeats the whole comparison for it (48 runs, the phase 4 gate rule, "
            "one read of the test records). The choice of the cnn was made on validation after v6's result and is reported as such.")  # fmt: skip
    if selection is None:
        return text + " " + _missing("P7: --part grid on Kaggle, then --part select") + "\n"
    strengths = ", ".join(f"{SHORT[c]} {v['strength']}" for c, v in selection["chosen"]["cnn"].items() if v["strength"] is not None)
    none_val = selection["chosen"]["cnn"]["none"]["mean"]
    text += f" On validation the cnn without augmentation reached S F1 {_n(none_val)} (seed mean); the rule chose {strengths}."
    if summary is None:
        return text + " " + _missing("P7 results: --part test") + "\n"
    c = summary["conditions"]["cnn/none"]
    conf = summary["families"]["confirmatory"]
    improved = [k for k, x in conf.items() if x["reject_holm"] and x["point"] > 0]
    secondary = {k: x for k, x in summary["families"]["secondary"].items() if x["reject_holm"]}
    diffs = summary["versus_v1"]
    up = [k for k, x in diffs.items() if x["ci_low"] > 0]
    down = [k for k, x in diffs.items() if x["ci_high"] < 0]
    final = _load(paths.results_for(FINAL_EVAL) / "summary.json")
    old_c = final["conditions"]["cnn/none"] if final else None
    text += (f" On the test records (Figure~\\ref{{fig:p7-results}}) the cnn without augmentation reached {_ci(c['rare_macro_f1'])}, "
             f"{_s(summary['versus_v1']['cnn/none']['point'])} against v1 (95\\% CI {_s(summary['versus_v1']['cnn/none']['ci_low'])} to "
             f"{_s(summary['versus_v1']['cnn/none']['ci_high'])}); S F1 {_n(c['f1_S']['point'])}"
             + (f" (v1 {_n(old_c['f1_S']['point'])})" if old_c else "") + f", S Se {_n(c['se_S']['point'])}, S +P {_n(c['ppv_S']['point'])}, F F1 "
             f"{_n(c['f1_F']['point'])}" + (f" (v1 {_n(old_c['f1_F']['point'])})" if old_c else "") + ". Against v1 the paired difference had a 95\\% CI above zero in {len(up)} of the {len(diffs)} "
             f"conditions and below zero in {len(down)}. Against no augmentation: {_family_sentence(conf)}. By the registered criterion, "
             f"generative augmentation {'improved' if improved else 'did not detectably improve'} the primary metric for this cnn. "
             f"{'In the secondary family Holm rejected ' + _family_sentence(secondary) + '.' if secondary else 'No secondary comparison was rejected.'}")  # fmt: skip
    from ecgaug.names import PHASE6_EXTERNAL_VALIDATION

    ext, lower = [], []
    for db in DATABASE:
        e = _load(folder / f"{db}_summary.json")
        if e:
            metric = e["primary_metric"]
            verbs = [_changed(x) for x in e["confirmatory"].values()]
            e1 = _load(paths.results_for(PHASE6_EXTERNAL_VALIDATION) / f"{db}_summary.json")
            point = e["conditions"]["cnn/none"][metric]["point"]
            versus = ""
            if e1:
                versus = f" (v1 {_n(e1['conditions']['cnn/none'][metric]['point'])})"
                lower.append(point < e1["conditions"]["cnn/none"][metric]["point"])
            ext.append(f"on {DATABASE[db]} the cnn without augmentation reached {_n(point)}{versus} ({METRIC[metric]}) and "
                       + ", ".join(f"generative augmentation {v} it in {verbs.count(v)}" for v in dict.fromkeys(verbs)) + f" of the {len(verbs)} confirmatory comparisons")
    if ext:
        text += " Externally, " + "; ".join(ext) + " (Figure~\\ref{fig:p7-external})"
        text += (": the gain on MIT-BIH did not carry over to every database." if any(lower) else ".")
    return text + " These results are post hoc and do not replace the primary results of Table~\\ref{tab:test}.\n"


def posthoc_mechanism() -> str:
    """P8: patient attribution of synthetic beats and the training-versus-validation gap, from results/phase6_augmentation_mechanism/."""
    from ecgaug.names import PHASE6_AUGMENTATION_MECHANISM

    folder = paths.results_for(PHASE6_AUGMENTATION_MECHANISM)
    attribution, gap = _load(folder / "attribution.json"), _load(folder / "gap.json")
    text = ("\\paragraph{Why augmentation did not carry over to new patients here (P8).} Two descriptive analyses on the training and validation "
            "records only (pre-registration version 7).")
    if attribution is None and gap is None:
        return text + " " + _missing("P8: --part all on Kaggle") + "\n"
    if attribution is not None:
        s = attribution["classes"]["S"]
        real = s["real_training"]
        own = s.get("real_nearest_other", {}).get("same_record")
        parts = []
        for condition in ("cvae", "wgan_gp"):
            seeds = s.get(f"{condition}_bank", {})
            if seeds:
                tops = [e["shares"].get(real["top_record"], 0.0) for e in seeds.values()]
                effs = [e["effective_records"] for e in seeds.values()]
                parts.append(f"{SHORT[condition]} {_n(float(np.mean(tops)), 2)} (effective records {_n(float(np.mean(effs)), 2)})")
        val = s.get("validation_nearest")
        text += (f" Each synthetic S beat was attributed to the training patient of its nearest real S beat. Record {real['top_record']} holds "
                 f"{_n(real['top_share'], 2)} of the real training S beats (effective number of records {_n(real['effective_records'], 2)}); "
                 f"the share of synthetic S beats attributed to it was {', '.join(parts)} (means over the three bank seeds)"
                 + (f", and a real training S beat's nearest other S beat came from the same patient in {_n(own, 2)} of cases" if own is not None else "")
                 + " (Figure~\\ref{fig:p8}).")  # fmt: skip
        if val:
            top_val = list(val["shares"].items())[:2]
            syn_share = {r: float(np.mean([e["shares"].get(r, 0.0) for c in ("cvae", "wgan_gp") for e in s.get(f"{c}_bank", {}).values()] or [0.0]))
                         for r, _ in top_val}  # fmt: skip
            text += (" The validation patients' S beats were nearest to " + " and ".join(
                f"record {r} ({_n(v, 2)} of them; {_n(real['shares'].get(r, 0.0), 2)} of the real and {_n(syn_share[r], 2)} of the synthetic training S beats)"
                for r, v in top_val) + ": the new patients resembled training patients that are rare in the real data and equally rare in the synthetic data.")  # fmt: skip
    if gap is not None:
        rows = []
        for arch, conds in gap["networks"].items():
            none = conds["none"]["gap"]["S"]
            gen = ", ".join(f"{SHORT[c]} {_s(conds[c]['gap']['S'])}" for c in ("cvae", "wgan_gp") if c in conds)
            rows.append(f"{arch}: none {_s(none)}, {gen}")
        text += (" The gap between training and validation S PR-AUC of the frozen networks (seed means; positive means better on the training "
                 f"patients) was {'; '.join(rows)}.")
    return text + " Descriptive and post hoc.\n"


def _p9() -> tuple[dict | None, dict | None]:
    from ecgaug.names import PHASE6_PATIENT_TRANSFER

    folder = paths.results_for(PHASE6_PATIENT_TRANSFER)
    return _load(folder / "selection.json"), _load(folder / "summary.json")


def posthoc_patient_transfer() -> str:
    """P9: patient-transfer augmentation, from results/phase6_patient_transfer/."""
    from ecgaug.names import PHASE6_PATIENT_TRANSFER

    folder = paths.results_for(PHASE6_PATIENT_TRANSFER)
    selection, summary = _p9()
    names = {"transfer_real": "transfer of real beats", "transfer_cvae": "transfer of cVAE beats", "transfer_wgan_gp": "transfer of WGAN-GP beats"}
    text = ("\\paragraph{Augmentation that adds patients (P9).} If synthetic beats fail because they come from the same few patients, augmentation "
            "should help when it adds patients instead. Version 8 of the pre-registration writes each rare-class training beat as its patient's "
            "template plus a deviation and places the deviation on the template of another training patient, with RR intervals scaled to that "
            "patient's rhythm" + (" (Figure~\\ref{fig:p9-example})" if (paths.figures_for("phase6_patient_transfer") / "transfer_example.png").is_file() else "")
            + "; the deviations come from real beats or from the frozen generators. The grid runs on "
            "dataset v2, where the networks see raw beats, unscreened, for the cnn and the rnn (60 runs).")  # fmt: skip
    if selection is None:
        return text + " " + _missing("P9: --part grid on Kaggle, then --part select") + "\n"
    val = "; ".join(f"{arch}: none {_n(c['none']['mean'])}, " + ", ".join(f"{names[k]} {_n(c[k]['mean'])}" for k in names) for arch, c in selection["chosen"].items())
    text += f" Validation S F1 (seed means, chosen strengths): {val}."
    if summary is None:
        return text + " " + _missing("P9 results: --part test") + "\n"
    cond = summary["conditions"]
    per = "; ".join(f"{arch}: none {_n(cond[f'{arch}/none']['rare_macro_f1']['point'])}, " + ", ".join(f"{names[k]} {_n(cond[f'{arch}/{k}']['rare_macro_f1']['point'])}" for k in names)
                    for arch in selection["chosen"])
    conf, sec = summary["families"]["confirmatory"], summary["families"]["secondary"]
    improved = [k for k, c in conf.items() if c["reject_holm"] and c["point"] > 0]
    secondary = {k: c for k, c in sec.items() if c["reject_holm"]}
    text += (f" On the test records (Figure~\\ref{{fig:p9-results}}) rare-class macro-F1 was {per}. Against no augmentation: {_family_sentence(conf)}. "
             f"By the registered criterion, generative augmentation with patient transfer {'improved' if improved else 'did not detectably improve'} "
             f"the primary metric. Secondary family: {_family_sentence(sec)}"
             + (f"; Holm rejected {len(secondary)}." if secondary else "; none rejected."))  # fmt: skip
    vs = summary.get("versus_v2", {})
    pairs = [(arch, g) for arch in selection["chosen"] for g in ("cvae", "wgan_gp") if f"{arch}/transfer_{g} vs v2 {g}" in vs]
    if pairs:
        higher = sum(vs[f"{a}/transfer_{g} vs v2 {g}"]["point"] > 0 for a, g in pairs)
        text += (f" Against plain generative augmentation of the same generator and architecture (v5), patient transfer was higher in {higher} of "
                 f"{len(pairs)} pairs (" + ", ".join(f"{a} {SHORT[g]} {_ci(vs[f'{a}/transfer_{g} vs v2 {g}'], signed=True)}" for a, g in pairs) + ").")
    ext = []
    for db in DATABASE:
        e = _load(folder / f"{db}_summary.json")
        if e:
            verbs = [_changed(c) for c in e["confirmatory"].values()]
            ext.append(f"on {DATABASE[db]} " + ", ".join(f"{v} it in {verbs.count(v)}" for v in dict.fromkeys(verbs)) + f" of {len(verbs)}")
    if ext:
        text += " Against no augmentation on the external databases, generative transfer " + "; ".join(ext) + "."
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_patient_adaptation() -> str:
    """P10: adaptive batch normalization per patient on the frozen checkpoints, from results/phase6_patient_adaptation/."""
    from ecgaug.names import PHASE6_PATIENT_ADAPTATION

    folder = paths.results_for(PHASE6_PATIENT_ADAPTATION)
    adoption, summary = _load(folder / "adoption.json"), _load(folder / "summary.json")
    label = {"v1/cnn": "the final evaluation's cnn", "v1/rnn": "the final evaluation's rnn", "p7/cnn": "the P7 cnn"}
    text = ("\\paragraph{Per-patient normalization inside the network (P10).} The same idea as the patient-relative input can be applied "
            "inside a trained network: with adaptive batch normalization each patient is classified by a copy of the network whose "
            "batch-normalization layers use the statistics of that patient's beats (no labels) instead of the training statistics. "
            "Nothing is retrained (pre-registration version 9).")  # fmt: skip
    if adoption is None:
        return text + " " + _missing("P10: --part validation on Kaggle") + "\n"
    parts = [f"{label.get(k, k)} {_n(e['plain_mean'])} to {_n(e['adapted_mean'])} ({'adopted' if e['adopted'] else 'not adopted'}, margin {_n(e['margin'])})"
             for k, e in adoption["pairs"].items()]
    text += " Validation S F1 without and with adaptation (seed means): " + "; ".join(parts) + "."
    if not any(e["adopted"] for e in adoption["pairs"].values()):
        return text + " No model met the adoption rule, so the test records were not read. Post hoc.\n"
    if summary is None:
        return text + " " + _missing("P10 results: --part test") + "\n"
    fam = summary["families"]
    final = _load(paths.results_for(FINAL_EVAL) / "summary.json")
    if final is not None and "v1/cnn/none" in summary["conditions"]:
        a, b = summary["conditions"]["v1/cnn/none"], final["conditions"]["cnn/none"]
        text += (f" On the test records the final evaluation's cnn without augmentation reached rare-class macro-F1 {_n(a['rare_macro_f1']['point'])} "
                 f"with adaptation ({_n(b['rare_macro_f1']['point'])} without): S F1 {_n(a['f1_S']['point'])} ({_n(b['f1_S']['point'])}) and F F1 "
                 f"{_n(a['f1_F']['point'])} ({_n(b['f1_F']['point'])}). The validation records, on which adoption was decided by S F1, hold too few F "
                 "beats to show the second.")
    text += (" Adaptation against none without it (Figure~\\ref{fig:p10}): " + _family_sentence(fam["adaptation"]) + ". "
             "Under adaptation, generative augmentation against none: " + _family_sentence(fam["augmentation_under_adaptation"]) + ".")
    ext = []
    for db in DATABASE:
        e = _load(folder / f"{db}_summary.json")
        if e:
            verbs = [_changed(c) for c in e["families"]["adaptation"].values()]
            ext.append(f"on {DATABASE[db]} adaptation " + ", ".join(f"{v} the score in {verbs.count(v)}" for v in dict.fromkeys(verbs)) + f" of {len(verbs)} models")
    if ext:
        text += " Externally, " + "; ".join(ext) + "."
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_more_patients() -> str:
    """P11: the P7 cnn trained on DS1 plus the INCART and SVDB patients, from results/phase6_more_patients/."""
    from ecgaug.names import PHASE6_MORE_PATIENTS

    folder = paths.results_for(PHASE6_MORE_PATIENTS)
    data, selection, summary = (_load(folder / n) for n in ("data.json", "selection.json", "summary.json"))
    text = ("\\paragraph{More patients (P11).} If the few training patients are the limit, more patients should help. Version 10 of the "
            "pre-registration trains the P7 cnn (patient-relative beats, settings and strengths unchanged) on DS1 plus every INCART and SVDB "
            "record, keeping all their S, V and F beats and at most 600 N beats per record; every fifth external patient goes to validation, "
            "and DS2 stays the test set, so no external database is left for testing.")  # fmt: skip
    if data is None:
        return text + " " + _missing("P11: --part grid on Kaggle") + "\n"
    ext = data["external"]
    added = {c: sum(e["beats"]["train"][c] for e in ext.values()) for c in ("S", "V", "F")}
    units = sum(e["units"] - len(e["validation_units"]) for e in ext.values())
    text += (f" The external training part adds {units} patients and {added['S']:,} S, {added['V']:,} V and {added['F']:,} F beats to DS1's "
             f"{data['ds1']['train']['S']:,} S, {data['ds1']['train']['V']:,} V and {data['ds1']['train']['F']:,} F.")
    split = "; ".join(f"{ {'incartdb': 'INCART', 'svdb': 'SVDB'}[db]} {len(e['validation_units'])} of {e['units']} patients and "
                      f"{e['beats']['val']['S']:,} of {e['beats']['val']['S'] + e['beats']['train']['S']:,} S beats" for db, e in ext.items())  # fmt: skip
    text += f" The every-fifth rule sent to validation: {split}, so where a database's S beats sit in few patients, the split decides how many are trained on."
    if selection is not None:
        text += " Validation S F1 (seed means, the enlarged validation set): " + ", ".join(
            f"{'no augmentation' if c == 'none' else SHORT[c]} {_n(v['mean'])}" for c, v in selection["chosen"]["cnn"].items()) + "."
    if summary is None:
        return text + " " + _missing("P11 results: --part select, then --part test") + "\n"
    conf, sec = summary["families"]["confirmatory"], summary["families"]["secondary"]
    c = summary["conditions"]["cnn/none"]
    text += (f" On the test records (Figure~\\ref{{fig:p11}}) the cnn without augmentation reached {_ci(c['rare_macro_f1'])} (S F1 {_n(c['f1_S']['point'])}, "
             f"F F1 {_n(c['f1_F']['point'])}); against the same cnn trained on DS1 only: {_family_sentence(conf)}. With more patients, each "
             f"condition against none: {_family_sentence(sec)}.")
    total = lambda part: sum(part.values())  # noqa: E731
    f_all = data["ds1"]["train"]["F"] + added["F"]
    n_all = total(data["ds1"]["train"]) + sum(total(e["beats"]["train"]) for e in ext.values())
    scores = ", ".join(f"{'no augmentation' if k == 'cnn/none' else SHORT[k.split('/')[1]]} {_n(v['rare_macro_f1']['point'])} "
                       f"({_n(v['f1_S']['point'])}, {_n(v['f1_F']['point'])})" for k, v in summary["conditions"].items())  # fmt: skip
    text += (f" Test scores (S F1, F F1): {scores}. F beats are {f_all / n_all:.1%} of the enlarged training set against "
             f"{data['ds1']['train']['F'] / total(data['ds1']['train']):.1%} of DS1's, so the method that restores the rare class decides the F score.")
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_relative_generators() -> str:
    """P12: the cVAE and WGAN-GP trained on P11's training beats in v3 form, from results/phase6_relative_generators/."""
    from ecgaug.names import PHASE6_MORE_PATIENTS, PHASE6_RELATIVE_GENERATORS

    folder = paths.results_for(PHASE6_RELATIVE_GENERATORS)
    gens = {f: _load(folder / f"{f}_generators.json") for f in GENERATIVE}
    mechs = {f: _load(folder / f"{f}_mechanism.json") for f in GENERATIVE}
    selection, summary = _load(folder / "selection.json"), _load(folder / "summary.json")
    text = ("\\paragraph{Generators trained on what the cnn sees (P12).} The cVAE and WGAN-GP conditions of P7 and P11 used the phase 3 generators, "
            "which learned ordinary beats of DS1's 17 training records; their beats were made patient-relative afterwards. Version 11 of the "
            "pre-registration trains the same generators (settings, step budget and seeds unchanged) on P11's training beats in the form the cnn "
            "sees them, divided by one scale, and repeats the comparison with each bank at 0.1, 0.25 and 0.5 (unscreened), one strength per "
            "generator chosen on P11's validation.")  # fmt: skip
    if not all(gens.values()):
        return text + " " + _missing("P12: --part train for both generator families on Kaggle") + "\n"
    diverged = [f"{SHORT[f]} seed {s}" for f, g in gens.items() for s, e in g["seeds"].items() if e["diverged"]]
    text += " " + ("No generator diverged." if not diverged else f"Diverged: {', '.join(diverged)}.")
    if all(mechs.values()):
        parts = [f"{SHORT[f]} {cls} {_n(m['classes'][cls]['p12']['effective_records'], 1)} (P11's bank {_n(m['classes'][cls]['p11']['effective_records'], 1)})"
                 for f, m in mechs.items() for cls in m["classes"]]  # fmt: skip
        real = next(iter(mechs.values()))["classes"]
        text += (" The synthetic beats' nearest real training beats came from an effective number of patients (exp of the entropy of their "
                 f"shares, P8's measure; Figure~\\ref{{fig:p12-mechanism}}) of {'; '.join(parts)}, against "
                 + " and ".join(f"{_n(e['real']['effective_records'], 1)} for the real {cls} beats" for cls, e in real.items()) + ".")  # fmt: skip
    if selection is not None:
        before = _load(paths.results_for(PHASE6_MORE_PATIENTS) / "selection.json")
        text += " Validation S F1 at the chosen strengths: " + ", ".join(
            f"{SHORT[f]} {_n(v['mean'])} at {v['strength']}" for f, v in selection["chosen"]["cnn"].items())
        if before is not None:
            text += " (P11's phase 3 generators on the same validation: " + ", ".join(
                f"{SHORT[f]} {_n(before['chosen']['cnn'][f]['mean'])}" for f in selection["chosen"]["cnn"]) + ")"
        text += "."
    if summary is None:
        return text + " " + _missing("P12 results: --part select, then --part test") + "\n"
    conf, sec = summary["families"]["confirmatory"], summary["families"]["secondary"]
    scores = ", ".join(f"{SHORT[k.split('/')[1]]} {_n(v['rare_macro_f1']['point'])} ({_n(v['f1_S']['point'])}, {_n(v['f1_F']['point'])})"
                       for k, v in summary["conditions"].items())  # fmt: skip
    text += f" On the test records (Figure~\\ref{{fig:p12}}), rare-class macro-F1 (S F1, F F1): {scores}."
    p11 = _load(paths.results_for(PHASE6_MORE_PATIENTS) / "summary.json")
    if p11 is not None:
        text += " Per class, from the phase 3 generator (P11) to the retrained one: " + "; ".join(
            f"{SHORT[k.split('/')[1]]} S F1 {_n(p11['conditions'][k]['f1_S']['point'])} to {_n(v['f1_S']['point'])} and F F1 "
            f"{_n(p11['conditions'][k]['f1_F']['point'])} to {_n(v['f1_F']['point'])}" for k, v in summary["conditions"].items()) + "."
    text += (f" Against the same generator in P11: {_family_sentence(conf)}. Against P11's cnn without augmentation and with random oversampling: "
             f"{_family_sentence(sec)}.")
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_record_rr() -> str:
    """P13: rr_prev and rr_next over the recording's mean RR interval, P11's and P12's conditions, from results/phase6_record_rr/."""
    from ecgaug.names import PHASE6_MORE_PATIENTS, PHASE6_RECORD_RR, PHASE6_RELATIVE_GENERATORS

    folder = paths.results_for(PHASE6_RECORD_RR)
    check, selection, summary = (_load(folder / f) for f in ("design_check.json", "selection.json", "summary.json"))
    if check is None:
        return _missing("P13: --part check (training-side annotations)") + "\n"
    names = {"v1": "v1's ratio", "v2": "v2's", "record": "the recording's mean"}
    heavy = sorted(check["s_heavy_records"], key=lambda h: -h["s_beats"])
    text = "\\paragraph{The design check.} On the annotations of the training-side databases only (DS1, INCART, SVDB),"
    if heavy:
        h = heavy[0]
        text += (f" the S beats of SVDB record {h['record']} ({int(h['s_beats']):,} of {int(h['n_and_s_beats']):,} N and S beats, median ratio of the "
                 f"following to the preceding RR interval {_n(h['median_post_pre_of_s'], 2)}) are found by the rule ``ratio below {check['threshold']}'' "
                 "in " + ", ".join(f"{_n(100 * h['rule_s_sensitivity'][r], 1)}\\% with {names[r]}" for r in ("v1", "v2", "record"))
                 + " (Figure~\\ref{fig:p13-check}).")  # fmt: skip
    dbs = {"mitdb_ds1": "DS1", "incartdb": "INCART", "svdb": "SVDB"}
    text += (" Over whole databases the recording-mean ratio is not better everywhere: the rule's S F1 with v2's ratio and with the recording's "
             "mean is " + "; ".join(f"{dbs.get(db, db)} {_n(e['rule']['v2']['f1'])} and {_n(e['rule']['record']['f1'])}" for db, e in check["databases"].items())
             + ", so the new features are added beside v2's ratio, not in its place.")  # fmt: skip
    if selection is None:
        return text + " " + _missing("P13 results: --part grid on Kaggle, then --part select and --part test") + "\n"
    before_sel = {**(_load(paths.results_for(PHASE6_MORE_PATIENTS) / "selection.json") or {"chosen": {"cnn": {}}})["chosen"]["cnn"],
                  **{f: v for f, v in (_load(paths.results_for(PHASE6_RELATIVE_GENERATORS) / "selection.json") or {"chosen": {"cnn": {}}})["chosen"]["cnn"].items()}}  # fmt: skip
    chosen = selection["chosen"]["cnn"]
    text += (" Validation S F1, before and with the recording-mean RR: " + ", ".join(
        f"{SHORT[c]} {_n(before_sel[c]['mean'])} and {_n(v['mean'])}" for c, v in chosen.items() if c in before_sel) + ".")
    if summary is None:
        return text + " " + _missing("P13 results: --part test on Kaggle") + "\n"
    scores = "; ".join(f"{SHORT[k.split('/')[1]]} {_n(summary['before'][k]['rare_macro_f1']['point'])} ({_n(summary['before'][k]['f1_S']['point'])}, "
                       f"{_n(summary['before'][k]['f1_F']['point'])}) to {_n(v['rare_macro_f1']['point'])} ({_n(v['f1_S']['point'])}, {_n(v['f1_F']['point'])})"
                       for k, v in summary["conditions"].items())  # fmt: skip
    conf, sec = summary["families"]["confirmatory"], summary["families"]["secondary"]
    rule = "; ".join(f"{SHORT[k.split('/')[1]]} {_n(v['rare_macro_f1']['point'])} (S F1 {_n(v['f1_S']['point'])})" for k, v in summary["sveb_rule"].items())
    text += (f" On the test records (Figure~\\ref{{fig:p13}}), rare-class macro-F1 (S F1, F F1) before and with the recording-mean RR: {scores}. "
             f"Against the same condition before: {_family_sentence(conf)}. With the new features, each generator against none, random oversampling "
             f"and class weighting: {_family_sentence(sec)}. With Huang et al.'s SVEB stage on top (a beat the cnn calls N becomes S when its RR "
             f"interval is below {summary['rule_threshold']} of the recording's mean; descriptive, not tested): {rule}.")  # fmt: skip
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_icentia() -> str:
    """P14: the cnn trained on P13's data plus Icentia11k patients, from results/phase6_icentia/."""
    from ecgaug.names import PHASE6_ICENTIA, PHASE6_RECORD_RR

    folder = paths.results_for(PHASE6_ICENTIA)
    check, data, selection, summary = (_load(folder / f) for f in ("design_check.json", "data.json", "selection.json", "summary.json"))
    if check is None:
        return _missing("P14: --part check (training-side annotations)") + "\n"
    rules, models = check["rules"], {k: v["without_held_out"] for k, v in check["models"].items()}
    local = {k: v for k, v in {**rules, **models}.items() if "recording mean" not in k or "local" in k or "v2" in k}
    local = {k: v for k, v in local.items() if "irregularity" not in k}
    record = {k: v for k, v in {**rules, **models}.items() if k in ("recording mean < 0.8", "logistic regression: recording mean only")}
    veto = {k: v for k, v in rules.items() if "irregularity" in k}
    lo_hi = lambda vals, digits=3: f"{_n(min(vals), digits)} to {_n(max(vals), digits)}"  # noqa: E731
    text = (f"\\paragraph{{The design check.}} On the training-side annotations only, with SVDB record {check['held_out'][1]} left out of training as an "
            f"unseen S-dominant patient ({check['held_out_s_beats']:,} S beats), every rule or interval model with a local reference reaches a "
            f"validation S F1 of up to {_n(max(v['validation']['f1'] for v in local.values()))} but finds "
            f"{lo_hi([100 * v['held_out_s_sensitivity'] for v in local.values()], 0)}\\% of the unseen patient's S beats; the recording mean "
            f"(Huang's rule, or a logistic regression on the recording-mean intervals alone) finds {lo_hi([100 * v['held_out_s_sensitivity'] for v in record.values()], 0)}\\% "
            f"at a validation S F1 of {lo_hi([v['validation']['f1'] for v in record.values()])}"
            + (f" (a gradient-boosted model on the same two intervals finds {_n(100 * models['gradient boosting: recording mean only']['held_out_s_sensitivity'], 0)}\\%)"
               if "gradient boosting: recording mean only" in models else "")
            + "; vetoing irregular rhythm brings validation S F1 down to "
            + f"{lo_hi([v['validation']['f1'] for v in veto.values()])} (Figure~\\ref{{fig:p14-check}}). Which reference to trust has to be learned from patients "
            "who show both patterns, and P11's training set holds one S-dominant patient.")  # fmt: skip
    if data is None:
        return text + " " + _missing("P14 data: --part grid on Kaggle") + "\n"
    used = [r for r in data["records"] if "excluded" not in r]
    beats = data["icentia_beats"]
    text += (f" P14 adds the first segment of {len(data['patients'])} Icentia11k patients drawn at random \\cite{{tan2022icentia,tan2021icentia}} "
             f"({len(used)} usable; training beats N {beats['train']['N']:,}, S {beats['train']['S']:,}, V {beats['train']['V']:,}; validation "
             f"S {beats['val']['S']:,}) and retrains the cnn in four of P13's conditions at P13's strengths.")  # fmt: skip
    if selection is not None:
        before = (_load(paths.results_for(PHASE6_RECORD_RR) / "selection.json") or {"chosen": {"cnn": {}}})["chosen"]["cnn"]
        text += (" Validation S F1 (now including the Icentia11k validation patients), P13 and with them: " + ", ".join(
            f"{SHORT[c]} {_n(before[c]['mean'])} and {_n(v['mean'])}" for c, v in selection["chosen"]["cnn"].items() if c in before) + ".")
    if summary is None:
        return text + " " + _missing("P14 results: --part select, then --part test") + "\n"
    scores = "; ".join(f"{SHORT[k.split('/')[1]]} {_n(summary['before'][k]['rare_macro_f1']['point'])} ({_n(summary['before'][k]['f1_S']['point'])}, "
                       f"{_n(summary['before'][k]['f1_F']['point'])}) to {_n(v['rare_macro_f1']['point'])} ({_n(v['f1_S']['point'])}, {_n(v['f1_F']['point'])})"
                       for k, v in summary["conditions"].items())  # fmt: skip
    rule = "; ".join(f"{SHORT[k.split('/')[1]]} {_n(v['rare_macro_f1']['point'])} (S F1 {_n(v['f1_S']['point'])})" for k, v in summary["sveb_rule"].items())
    text += (f" On the test records (Figure~\\ref{{fig:p14}}), rare-class macro-F1 (S F1, F F1) in P13 and with the Icentia11k patients: {scores}. "
             f"Against P13: {_family_sentence(summary['families']['confirmatory'])}. The cVAE against the other conditions: "
             f"{_family_sentence(summary['families']['secondary'])}. With Huang et al.'s SVEB stage on top (descriptive): {rule}.")  # fmt: skip
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def _scores(entry: dict) -> str:
    return f"{_n(entry['rare_macro_f1']['point'])} (S F1 {_n(entry['f1_S']['point'])}, F F1 {_n(entry['f1_F']['point'])})"


def posthoc_diffusion() -> str:
    """P15: a class-conditional diffusion model as the third generator family, from results/phase6_diffusion/."""
    from ecgaug.names import PHASE6_DIFFUSION, PHASE6_RECORD_RR

    folder = paths.results_for(PHASE6_DIFFUSION)
    gens, mech, selection, summary = (_load(folder / f) for f in ("diffusion_generators.json", "mechanism.json", "selection.json", "summary.json"))
    if gens is None:
        return _missing("P15: --part train on Kaggle") + "\n"
    s, seeds = gens["settings"], gens["seeds"]
    minutes = [e["generator_seconds"] / 60 for e in seeds.values()]
    text = (f"\\paragraph{{A diffusion model as the generator (P15).}} The study's generators are a cVAE and a WGAN-GP; the current standard for "
            f"synthetic ECG is a diffusion model \\cite{{alcaraz2023sssd,kwak2024diffusion}}. P15 trains a class-conditional DDPM ({s['timesteps']:,} "
            f"noise levels, cosine schedule; a 1D U-Net with {next(iter(seeds.values()))['n_parameters']:,} parameters; classifier-free guidance, scale "
            f"{s['guidance']}; DDIM sampling with {s['sample_steps']} levels; weights averaged with decay {s['ema_decay']}) on P13's training beats in the "
            f"form the cnn sees them, with the phase 3 budget ({gens['steps']:,} steps, batch {s['batch_size']}, seeds "
            f"{', '.join(seeds)}): {sum(not e['diverged'] for e in seeds.values())} of {len(seeds)} fits finished without divergence, "
            + (f"about {_n(min(minutes), 0)} minutes each on a T4." if round(min(minutes)) == round(max(minutes)) else f"{_n(min(minutes), 0)} to {_n(max(minutes), 0)} minutes each on a T4."))  # fmt: skip
    if mech is not None:
        eff = {c: {k: v["effective_records"] for k, v in e.items() if isinstance(v, dict) and "effective_records" in v} for c, e in mech["classes"].items()}
        text += (" Its synthetic beats resemble more training patients than the cVAE's (effective number of patients of the nearest real beat, "
                 + "; ".join(f"{c}: diffusion {_n(e['diffusion'], 1)}, cVAE {_n(e['cvae_p12'], 1)}, real beats {_n(e['real'], 1)}" for c, e in eff.items())
                 + "), but their class shapes match the real training beats less closely than the cVAE's, and the training loss was still "
                 "falling at the end of the budget (Figure~\\ref{fig:p15-samples}).")  # fmt: skip
    if selection is None:
        return text + " " + _missing("P15: --part select") + "\n"
    chosen = selection["chosen"]["cnn"]["diffusion"]
    before = (_load(paths.results_for(PHASE6_RECORD_RR) / "selection.json") or {"chosen": {"cnn": {}}})["chosen"]["cnn"]
    text += (f" On validation the phase 4 gate rule chose strength {chosen['strength']} (S F1 {_n(chosen['mean'])} $\\pm$ {_n(chosen['sd'])}; P13: "
             + ", ".join(f"{SHORT[c]} {_n(v['mean'])}" for c, v in before.items() if c in ("none", "cvae")) + ").")
    if summary is None:
        return text + " " + _missing("P15: --part test") + "\n"
    new = summary["conditions"]["cnn/diffusion"]
    old = "; ".join(f"{SHORT[k.split('/')[1]]} {_scores(v)}" for k, v in summary["before"].items())
    rule = "; ".join(f"{SHORT.get(k.split('/')[1], k.split('/')[1])} {_n(v['rare_macro_f1']['point'])} (S F1 {_n(v['f1_S']['point'])})" for k, v in summary["sveb_rule"].items())
    text += (f" On the test records (Figure~\\ref{{fig:p15}}) the diffusion condition reaches rare-class macro-F1 {_scores(new)}, against P13's {old}. "
             f"Confirmatory: {_family_sentence(summary['families']['confirmatory'])}. Secondary: {_family_sentence(summary['families']['secondary'])}. "
             f"With Huang et al.'s SVEB stage on top (descriptive): {rule}.")  # fmt: skip
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def _window_loss(experiment: str, end: int) -> list[float]:
    """Per seed of a diffusion experiment, the mean training loss over steps ``end - 999`` to ``end``; seeds whose run log is
    not on disk are left out."""
    from ecgaug import runs

    out = []
    for seed in ((_load(paths.results_for(experiment) / "diffusion_generators.json") or {}).get("seeds") or {}).values():
        run_dir = runs.find_run_dir(seed["run_id"])
        if run_dir is None or not (run_dir / "train_log.csv").is_file():
            continue
        log = np.genfromtxt(run_dir / "train_log.csv", delimiter=",", names=True)
        at = (log["step"] > end - 1000) & (log["step"] <= end)
        if at.any():
            out.append(float(log["loss"][at].mean()))
    return out


def posthoc_diffusion_long() -> str:
    """P17: P15's diffusion model trained five times longer, from results/phase6_diffusion_long/."""
    from ecgaug.names import PHASE6_DIFFUSION, PHASE6_DIFFUSION_LONG

    folder = paths.results_for(PHASE6_DIFFUSION_LONG)
    shape, gens, mech, selection, summary = (_load(folder / f) for f in ("shape_check.json", "diffusion_generators.json", "mechanism.json", "selection.json", "summary.json"))
    if shape is None:
        return _missing("P17: --part shape (local)") + "\n"

    def vals(name: str, cls: str, key: str = "corr") -> list[float]:
        return [e[cls][key] for e in shape["banks"].get(name, {}).values() if key in e[cls]]

    def span(values: list[float], digits: int = 2) -> str:
        if not values:
            return "--"
        lo, hi = _n(min(values), digits), _n(max(values), digits)
        return (lo if lo == hi else f"{lo} to {hi}").replace("-", "$-$")

    def corr(name: str, cls: str, key: str = "corr") -> str:
        return span(vals(name, cls, key))

    text = ("\\paragraph{The diffusion model trained five times longer (P17).} At the phase 3 budget P15's loss was still falling, and the median "
            "beat of each class in its banks matched the median of real DS1 training beats far less closely than the cVAE's (correlation over the "
            f"three seed banks S {corr('diffusion_p15', 'S')}, V {corr('diffusion_p15', 'V')}, F {corr('diffusion_p15', 'F')}, against "
            f"{corr('cvae_p12', 'S')}, {corr('cvae_p12', 'V')} and {corr('cvae_p12', 'F')}). P17 trains the same model for five times as many steps, "
            "with every other setting unchanged.")  # fmt: skip
    if gens is None:
        return text + " " + _missing("P17: --part train on Kaggle") + "\n"
    seeds = gens["seeds"]
    minutes = [e["generator_seconds"] / 60 for e in seeds.values()]
    ok = sum(not e["diverged"] for e in seeds.values())
    took = f"about {_n(min(minutes), 0)}" if round(min(minutes)) == round(max(minutes)) else f"{_n(min(minutes), 0)} to {_n(max(minutes), 0)}"
    text += (f" {'All' if ok == len(seeds) else f'{ok} of'} {len(seeds)} fits of {gens['steps']:,} steps finished without divergence "
             f"({took} minutes each on a T4).")
    early, late = _window_loss(PHASE6_DIFFUSION_LONG, 20000), _window_loss(PHASE6_DIFFUSION_LONG, int(gens["steps"]))
    if early and late:
        p15 = _window_loss(PHASE6_DIFFUSION, 20000)
        same = ", as in P15," if p15 and span(p15, 3) == span(early, 3) else (f" (P15 {span(p15, 3)})" if p15 else "")
        text += (f" The loss kept falling: {span(early, 3)} over steps 19,001--20,000{same} and {span(late, 3)} over the last 1,000 steps "
                 "(Figure~\\ref{fig:p17-check}).")
    if "diffusion_p17" in shape["banks"]:
        below = all(vals("diffusion_p17", c) and vals("cvae_p12", c) and max(vals("diffusion_p17", c)) < min(vals("cvae_p12", c)) for c in "SVF")
        text += (f" Its banks' class medians correlate with the real ones at S {corr('diffusion_p17', 'S')}, V {corr('diffusion_p17', 'V')} and F "
                 f"{corr('diffusion_p17', 'F')}" + (", below the cVAE's in every class." if below else "."))
    align = shape.get("alignment")
    if align is not None and all(vals(k, c, "aligned") for k in ("diffusion_p17", "diffusion_p15", "cvae_p12") for c in "SVF"):
        text += (f" An exploratory count, added after this training and before the test read, gives the share of beats whose largest deflection "
                 f"lies within {align['window']} samples ({align['window'] / 0.36:.0f}~ms) of where the real beats have theirs: real beats "
                 + ", ".join(f"{c} {_n(align['real'][c], 2)}" for c in "SVF") + "; "
                 + "; ".join(f"{label} " + ", ".join(f"{c} {corr(key, c, 'aligned')}" for c in "SVF")
                             for key, label in (("diffusion_p17", "P17"), ("diffusion_p15", "P15"), ("cvae_p12", "the cVAE"))) + ".")  # fmt: skip
        closer = all(min(vals("diffusion_p17", c, "aligned")) > max(vals("diffusion_p15", c, "aligned")) for c in "VF")
        short = all(max(vals("diffusion_p17", c, "aligned")) < align["real"][c] for c in "VF")
        if closer:
            text += " The longer training put more V and F peaks where the real beats have theirs" + (", though still fewer than among real beats." if short else ".")
    if mech is not None:
        eff = {c: e["diffusion"]["effective_records"] for c, e in mech["classes"].items()}
        text += " Effective number of training patients its beats resemble: " + ", ".join(f"{c} {_n(v, 1)}" for c, v in eff.items()) + "."
    if selection is None:
        return text + " " + _missing("P17: --part select") + "\n"
    chosen = selection["chosen"]["cnn"]["diffusion"]
    p15 = (_load(paths.results_for(PHASE6_DIFFUSION) / "selection.json") or {"chosen": {"cnn": {}}})["chosen"]["cnn"].get("diffusion")
    text += (f" On validation the phase 4 gate rule chose strength {chosen['strength']} (S F1 {_n(chosen['mean'])} $\\pm$ {_n(chosen['sd'])}"
             + (f"; P15 {_n(p15['mean'])}" if p15 else "") + ").")
    if summary is None:
        return text + " " + _missing("P17: --part test") + "\n"
    new = summary["conditions"]["cnn/diffusion_long"]
    rule, veto = summary["sveb_rule"]["cnn/diffusion_long"], summary["with_af_veto"]["cnn/diffusion_long"]
    text += (f" On the test records (Figure~\\ref{{fig:p17}}) it reaches rare-class macro-F1 {_scores(new)}; P15's model {_scores(summary['before']['cnn/diffusion'])}, "
             f"P13's cVAE {_scores(summary['before']['cnn/cvae'])}. Confirmatory: {_family_sentence(summary['families']['confirmatory'])}. Secondary: "
             f"{_family_sentence(summary['families']['secondary'])}. With Huang et al.'s SVEB stage on top {_n(rule['rare_macro_f1']['point'])} "
             f"(S F1 {_n(rule['f1_S']['point'])}), and with P16's AF veto as well {_n(veto['rare_macro_f1']['point'])} (S F1 {_n(veto['f1_S']['point'])}; "
             "descriptive). Only the diffusion model had the longer budget, so P17 against the other generators is not an equal-budget comparison.")  # fmt: skip
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_af_rule() -> str:
    """P16: Huang's SVEB stage with an atrial-fibrillation veto, from results/phase6_af_rule/."""
    from ecgaug.names import PHASE6_AF_RULE

    folder = paths.results_for(PHASE6_AF_RULE)
    det, check, summary = (_load(folder / f) for f in ("af_detector.json", "design_check.json", "summary.json"))
    if det is None or check is None or "validation" not in check:
        return _missing("P16: --part fit and --part check (annotations only)") + "\n"
    r, v = check["ds1_rhythm"], check["validation"]["mean_s_f1"]
    dbs = {"mitdb_ds1": "DS1", "incartdb": "INCART", "svdb": "SVDB"}
    t = check["chosen_threshold"]
    text = (f"\\paragraph{{An atrial-fibrillation veto on the SVEB stage (P16).}} Huang et al.'s stage reads only a beat's RR interval over its "
            f"recording's mean, so an irregular rhythm makes normal beats look premature: on DS1's annotations {r['rule_false_s_in_af']:,} of its "
            f"{r['rule_false_s']:,} false S calls lie in atrial fibrillation or flutter, where {r['s_beats_in_af']} of {r['s_beats']:,} S beats lie "
            "(without organized atrial activity a beat is not labelled S). P16 undoes the stage's S calls where an AF detector fires. The detector reads "
            f"only beat times (five irregularity measures of the 64 surrounding RR intervals after premature beats and their pauses are removed; a "
            f"logistic regression) and is fitted on the {len(det['records'])} records of the MIT-BIH Atrial Fibrillation Database \\cite{{moody1983afdb}} "
            f"({det['beats']:,} beats; sensitivity {_n(det['training_sensitivity'])}, specificity {_n(det['training_specificity'])}). Its threshold, "
            f"{t}, has the highest mean S F1 of the vetoed stage over the training-side databases ("
            + ", ".join(f"{dbs[db]} {_n(e['rule']['f1'])} to {_n(e[f'veto_{t}']['f1'])}" for db, e in check["databases"].items())
            + f"); its form, undoing only the stage's own conversions, has the higher validation S F1 over the compared conditions and seeds "
            f"({_n(v['veto_rule'])}, against {_n(v['veto_all'])} for undoing every S call there; the stage alone {_n(v['rule'])}, the cnn alone "
            f"{_n(v['cnn'])}; Figure~\\ref{{fig:p16-check}}).")  # fmt: skip
    if summary is None:
        return text + " " + _missing("P16: --part test") + "\n"
    rows = "; ".join(f"{SHORT.get(k.split('/')[1], k.split('/')[1])} {_n(summary['rule_alone'][k]['rare_macro_f1']['point'])} to "
                     f"{_n(e['rare_macro_f1']['point'])} (S F1 {_n(summary['rule_alone'][k]['f1_S']['point'])} to {_n(e['f1_S']['point'])}, "
                     f"+P {_n(summary['rule_alone'][k]['ppv_S']['point'])} to {_n(e['ppv_S']['point'])})"
                     for k, e in summary["with_af_veto"].items())  # fmt: skip
    text += (f" On the test records (Figure~\\ref{{fig:p16}}), rare-class macro-F1 with the stage alone and with the veto: {rows}. "
             f"Confirmatory: {_family_sentence(summary['families']['confirmatory'])}.")
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_seed_ensemble() -> str:
    """P18: each condition's ten runs averaged, with Huang's SVEB stage and P16's AF veto on top, from results/phase6_seed_ensemble/."""
    from ecgaug.names import PHASE6_SEED_ENSEMBLE

    folder = paths.results_for(PHASE6_SEED_ENSEMBLE)
    members, summary = _load(folder / "members.json"), _load(folder / "summary.json")
    if members is None:
        return _missing("P18: --part grid on Kaggle, then --part check") + "\n"
    e = members["epochs"]["seeds_0_2"]
    j = e["largest_jump"]
    text = ("\\paragraph{Ten runs averaged per condition (P18).} Each run of the cnn keeps the epoch with the highest validation S F1. In the "
            f"{e['runs']} runs P16 compares, that score moved by {_n(j['min'], 2)} to {_n(j['max'], 2)} between consecutive epochs (median "
            f"{_n(j['median'], 2)}), and {e['kept_first_epoch']} of them kept their first epoch, so the model a run keeps is a noisy draw. "
            "Averaging the class probabilities of independently trained runs, a deep ensemble \\cite{lakshminarayanan2017ensembles}, is the "
            "standard remedy, and such ensembles held up best under dataset shift in Ovadia et al.'s comparison \\cite{ovadia2019shift}. P18 trains seeds 3--9 of P16's six "
            "conditions with every setting unchanged (a run of seed $s$ uses bank seed $s \\bmod 3$), averages each condition's ten runs before "
            "the argmax, and puts Huang et al.'s stage and P16's veto on top. Nothing is chosen on validation.")  # fmt: skip
    v, m = members["validation"], members["mean_s_f1"]["veto"]
    up = {cls: sum(v[c]["veto"][cls]["ten"] > float(np.mean(v[c]["veto"][cls]["single"])) for c in v) for cls in ("S", "V")}
    cv = v["cvae"]["veto"]
    text += (f" On the validation patients (Figure~\\ref{{fig:p18-check}}; the runs' stored predictions), with the stage and the veto, the mean "
             f"S F1 over the conditions was {_n(m['single'])} for single runs, {_n(m['three'])} for three-run and {_n(m['ten'])} for ten-run "
             f"ensembles. The ten-run ensemble beat its single runs' mean S F1 in {up['S']} of {len(v)} conditions and their mean V F1 in "
             f"{up['V']} (the cVAE: S F1 {_n(float(np.mean(cv['S']['single'])))} to {_n(cv['S']['ten'])}, V F1 "
             f"{_n(float(np.mean(cv['V']['single'])))} to {_n(cv['V']['ten'])}).")  # fmt: skip
    if summary is None:
        return text + " " + _missing("P18: --part test") + "\n"
    ens, p16 = summary["ensemble"]["with_af_veto"], summary["p16"]["with_af_veto"]
    rows = "; ".join(f"{SHORT[k.split('/')[1]]} {_n(p16[k]['rare_macro_f1']['point'])} to {_n(x['rare_macro_f1']['point'])}" for k, x in ens.items())
    cvt, cpt = ens["cnn/cvae"], p16["cnn/cvae"]
    points = summary["per_run_points"]["cnn/cvae"]
    top = ", the highest test score of the study" if cvt["rare_macro_f1"]["point"] >= _best_test_score() else ""
    w202 = summary["without_202"]
    text += (f" On the test records (Figure~\\ref{{fig:p18}}), with the stage and the veto, rare-class macro-F1 went from P16's single runs to "
             f"the ten-run ensemble as follows: {rows}. The cVAE's ensemble reached {_ci(cvt['rare_macro_f1'])}{top} (S F1 {_n(cpt['f1_S']['point'])} "
             f"to {_n(cvt['f1_S']['point'])}, F F1 {_n(cpt['f1_F']['point'])} to {_n(cvt['f1_F']['point'])}); its ten single runs scored "
             f"{_n(min(points))} to {_n(max(points))} (mean {_n(float(np.mean(points)))}; seeds 0--2, P16's runs, {_n(float(np.mean(points[:3])))}), "
             f"and without record 202 the ensemble scored {_n(w202['ensemble']['cnn/cvae']['rare_macro_f1']['point'])} "
             f"(P16 {_n(w202['p16']['cnn/cvae']['rare_macro_f1']['point'])}). Confirmatory: {_family_sentence(summary['families']['confirmatory'])}. "
             f"Secondary: {_family_sentence(summary['families']['secondary'])}. An ensemble needs ten trained models where P16 used one, and P16's "
             "scores stay as reported.")  # fmt: skip
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_fresh_patients() -> str:
    """P19: P18's frozen pipeline on 1,000 unused Icentia11k patients, from results/phase6_fresh_patients/."""
    from ecgaug.names import PHASE6_FRESH_PATIENTS, PHASE6_SEED_ENSEMBLE

    s = _load(paths.results_for(PHASE6_FRESH_PATIENTS) / "summary.json")
    text = ("\\paragraph{The same pipeline on 1,000 new patients (P19).} P18's highest score is the best of 14 post hoc reads of the same 22 "
            "test patients, one of whom holds three quarters of their S beats. P19 runs P18's sixty runs, Huang et al.'s stage and P16's veto "
            "unchanged on the first segment (about 70 minutes) of 1,000 Icentia11k patients \\cite{tan2022icentia} drawn at random from those "
            "no analysis had used, keeping every usable beat; nothing is fitted or chosen, and the primary measure is S F1 because "
            "Icentia11k has no F labels.")  # fmt: skip
    if s is None:
        return text + " " + _missing("P19: --part run on Kaggle") + "\n"
    b, d = s["beats"], s.get("data", {})
    text += (f" {s['patients']:,} segments were usable ({b['N']:,} N, {b['S']:,} S and {b['V']:,} V beats"
             + (f"; {d['patients_with']['S']:,} patients with an S beat, {d['s_dominant_patients']} with more S than N beats, the largest share "
                f"of S beats in one patient {_n(d['largest_s_share'], 2)}" if "patients_with" in d else "") + ").")  # fmt: skip
    ens, alone = s["ensemble"]["with_af_veto"], s["ensemble"]["cnn_alone"]
    cv = ens["cnn/cvae"]
    p18 = _load(paths.results_for(PHASE6_SEED_ENSEMBLE) / "summary.json")
    on_ds2 = ""
    if p18 is not None:
        e = p18["ensemble"]["with_af_veto"]["cnn/cvae"]
        on_ds2 = f" (on DS2 {_n(e['f1_S']['point'])} and {_n(e['f1_V']['point'])})"
    rows = "; ".join(f"{SHORT[k.split('/')[1]]} {_n(e['f1_S']['point'])} (alone {_n(alone[k]['f1_S']['point'])})" for k, e in ens.items())
    text += (f" With the stage and the veto, the cVAE's ten-run ensemble reached S F1 {_ci(cv['f1_S'])} and V F1 {_ci(cv['f1_V'])}{on_ds2}; "
             f"S Se {_n(cv['se_S']['point'])}, S +P {_n(cv['ppv_S']['point'])} (Figure~\\ref{{fig:p19}}). S F1 per condition, with the stage "
             f"and the veto and for the ensemble alone: {rows}. Confirmatory: {_family_sentence(s['families']['confirmatory'])}. Secondary: "
             f"{_family_sentence(s['families']['secondary'])}.")  # fmt: skip
    veto = {k.split(": ")[0]: v for k, v in s["stages"].items() if k.endswith("veto vs cnn alone")}
    up = sum(v["ci_low"] > 0 for v in veto.values())
    down = sum(v["ci_high"] < 0 for v in veto.values())
    text += (f" Against the ensemble alone, the stage with the veto changed S F1 by {_s(min(v['point'] for v in veto.values()))} to "
             f"{_s(max(v['point'] for v in veto.values()))} (interval above zero in {up} of {len(veto)} conditions, below zero in {down}; descriptive).")
    rule, lone = s["ensemble"]["rule_alone"]["cnn/cvae"], alone["cnn/cvae"]
    text += (f" For the cVAE, Huang et al.'s stage raised S Se from {_n(lone['se_S']['point'])} to {_n(rule['se_S']['point'])} and lowered S +P from "
             f"{_n(lone['ppv_S']['point'])} to {_n(rule['ppv_S']['point'])}; the veto took S +P back to {_n(cv['ppv_S']['point'])}. The stage's 0.8 "
             "threshold comes from Huang et al.'s MIT-BIH study and the veto was chosen on MIT-BIH's training patients; choosing both on a larger, "
             "more varied pool of patients is the next step.")  # fmt: skip
    return text + " Post hoc; Icentia11k comes from a different device and annotation process, and the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_rule_tuning() -> str:
    """P20: Huang's stage and P16's veto chosen on 400 Icentia11k patients, scored on P19's patients and the test records,
    from results/phase6_rule_tuning/."""
    from ecgaug.names import PHASE6_RULE_TUNING

    folder = paths.results_for(PHASE6_RULE_TUNING)
    choice, s, test = (_load(folder / name) for name in ("choice.json", "summary.json", "test_summary.json"))
    text = ("\\paragraph{The stage and the veto chosen on other patients (P20).} On P19's patients Huang et al.'s stage lowered S F1 against "
            "the ensembles alone. Its threshold 0.8 is Huang et al.'s MIT-BIH value, and P16's veto threshold came from MIT-BIH, INCART and SVDB "
            "annotations. P20 chooses, per condition, one of 50 settings (the ten-run ensemble alone, or the stage at a threshold from 0.60 to 0.90 "
            "with no veto or with the veto at a threshold from 0.50 to 0.95) by S F1 on P14's 400 Icentia11k patients, which none of P18's runs "
            "has seen, then scores the choice on P19's patients, which played no part in it, and once on the test records.")  # fmt: skip
    if choice is None or s is None:
        return text + " " + _missing("P20: --part build, then --part run on Kaggle") + "\n"
    picks = "; ".join(f"{SHORT[c]}: {v['label']}" for c, v in choice["chosen"].items())
    text += f" Chosen on {choice['patients']:,} tuning patients ({choice['beats']['S']:,} S beats): {picks} (Figure~\\ref{{fig:p20-choice}})."
    cond = s["conditions"]
    rows = "; ".join(f"{SHORT[k.split('/')[1]]} {_n(e['f1_S']['point'])} (P19's setting {_n(cond['p19_setting'][k]['f1_S']['point'])}, "
                     f"alone {_n(cond['ensemble_alone'][k]['f1_S']['point'])})" for k, e in cond["chosen"].items())  # fmt: skip
    text += (f" On P19's {s['patients']:,} patients, S F1 with the chosen setting: {rows} (Figure~\\ref{{fig:p20}}). Confirmatory, chosen vs "
             f"P19's setting: {_family_sentence(s['families']['confirmatory'])}. Secondary, chosen vs the ensemble alone: "
             f"{_family_sentence(s['families']['secondary'])}.")  # fmt: skip
    if test is None:
        text += " " + _missing("P20: --part test on Kaggle")
    else:
        tc = test["conditions"]
        rows = "; ".join(f"{SHORT[k.split('/')[1]]} {_n(e['rare_macro_f1']['point'])} (P18's setting {_n(tc['p18_setting'][k]['rare_macro_f1']['point'])})"
                         for k, e in tc["chosen"].items())  # fmt: skip
        cv = tc["chosen"]["cnn/cvae"]
        text += (f" On the test records (one read), rare-class macro-F1 with the chosen settings: {rows}; for the cVAE S F1 {_n(cv['f1_S']['point'])}, "
                 f"F F1 {_n(cv['f1_F']['point'])} and V F1 {_n(cv['f1_V']['point'])}. Chosen vs P18's setting: {_family_sentence(test['families']['ds2'])}.")  # fmt: skip
        data = _load(folder / "data.json")
        errors = _load(paths.results_for("error_analysis") / "test.json")
        if data is not None and errors:
            support = {rec: v["S"]["support"] for rec, v in next(iter(errors.values()))["per_record"].items() if "S" in v}
            top = max(support, key=support.get)
            p18_s = tc["p18_setting"]["cnn/cvae"]["f1_S"]["point"]
            text += (f" The setting that helps on the new patients lowers the test score through S F1 (cVAE {_n(p18_s)} with P18's setting, "
                     f"{_n(cv['f1_S']['point'])} with the chosen one): record {top} holds {support[top]:,} of the {sum(support.values()):,} test S beats, "
                     f"and {_count_word(data['tuning']['s_dominant_patients'])} of the {data['tuning']['segments_used']} tuning patients has more S than N beats. "
                     "Choosing the stage for patients like record " + str(top) + " needs more such patients than these data hold.")  # fmt: skip
    return text + " Post hoc; the primary results stay those of Table~\\ref{tab:test}.\n"


def posthoc_table() -> str:
    """One row per post-hoc analysis (P1-P8): what it asked and what it found, from the summaries."""
    from ecgaug.names import (PHASE6_AUDIT_CHECK, PHASE6_AUGMENTATION_MECHANISM, PHASE6_EXTERNAL_VALIDATION, PHASE6_EXTRA_SEEDS,
                              PHASE6_PATIENT_TEMPLATE, PHASE6_RELATIVE_CNN, PHASE6_RR_BASELINE, PHASE6_RR_REFERENCE)  # fmt: skip

    def verdicts(conf: dict) -> str:
        verbs = [_changed(c) for c in conf.values()]
        return ", ".join(f"{v} in {verbs.count(v)}" for v in dict.fromkeys(verbs)) + f" of {len(verbs)}"

    rows = []
    r = _load(paths.results_for(PHASE6_RR_REFERENCE) / "summary.json")
    if r:
        rows.append(("P1", "Is the cnn simply weak?", f"RR-only LDA {_n(r['references']['rr_lda']['rare_macro_f1']['point'])} vs cnn {_n(r['cnn']['none']['rare_macro_f1']['point'])}"))
    ext = [(db, _load(paths.results_for(PHASE6_EXTERNAL_VALIDATION) / f"{db}_summary.json")) for db in DATABASE]
    if all(e for _, e in ext):
        rows.append(("P2", "Does it hold on other databases?", "; ".join(f"{ {'incartdb': 'INCART', 'svdb': 'SVDB'}[db]}: generative vs none {verdicts(e['confirmatory'])}" for db, e in ext)))
    s = _load(paths.results_for(PHASE6_EXTRA_SEEDS) / "summary.json")
    if s:
        rows.append(("P3", "Are three seeds enough?", f"five seeds: {len(s['changed'])} verdicts changed"))
    a = _load(paths.results_for(PHASE6_AUDIT_CHECK) / "summary.json")
    if a:
        inter = a["inter_patient"]["none"]["point"]
        nones = [a["audit_split0"]["none"]["mean"], *(c["intra"]["mean"] for c in a["checks"] if c["condition"] == "none")]
        rows.append(("P4", "Does the audit depend on its split?", f"intra-patient {_n(min(nones))}--{_n(max(nones))} vs {_n(inter)} inter-patient"))
    v5 = _load(paths.results_for(PHASE6_RR_BASELINE) / "summary.json")
    if v5:
        d = [c["point"] for c in v5["versus_v1"].values()]
        rows.append(("P5", "A better RR feature?", f"test v2 $-$ v1 {_s(min(d))} to {_s(max(d))}; generative vs none {verdicts(v5['families']['confirmatory'])}"))
    v6 = _load(paths.results_for(PHASE6_PATIENT_TEMPLATE) / "adoption.json")
    if v6:
        per = ", ".join(f"{arch} {_s(v['difference'])}" for arch, v in v6["per_architecture"].items())
        rows.append(("P6", "Patient-relative beats?", f"validation S F1 v3 $-$ v2: {per}; {'adopted' if v6['adopted'] else 'not adopted'}"))
    v7 = _load(paths.results_for(PHASE6_RELATIVE_CNN) / "summary.json")
    if v7:
        fin = _load(paths.results_for(FINAL_EVAL) / "summary.json")
        base = f" (v1 {_n(fin['conditions']['cnn/none']['rare_macro_f1']['point'])})" if fin else ""
        rows.append(("P7", "A stronger cnn?", f"test cnn none {_n(v7['conditions']['cnn/none']['rare_macro_f1']['point'])}{base}; generative vs none {verdicts(v7['families']['confirmatory'])}"))
    _, p9 = _p9()
    m = _load(paths.results_for(PHASE6_AUGMENTATION_MECHANISM) / "attribution.json")
    if m:
        s_cls = m["classes"]["S"]
        top = s_cls["real_training"]["top_record"]
        syn = [e["shares"].get(top, 0.0) for c in ("cvae", "wgan_gp") for e in s_cls.get(f"{c}_bank", {}).values()]
        rows.append(("P8", "Do synthetic beats add patients?", f"synthetic S beats nearest to record {top}: {_n(float(np.mean(syn)), 2)} (real {_n(s_cls['real_training']['top_share'], 2)})"))
    if p9:
        rows.append(("P9", "Augmentation that adds patients?", f"generative transfer vs none {verdicts(p9['families']['confirmatory'])}"))
    from ecgaug.names import PHASE6_PATIENT_ADAPTATION

    p10a = _load(paths.results_for(PHASE6_PATIENT_ADAPTATION) / "adoption.json")
    p10 = _load(paths.results_for(PHASE6_PATIENT_ADAPTATION) / "summary.json")
    if p10a:
        found = "; ".join(f"{k} {_s(e['difference'])}" for k, e in p10a["pairs"].items()) + " validation S F1"
        if p10:
            found += "; test: adaptation " + verdicts(p10["families"]["adaptation"]) + ", generative vs none " + verdicts(p10["families"]["augmentation_under_adaptation"])
        rows.append(("P10", "Per-patient normalization in the network?", found))
    from ecgaug.names import PHASE6_MORE_PATIENTS

    p11 = _load(paths.results_for(PHASE6_MORE_PATIENTS) / "summary.json")
    if p11:
        best_key, best = max(p11["conditions"].items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
        rows.append(("P11", "More training patients?", f"cnn none {_n(p11['conditions']['cnn/none']['rare_macro_f1']['point'])} vs DS1 only: "
                     + verdicts(p11["families"]["confirmatory"]) + "; each condition vs none: " + verdicts(p11["families"]["secondary"])
                     + f"; best {SHORT[best_key.split('/')[1]]} {_n(best['rare_macro_f1']['point'])}"))
    from ecgaug.names import PHASE6_RELATIVE_GENERATORS

    p12 = _load(paths.results_for(PHASE6_RELATIVE_GENERATORS) / "summary.json")
    if p12:
        rows.append(("P12", "Generators trained on what the cnn sees?", ", ".join(f"{SHORT[k.split('/')[1]]} {_n(v['rare_macro_f1']['point'])}" for k, v in p12["conditions"].items())
                     + "; vs the same generator in P11: " + verdicts(p12["families"]["confirmatory"]) + "; vs P11 none and oversampling: " + verdicts(p12["families"]["secondary"])))
    from ecgaug.names import PHASE6_RECORD_RR

    p13 = _load(paths.results_for(PHASE6_RECORD_RR) / "summary.json")
    if p13:
        best_key, best = max(p13["conditions"].items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
        rows.append(("P13", "RR intervals over the recording's mean?", f"none {_n(p13['before']['cnn/none']['rare_macro_f1']['point'])} to "
                     f"{_n(p13['conditions']['cnn/none']['rare_macro_f1']['point'])}; best {SHORT[best_key.split('/')[1]]} {_n(best['rare_macro_f1']['point'])}; "
                     "vs before: " + verdicts(p13["families"]["confirmatory"]) + "; generative vs the others: " + verdicts(p13["families"]["secondary"])))
    from ecgaug.names import PHASE6_ICENTIA

    p14 = _load(paths.results_for(PHASE6_ICENTIA) / "summary.json")
    if p14:
        best_key, best = max(p14["conditions"].items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
        rows.append(("P14", "Many more training patients (Icentia11k)?", f"none {_n(p14['before']['cnn/none']['rare_macro_f1']['point'])} to "
                     f"{_n(p14['conditions']['cnn/none']['rare_macro_f1']['point'])}; best {SHORT[best_key.split('/')[1]]} {_n(best['rare_macro_f1']['point'])}; "
                     "vs P13: " + verdicts(p14["families"]["confirmatory"]) + "; cVAE vs the others: " + verdicts(p14["families"]["secondary"])))
    if not rows:
        return _missing("post-hoc table: no summaries yet") + "\n"
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    return ("\\begin{table}[htbp]\\centering\\small\n\\caption{The post-hoc analyses (pre-registration versions 4 to 13), each reported apart from the primary "
            "results. Numbers are test rare-class macro-F1 unless stated.}\\label{tab:posthoc}\n"
            "\\begin{tabular}{@{}lp{0.3\\linewidth}p{0.55\\linewidth}@{}}\n\\toprule\n & question & finding \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def _model_sizes() -> dict[str, int]:
    """Parameter counts of the phase 2 gate's classifiers, built from their frozen settings."""
    from ecgaug.experiments.common import SIDE_FIELDS
    from ecgaug.models import build_model

    gate = workflow.manifest().get(PHASE2_GATE) or {}
    return {arch: int(sum(p.numel() for p in build_model(gate[arch]["model"], 4, len(SIDE_FIELDS)).parameters())) for arch in ("cnn", "rnn") if arch in gate}


def outlook(heading: bool = True) -> str:
    """What better results need: the limits of this study's budget, from the result files and the frozen settings."""
    from ecgaug.names import PHASE6_RELATIVE_CNN

    final = _load(paths.results_for(FINAL_EVAL) / "summary.json")
    p7 = _load(paths.results_for(PHASE6_RELATIVE_CNN) / "summary.json")
    sizes = _model_sizes()
    text = "\\paragraph{What better results need.} " if heading else ""
    text += "The results are bounded by what this study could afford, and nothing in them shows a ceiling."
    if sizes:
        text += (" The classifiers are small (" + " and ".join(f"{_n(v / 1000, 0)} thousand parameters for the {k}" for k, v in sizes.items())
                 + "), each condition was trained with three seeds, and the training settings were fixed once in phase 2.")
    if final is not None:
        widths = [c["ci_high"] - c["ci_low"] for c in final["families"]["confirmatory"].values()]
        text += (f" With 22 test patients the 95\\% intervals of the pre-registered differences are {_n(min(widths), 2)} to {_n(max(widths), 2)} wide, "
                 "so a real but smaller benefit of augmentation would not be detected; not detected is not the same as absent.")
    if p7 is not None and final is not None:
        text += (f" Changing only the input, with no retuning, raised the cnn's test rare-class macro-F1 from "
                 f"{_n(final['conditions']['cnn/none']['rare_macro_f1']['point'])} to {_n(p7['conditions']['cnn/none']['rare_macro_f1']['point'])}, "
                 "so the models are far from saturated.")
    text += (" Better results need more: larger models tuned on the patient-relative input, more runs per condition and a wider search of "
             "training settings, and more training patients (several databases); generators conditioned on the patient, or trained on many more "
             "patients, are the natural next test of augmentation. The answer of this study holds for its budget and should be revisited at that scale.")
    return text + "\n"


def patient_variation_paragraph() -> str | None:
    """Rebalancing versus patient-level methods, side by side, from the result files (None until at least two patient-level results exist)."""
    from ecgaug.names import PHASE6_PATIENT_ADAPTATION, PHASE6_PATIENT_TEMPLATE, PHASE6_PATIENT_TRANSFER, PHASE6_RELATIVE_CNN

    final = _load(paths.results_for(FINAL_EVAL) / "summary.json")
    v6 = _load(paths.results_for(PHASE6_PATIENT_TEMPLATE) / "adoption.json")
    p9 = _load(paths.results_for(PHASE6_PATIENT_TRANSFER) / "selection.json")
    p10 = _load(paths.results_for(PHASE6_PATIENT_ADAPTATION) / "adoption.json")
    p7 = _load(paths.results_for(PHASE6_RELATIVE_CNN) / "summary.json")
    p10t = _load(paths.results_for(PHASE6_PATIENT_ADAPTATION) / "summary.json")
    gains = []
    if v6 and "cnn" in v6["per_architecture"]:
        gains.append(f"patient-relative input {_s(v6['per_architecture']['cnn']['difference'])}")
    if p10 and "v1/cnn" in p10["pairs"]:
        gains.append(f"per-patient batch normalization {_s(p10['pairs']['v1/cnn']['difference'])}")
    if p9 and "cnn" in p9["chosen"]:
        gains.append(f"transfer of real rare beats onto other patients {_s(p9['chosen']['cnn']['transfer_real']['mean'] - p9['chosen']['cnn']['none']['mean'])}")
    if final is None or len(gains) < 2:
        return None
    rebal = final["families"]["confirmatory"]
    base = final["conditions"]["cnn/none"]["rare_macro_f1"]["point"]
    others = {c: final["conditions"][f"cnn/{c}"]["rare_macro_f1"]["point"] - base for c in ("class_weighted", "random_oversampling", "smote") if f"cnn/{c}" in final["conditions"]}
    text = ("\\paragraph{Patient variation, not class imbalance.} Generative augmentation, which adds beats of the rare classes, was not detectably "
            f"better than no augmentation in any of the {len(rebal)} confirmatory comparisons of the final evaluation")
    if others:
        text += (", and the simpler rebalancing methods moved the cnn's test rare-class macro-F1 by " + ", ".join(f"{_s(v)} ({SHORT[c]})" for c, v in others.items())
                 + " (point estimates, not tested against none)")
    text += ". The methods that act on the patient changed the cnn's validation S F1 without augmentation by " + ", ".join(gains) + "."
    test = {}
    if p7 is not None:
        test["patient-relative input"] = p7["conditions"]["cnn/none"]["rare_macro_f1"]["point"]
    if p10t is not None and "v1/cnn/none" in p10t["conditions"]:
        test["per-patient batch normalization"] = p10t["conditions"]["v1/cnn/none"]["rare_macro_f1"]["point"]
    from ecgaug.names import PHASE6_MORE_PATIENTS

    p11 = _load(paths.results_for(PHASE6_MORE_PATIENTS) / "summary.json")
    if test:
        text += (f" On the test records the cnn without augmentation reached {_n(base)}; with " + " and with ".join(f"{k} {_n(v)}" for k, v in test.items()) + ".")
        carried = [k for k, v in test.items() if v > base + 0.05]
        lost = [k for k in test if k not in carried]
        if carried and lost:
            text += (f" Only {' and '.join(carried)} carried the validation gain over to the test patients; {' and '.join(lost)} did not, so where the "
                     "patient is normalized matters as much as whether it is.")
    if p11 is not None:
        c = p11["conditions"]
        best_key, best = max(c.items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
        top = ", the highest test score of the study" if best["rare_macro_f1"]["point"] >= _best_test_score() else ""
        text += (f" Adding the INCART and SVDB patients (Section~\\ref{{sec:more-patients}}) changed which method is best rather than lifting the cnn "
                 f"without augmentation: it reached {_n(c['cnn/none']['rare_macro_f1']['point'])} (F F1 {_n(c['cnn/none']['f1_F']['point'])}), while "
                 f"{SHORT[best_key.split('/')[1]]} reached {_n(best['rare_macro_f1']['point'])} (F F1 {_n(best['f1_F']['point'])}){top}; with F beats "
                 "rarer in the enlarged training set, rebalancing decided whether F was detected at all.")  # fmt: skip
    from ecgaug.names import PHASE6_RELATIVE_GENERATORS

    p12 = _load(paths.results_for(PHASE6_RELATIVE_GENERATORS) / "summary.json")
    if p11 is not None and p12 is not None:
        conf = p12["families"]["confirmatory"]
        moves = " and of the ".join(f"{SHORT[k.split('/')[1]]} from {_n(p11['conditions'][k]['rare_macro_f1']['point'])} to {_n(c['rare_macro_f1']['point'])}"
                                    for k, c in p12["conditions"].items())  # fmt: skip
        rejected = sum(c["reject_holm"] for c in conf.values())
        verdict = ("neither change was detectable" if len(conf) == 2 else "no change was detectable") if rejected == 0 else f"{rejected} of {len(conf)} changes were detectable"
        text += (" Retraining the generators on those patients' beats, in the form the cnn sees them (Section~\\ref{sec:relative-generators}), took the "
                 f"test score of the {moves}; with {len(p12['records'])} test records {verdict} (Holm-adjusted $p$ "
                 + " and ".join(f"{c['p_holm']:.3f}" for c in conf.values()) + ").")  # fmt: skip
    if (rr := _record_rr_sentence(with_section=True)) is not None:
        text += " " + rr
        from ecgaug.names import PHASE6_RECORD_RR

        s13 = _load(paths.results_for(PHASE6_RECORD_RR) / "summary.json")
        check = _load(paths.results_for(PHASE6_RECORD_RR) / "design_check.json")
        if s13 and check and not any(c["reject_holm"] for c in s13["families"]["confirmatory"].values()) and len(check["s_heavy_records"]) == 1:
            h = check["s_heavy_records"][0]
            text += (" The cnn had the same information among its inputs but did not learn to use it from the one S-dominant recording among the "
                     f"training-side databases (SVDB {h['record']}), where a fixed rule from the literature did; learning it would take more such patients, "
                     "which is where augmentation that adds patients rather than beats could help.")
    if (ic := _icentia_sentence(with_section=True)) is not None:
        text += " " + ic
    for extra in (_diffusion_sentence(with_section=True), _af_sentence(with_section=True), _ensemble_sentence(with_section=True),
                  _fresh_sentence(with_section=True), _tuning_sentence(with_section=True)):
        if extra is not None:
            text += " " + extra
    return text + (" The evidence points to how much patients differ as the main obstacle here; on DS1's patients, normalizing the input for the "
                   "patient raised the test score where rebalancing the classes did not, and augmentation that adds patients rather than beats is "
                   "the form of augmentation it points to.")


def _record_rr_sentence(with_section: bool = False) -> str | None:
    """One or two sentences on P13 (RR over the recording's mean) from its summary, or None before its test."""
    from ecgaug.names import PHASE6_RECORD_RR

    s = _load(paths.results_for(PHASE6_RECORD_RR) / "summary.json")
    if s is None:
        return None
    conf = s["families"]["confirmatory"]
    up = sum(c["reject_holm"] and c["point"] > 0 for c in conf.values())
    down = sum(c["reject_holm"] and c["point"] < 0 for c in conf.values())
    verdict = (f"raised the cnn's test score detectably in {up} of {len(conf)} conditions" if up and not down else
               f"changed the cnn's test score detectably in {up + down} of {len(conf)} conditions ({up} up, {down} down)" if up or down else
               f"did not detectably change the cnn's test score in any of the {len(conf)} conditions")
    none_before, none_after = s["before"]["cnn/none"], s["conditions"]["cnn/none"]
    where = " (Section~\\ref{sec:record-rr})" if with_section else ""
    text = (f"Dividing the RR intervals by the recording's mean RR interval, as published methods do{where}, {verdict} (without augmentation "
            f"{_n(none_before['rare_macro_f1']['point'])} to {_n(none_after['rare_macro_f1']['point'])}, S F1 {_n(none_before['f1_S']['point'])} to "
            f"{_n(none_after['f1_S']['point'])})")
    rule = s.get("sveb_rule")
    if not rule:
        return text + "."
    key, best = max(rule.items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
    f1s = [v["f1_S"]["point"] for v in rule.values()]
    ses = [v["se_S"]["point"] for v in rule.values()]
    top = ", the highest test score of the study" if best["rare_macro_f1"]["point"] >= _best_test_score() else ""
    return text + (f"; with Huang et al.'s rule on top, which uses the same feature directly, S F1 reached {_n(min(f1s))} to {_n(max(f1s))} "
                   f"(sensitivity {_n(min(ses))} to {_n(max(ses))}) and the {SHORT[key.split('/')[1]]}'s rare-class macro-F1 "
                   f"{_n(best['rare_macro_f1']['point'])}{top} (descriptive, not tested).")


def _icentia_sentence(with_section: bool = False) -> str | None:
    """One sentence on P14 (many more training patients) from its summary, or None before its test."""
    from ecgaug.names import PHASE6_ICENTIA

    s = _load(paths.results_for(PHASE6_ICENTIA) / "summary.json")
    data = _load(paths.results_for(PHASE6_ICENTIA) / "data.json")
    if s is None or data is None:
        return None
    conf = s["families"]["confirmatory"]
    up = sum(c["reject_holm"] and c["point"] > 0 for c in conf.values())
    down = sum(c["reject_holm"] and c["point"] < 0 for c in conf.values())
    verdict = (f"raised the test score detectably in {up} of {len(conf)} conditions" if up and not down else
               f"changed the test score detectably in {up + down} of {len(conf)} conditions ({up} up, {down} down)" if up or down else
               f"did not detectably change the test score in any of the {len(conf)} conditions")
    where = " (Section~\\ref{sec:icentia})" if with_section else ""
    nb, na = s["before"]["cnn/none"], s["conditions"]["cnn/none"]
    key, best = max(s["conditions"].items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
    rkey, rbest = max(s["sveb_rule"].items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
    top = ", the highest test score of the study" if rbest["rare_macro_f1"]["point"] >= _best_test_score() else ""
    return (f"Training on {len(data['patients'])} more patients from Icentia11k{where} {verdict} (without augmentation "
            f"{_n(nb['rare_macro_f1']['point'])} to {_n(na['rare_macro_f1']['point'])}, S F1 {_n(nb['f1_S']['point'])} to {_n(na['f1_S']['point'])}; "
            f"best {SHORT[key.split('/')[1]]} {_n(best['rare_macro_f1']['point'])}); with Huang et al.'s rule on top, the {SHORT[rkey.split('/')[1]]} "
            f"reached {_n(rbest['rare_macro_f1']['point'])} (S F1 {_n(rbest['f1_S']['point'])}){top} (descriptive).")


def _diffusion_sentence(with_section: bool = False) -> str | None:
    """One sentence on P15 (the diffusion generator at the phase 3 budget) from its summary, or None before its test."""
    from ecgaug.names import PHASE6_DIFFUSION

    s = _load(paths.results_for(PHASE6_DIFFUSION) / "summary.json")
    if s is None:
        return None
    new, cvae = s["conditions"]["cnn/diffusion"], s["before"]["cnn/cvae"]
    c = s["families"]["confirmatory"]["cnn: diffusion vs cvae (P13)"]
    where = " (Section~\\ref{sec:diffusion})" if with_section else ""
    text = (f"A class-conditional diffusion model trained with the same step budget{where} reached {_n(new['rare_macro_f1']['point'])} (S F1 "
            f"{_n(new['f1_S']['point'])}, F F1 {_n(new['f1_F']['point'])}) against the cVAE's {_n(cvae['rare_macro_f1']['point'])} ({_p(c)}); its "
            "loss was still falling at the end of the budget and its samples matched the real beats' class shapes less closely")
    from ecgaug.names import PHASE6_DIFFUSION_LONG

    long = _load(paths.results_for(PHASE6_DIFFUSION_LONG) / "summary.json")
    if long is None:
        return text + ", so the budget, not the family, may be what limits it."
    lc = long["families"]["confirmatory"]["cnn: diffusion_long vs diffusion (P15)"]
    ln = long["conditions"]["cnn/diffusion_long"]
    early, late = _window_loss(PHASE6_DIFFUSION_LONG, 20000), _window_loss(PHASE6_DIFFUSION_LONG, 100000)
    fell = " its loss fell further, but" if early and late and max(late) < min(early) else ""
    change = "a detectable change" if lc["reject_holm"] else "no detectable change"
    gap = ", so more steps alone did not close the gap to the cVAE" if ln["rare_macro_f1"]["point"] < cvae["rare_macro_f1"]["point"] else ""
    where_long = " (Section~\\ref{sec:diffusion-long})" if with_section else ""
    return text + (f". Trained for five times as many steps{where_long}{fell} it reached {_n(ln['rare_macro_f1']['point'])} "
                   f"(S F1 {_n(ln['f1_S']['point'])}, F F1 {_n(ln['f1_F']['point'])}; {change} from the shorter training, {_p(lc)}){gap}.")


def _af_sentence(with_section: bool = False) -> str | None:
    """One sentence on P16 (the AF veto on Huang's stage) from its summary, or None before its test."""
    from ecgaug.names import PHASE6_AF_RULE

    s = _load(paths.results_for(PHASE6_AF_RULE) / "summary.json")
    if s is None:
        return None
    conf = s["families"]["confirmatory"]
    up = sum(c["reject_holm"] and c["point"] > 0 for c in conf.values())
    lows = [c["ci_low"] for c in conf.values()]
    key, best = max(s["with_af_veto"].items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
    before = s["rule_alone"][key]
    top = ", the highest test score of the study" if best["rare_macro_f1"]["point"] >= _best_test_score() else ""
    where = " (Section~\\ref{sec:af-rule})" if with_section else ""
    ci = "every 95\\% interval above zero" if min(lows) > 0 else "not every interval above zero"
    return (f"Undoing the rule's S calls inside atrial fibrillation, found by a detector that reads only beat times{where}, raised rare-class "
            f"macro-F1 in all {len(conf)} conditions (by {_n(min(c['point'] for c in conf.values()))} to {_n(max(c['point'] for c in conf.values()))}; "
            f"{ci}; detectable after Holm correction in {up}), by removing false S calls: the {SHORT[key.split('/')[1]]} went from "
            f"{_n(before['rare_macro_f1']['point'])} to {_n(best['rare_macro_f1']['point'])}{top} (S F1 {_n(before['f1_S']['point'])} to "
            f"{_n(best['f1_S']['point'])}, F F1 {_n(best['f1_F']['point'])}).")


def _ensemble_sentence(with_section: bool = False) -> str | None:
    """One sentence on P18 (ten runs averaged per condition) from its summary, or None before its test."""
    from ecgaug.names import PHASE6_SEED_ENSEMBLE

    s = _load(paths.results_for(PHASE6_SEED_ENSEMBLE) / "summary.json")
    if s is None:
        return None
    conf = s["families"]["confirmatory"]
    up = sum(c["reject_holm"] and c["point"] > 0 for c in conf.values())
    down = sum(c["reject_holm"] and c["point"] < 0 for c in conf.values())
    points = [c["point"] for c in conf.values()]
    cv, cp = s["ensemble"]["with_af_veto"]["cnn/cvae"], s["p16"]["with_af_veto"]["cnn/cvae"]
    top = ", the highest test score of the study" if cv["rare_macro_f1"]["point"] >= _best_test_score() else ""
    where = " (Section~\\ref{sec:seed-ensemble})" if with_section else ""
    found = [f"raised it in {up}" if up else "", f"lowered it in {down}" if down else ""]
    verdict = " and ".join(f for f in found if f) + f" of {len(conf)} conditions" if up or down else f"no detectable change in any of {len(conf)} conditions"
    cc = conf["cnn: cvae, ensemble vs P16"]
    return (f"Averaging the class probabilities of ten independently trained runs per condition{where} changed rare-class macro-F1 by "
            f"{_s(min(points))} to {_s(max(points))} against P16's single runs (after Holm correction: {verdict}); the cVAE's ensemble reached "
            f"{_n(cv['rare_macro_f1']['point'])}{top} (S F1 {_n(cv['f1_S']['point'])}, F F1 {_n(cv['f1_F']['point'])}), against "
            f"{_n(cp['rare_macro_f1']['point'])} for its single runs ({'a detectable' if cc['reject_holm'] else 'not a detectable'} change, {_p(cc)}).")


def _fresh_sentence(with_section: bool = False) -> str | None:
    """One sentence on P19 (P18's pipeline on 1,000 unused Icentia11k patients), or None before it ran."""
    from ecgaug.names import PHASE6_FRESH_PATIENTS

    s = _load(paths.results_for(PHASE6_FRESH_PATIENTS) / "summary.json")
    if s is None:
        return None
    cv, alone = s["ensemble"]["with_af_veto"]["cnn/cvae"], s["ensemble"]["cnn_alone"]["cnn/cvae"]
    where = " (Section~\\ref{sec:fresh-patients})" if with_section else ""
    return (f"On {s['patients']:,} Icentia11k patients that no analysis had used{where}, the same pipeline, unchanged, reached S F1 "
            f"{_ci(cv['f1_S'])} and V F1 {_n(cv['f1_V']['point'])} for the cVAE's ensemble with the stage and the veto, and S F1 "
            f"{_n(alone['f1_S']['point'])} for the ensemble alone.")


def _tuning_sentence(with_section: bool = False) -> str | None:
    """One sentence on P20 (the stage and the veto chosen on 400 other Icentia11k patients), or None before it ran."""
    from ecgaug.names import PHASE6_RULE_TUNING

    folder = paths.results_for(PHASE6_RULE_TUNING)
    s, test = _load(folder / "summary.json"), _load(folder / "test_summary.json")
    if s is None:
        return None
    cond, conf = s["conditions"], s["families"]["confirmatory"]["cnn: cvae, chosen vs P19's setting"]
    where = " (Section~\\ref{sec:rule-tuning})" if with_section else ""
    text = (f"Choosing the stage and the veto on 400 other Icentia11k patients{where} picked \\emph{{{s['chosen']['cvae']['label']}}} for the cVAE, "
            f"which took its S F1 on P19's patients from {_n(cond['p19_setting']['cnn/cvae']['f1_S']['point'])} to "
            f"{_n(cond['chosen']['cnn/cvae']['f1_S']['point'])} ({_changed(conf)}, {_p(conf)})")  # fmt: skip
    if test is not None:
        tc = test["conditions"]
        text += (f"; on the test records its rare-class macro-F1 went from {_n(tc['p18_setting']['cnn/cvae']['rare_macro_f1']['point'])} with "
                 f"P18's setting to {_n(tc['chosen']['cnn/cvae']['rare_macro_f1']['point'])}")
    return text + "."


def _best_test_score() -> float:
    """The highest test rare-class macro-F1 of any condition in any summary written so far (final evaluation and post hoc,
    P13's descriptive SVEB-rule results included)."""
    from ecgaug.names import POST_HOC

    best = 0.0
    for experiment in (FINAL_EVAL, *POST_HOC):
        tuned = _load(paths.results_for(experiment) / "test_summary.json") or {}  # P20's test read: one block per setting
        for block in tuned.get("conditions", {}).values():
            best = max([best, *(float(e["rare_macro_f1"]["point"]) for e in block.values() if "rare_macro_f1" in e)])
        s = _load(paths.results_for(experiment) / "summary.json") or {}
        ensembles = [e for block in s.get("ensemble", {}).values() for e in block.values()]  # P18's ensembles, stage and veto included
        for entry in [*s.get("conditions", {}).values(), *s.get("sveb_rule", {}).values(), *s.get("with_af_veto", {}).values(), *ensembles]:
            value = entry.get("rare_macro_f1") if isinstance(entry, dict) else None
            if isinstance(value, dict) and "point" in value:
                best = max(best, float(value["point"]))
    return best


def discussion() -> str:
    """The discussion, every number from the result files; claims follow the numbers (nothing is asserted that a file does not show)."""
    from ecgaug.names import (PHASE6_AUDIT_CHECK, PHASE6_AUGMENTATION_MECHANISM, PHASE6_EXTERNAL_VALIDATION, PHASE6_EXTRA_SEEDS,
                              PHASE6_PATIENT_TEMPLATE, PHASE6_RELATIVE_CNN, PHASE6_RR_BASELINE)  # fmt: skip

    final = _load(paths.results_for(FINAL_EVAL) / "summary.json")
    if final is None:
        return _missing("discussion: after the final evaluation") + "\n"
    conf = final["families"]["confirmatory"]
    helped = [k for k, c in conf.items() if c["reject_holm"] and c["point"] > 0]
    support = []
    if (s := _load(paths.results_for(PHASE6_EXTRA_SEEDS) / "summary.json")) and not s["changed"]:
        support.append("with five seeds")
    if (s := _load(paths.results_for(PHASE6_RR_BASELINE) / "summary.json")) and not any(c["reject_holm"] and c["point"] > 0 for c in s["families"]["confirmatory"].values()):
        support.append("with a patient-baseline RR feature")
    p7 = _load(paths.results_for(PHASE6_RELATIVE_CNN) / "summary.json")
    if p7 and not any(c["reject_holm"] and c["point"] > 0 for c in p7["families"]["confirmatory"].values()):
        support.append("for a cnn made stronger on validation by patient-relative beats")
    _, p9 = _p9()
    p9_helped = None if p9 is None else any(c["reject_holm"] and c["point"] > 0 for c in p9["families"]["confirmatory"].values())
    if p9_helped is False:
        support.append("with patient-transfer augmentation")
    ext = [_load(paths.results_for(PHASE6_EXTERNAL_VALIDATION) / f"{db}_summary.json") for db in DATABASE]
    ext_up = sum(c["reject_holm"] and c["point"] > 0 for e in ext if e for c in e["confirmatory"].values())
    ext_down = sum(c["reject_holm"] and c["point"] < 0 for e in ext if e for c in e["confirmatory"].values())
    ext_n = sum(len(e["confirmatory"]) for e in ext if e)
    paras = []
    text = ("\\paragraph{What the study shows.} Under the inter-patient protocol, generative augmentation "
            + ("improved" if helped else "did not detectably improve") + " the detection of rare classes over no augmentation in the "
            f"pre-registered test ({len(helped)} of {len(conf)} confirmatory comparisons rejected in its favour)")
    if support:
        text += ", and the same held " + ", ".join(support[:-1]) + (" and " if len(support) > 1 else "") + support[-1]
    if ext_n:
        text += (f"; on two external databases it was better than no augmentation in {ext_up} and worse in {ext_down} of {ext_n} confirmatory "
                 "comparisons")
    paras.append(text + ". Whether a synthetic beat looks realistic and whether it helps a classifier on new patients are different questions, and this study answers the second.")
    diag = _load(paths.results_for(PHASE6_PATIENT_TEMPLATE) / "diagnostic.json")
    mech_gap = _load(paths.results_for(PHASE6_AUGMENTATION_MECHANISM) / "gap.json")
    attribution = _load(paths.results_for(PHASE6_AUGMENTATION_MECHANISM) / "attribution.json")
    why = []
    if diag is not None and "cnn" in diag["networks"] and "v1" in diag["networks"]["cnn"]:
        g = diag["networks"]["cnn"]["v1"]
        why.append(f"the frozen cnn without augmentation reached S PR-AUC {_n(g['train']['pr_auc']['S'])} on the training patients and "
                   f"{_n(g['val']['pr_auc']['S'])} on the validation patients")
    if mech_gap is not None:
        cases = [(arch, c, conds[c], conds["none"]) for arch, conds in mech_gap["networks"].items() for c in ("cvae", "wgan_gp") if c in conds]
        if cases:
            narrowed = sum(g["gap"]["S"] < n["gap"]["S"] for *_, g, n in cases)
            detail = ", ".join(f"{arch} {SHORT[c]}: validation {_n(n['val']['pr_auc']['S'])} to {_n(g['val']['pr_auc']['S'])}, training "
                               f"{_n(n['train']['pr_auc']['S'])} to {_n(g['train']['pr_auc']['S'])}" for arch, c, g, n in cases)
            why.append(f"generative augmentation narrowed the training-versus-validation gap in S PR-AUC in {narrowed} of {len(cases)} cases "
                       f"({detail}), but none gave a detectably better test result")
    if attribution is not None:
        s = attribution["classes"]["S"]
        real = s["real_training"]
        syn = [e["shares"].get(real["top_record"], 0.0) for c in ("cvae", "wgan_gp") for e in s.get(f"{c}_bank", {}).values()]
        if syn:
            mean_syn = float(np.mean(syn))
            relation = "at least as concentrated on" if mean_syn >= real["top_share"] - 0.05 else "less concentrated on"
            why.append(f"synthetic S beats were {relation} record {real['top_record']} as the real ones ({_n(mean_syn, 2)} against {_n(real['top_share'], 2)})")
    if why:
        from ecgaug.names import PHASE6_RELATIVE_GENERATORS

        tested = _load(paths.results_for(PHASE6_RELATIVE_GENERATORS) / "summary.json") is not None
        paras.append("\\paragraph{Why.} The analyses of Sections~\\ref{sec:patient-template} and~\\ref{sec:mechanism} point to the patients rather than the "
                     "beats: " + "; ".join(why) + ". With so few training patients, a generator can add beats of those patients but not new ones; generators trained "
                     "on more patients, or conditioned on the patient, are what this points to."
                     + (" Section~\\ref{sec:relative-generators} tests the first of these; its result is in the next paragraphs." if tested else ""))  # fmt: skip
    if p9_helped:
        paras.append("\\paragraph{Augmentation that adds patients.} Moving the deviations of rare beats onto other patients' templates improved the "
                     "primary metric over no augmentation in the registered comparison (Section~\\ref{sec:patient-transfer}), where class-conditional "
                     "generation alone did not: what new patients need is the arrhythmia seen against many patients' normal beats.")
    if (pv := patient_variation_paragraph()) is not None:
        paras.append(pv)
    dom = final["conditions"]["cnn/none"]["rare_macro_f1"]["dominant_record"]
    train_f = ""
    if attribution is not None and "F" in attribution["classes"]:
        f = attribution["classes"]["F"]["real_training"]
        train_f = f"; record {f['top_record']} holds {_n(f['top_share'], 2)} of the training F beats"
    paras.append(f"\\paragraph{{Absolute scores.}} The test scores are low. Record {dom['S']['record']} holds {_n(dom['S']['share_of_class'], 2)} of the test S "
                 f"beats and record {dom['F']['record']} {_n(dom['F']['share_of_class'], 2)} of the F beats, so the rare-class metrics largely measure two "
                 f"patients{train_f}. A method that is good on average and poor on these patients scores low here, which is a property of the benchmark "
                 "as much as of the method.")  # fmt: skip
    audit = _load(paths.results_for(PHASE6_AUDIT_CHECK) / "summary.json")
    if audit is not None:
        paras.append(f"\\paragraph{{Relation to earlier reports.}} With beats split at random, the same pipeline reached {_n(audit['audit_split0']['none']['mean'])} "
                     f"rare-class macro-F1 without augmentation, against {_n(audit['inter_patient']['none']['point'])} inter-patient (Section~\\ref{{sec:posthoc}}). "
                     "Gains from augmentation reported under such splits may reflect, at least in part, how well a model recognises patients it has seen.")  # fmt: skip
    paras.append(outlook().rstrip("\n"))
    return "\n\n".join(paras) + "\n"


def conclusion() -> str:
    """Two or three sentences, each conditional on the result files."""
    from ecgaug.names import PHASE6_AUGMENTATION_MECHANISM, PHASE6_EXTRA_SEEDS, PHASE6_PATIENT_TRANSFER, PHASE6_RELATIVE_CNN, PHASE6_RR_BASELINE

    final = _load(paths.results_for(FINAL_EVAL) / "summary.json")
    if final is None:
        return _missing("conclusion: after the final evaluation") + "\n"
    def helps(summary: dict | None) -> bool | None:
        return None if summary is None else any(c["reject_holm"] and c["point"] > 0 for c in summary["families"]["confirmatory"].values())
    primary = helps(final)
    follow = [helps(_load(paths.results_for(e) / "summary.json")) for e in (PHASE6_EXTRA_SEEDS, PHASE6_RR_BASELINE, PHASE6_RELATIVE_CNN, PHASE6_PATIENT_TRANSFER)]
    follow = [f for f in follow if f is not None]
    text = ("Under a strict inter-patient protocol with rules fixed in advance, and with the small models and few runs this study could afford, "
            "class-conditional generative augmentation " + ("improved" if primary else "did not detectably improve") + " the detection of rare "
            "arrhythmias on MIT-BIH")
    if follow:
        text += (", and the follow-up analyses " + ("did not change that answer" if all(f == primary for f in follow) else "did not all agree with it")) + "."
    else:
        text += "."
    attribution = _load(paths.results_for(PHASE6_AUGMENTATION_MECHANISM) / "attribution.json")
    if attribution is not None:
        s = attribution["classes"]["S"]
        real = s["real_training"]
        syn = [e["shares"].get(real["top_record"], 0.0) for c in ("cvae", "wgan_gp") for e in s.get(f"{c}_bank", {}).values()]
        if syn and float(np.mean(syn)) >= real["top_share"] - 0.05:
            text += " The evidence points to the few training patients, which the synthetic beats reproduce, rather than to a limit of augmentation itself."
    p9 = helps(_load(paths.results_for(PHASE6_PATIENT_TRANSFER) / "summary.json"))
    if p9 is True:
        text += " Placing the deviations of rare beats on other patients' normal beats, a post-hoc variant, did improve detection over no augmentation."
    elif p9 is False:
        text += " Placing the deviations of rare beats on other patients' normal beats did not detectably improve it at this scale either."
    from ecgaug.names import PHASE6_MORE_PATIENTS

    p11 = _load(paths.results_for(PHASE6_MORE_PATIENTS) / "summary.json")
    if p11 is not None:
        more = p11["families"]["confirmatory"]
        up = any(c["reject_holm"] and c["point"] > 0 for c in more.values())
        best_key, best = max(p11["conditions"].items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
        top = ", the highest test score of the study" if best["rare_macro_f1"]["point"] >= _best_test_score() else ""
        text += (" Training on more patients " + ("raised" if up else "did not detectably raise") + " the score of the best configuration without "
                 f"augmentation; with more patients, {SHORT[best_key.split('/')[1]]} reached {_n(best['rare_macro_f1']['point'])}{top}.")  # fmt: skip
        from ecgaug.names import PHASE6_RELATIVE_GENERATORS

        p12 = _load(paths.results_for(PHASE6_RELATIVE_GENERATORS) / "summary.json")
        if p12 is not None:
            better = any(c["reject_holm"] and c["point"] > 0 for c in p12["families"]["confirmatory"].values())
            key = max(p12["conditions"], key=lambda k: p12["conditions"][k]["rare_macro_f1"]["point"])
            text += (" Generators retrained on those patients' beats, in the form the classifier sees them, " + ("improved" if better else "did not detectably improve")
                     + f" on the same generators trained on DS1 alone (the {SHORT[key.split('/')[1]]} went from "
                     f"{_n(p11['conditions'][key]['rare_macro_f1']['point'])} to {_n(p12['conditions'][key]['rare_macro_f1']['point'])}).")  # fmt: skip
        if (rr := _record_rr_sentence()) is not None:
            text += " " + rr
        if (ic := _icentia_sentence()) is not None:
            text += " " + ic
        for extra in (_diffusion_sentence(), _af_sentence(), _ensemble_sentence(), _fresh_sentence(), _tuning_sentence()):
            if extra is not None:
                text += " " + extra
    p7 = _load(paths.results_for(PHASE6_RELATIVE_CNN) / "summary.json")
    if p7 is not None and final is not None:
        text += (f" Changing the input alone raised the test score from {_n(final['conditions']['cnn/none']['rare_macro_f1']['point'])} to "
                 f"{_n(p7['conditions']['cnn/none']['rare_macro_f1']['point'])}, so the results are far from a ceiling.")
    return text + (" Larger models, more runs and more training patients are needed to establish how much generative augmentation can add, "
                   "and gains measured with random beat-level splits need to be re-measured across patients.\n")


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
    from ecgaug.names import PHASE6_PATIENT_TEMPLATE

    v6 = _load(paths.results_for(PHASE6_PATIENT_TEMPLATE) / "adoption.json")
    if v6 is not None:
        moves = " and ".join(f"{'raised' if v['difference'] > 0 else 'lowered'} the {arch}'s validation S F1 by {_n(abs(v['difference']))}"
                          for arch, v in v6["per_architecture"].items())
        text += (f" Giving each beat relative to the patient's own median beat {moves}; "
                 + ("by the pre-registered rule the full grid was rerun." if v6["adopted"] else "by the pre-registered rule the full grid was not rerun."))  # fmt: skip
    _, p9 = _p9()
    if p9 is not None:
        helped9 = any(c["reject_holm"] and c["point"] > 0 for c in p9["families"]["confirmatory"].values())
        text += (" Moving the deviations of rare beats onto other patients' templates (patient transfer) "
                 + ("improved" if helped9 else "did not detectably improve") + " on no augmentation.")
    from ecgaug.names import PHASE6_RELATIVE_CNN

    p7 = _load(paths.results_for(PHASE6_RELATIVE_CNN) / "summary.json")
    final = _load(paths.results_for(FINAL_EVAL) / "summary.json")
    if p7 is not None and final is not None:
        helped = any(c["reject_holm"] and c["point"] > 0 for c in p7["families"]["confirmatory"].values())
        text += (f" Repeating the comparison for the cnn alone with patient-relative beats (a choice made on validation) raised its test rare-class "
                 f"macro-F1 without augmentation from {_n(final['conditions']['cnn/none']['rare_macro_f1']['point'])} to "
                 f"{_n(p7['conditions']['cnn/none']['rare_macro_f1']['point'])}, and generative augmentation "
                 + ("improved" if helped else "still did not improve") + " on it.")  # fmt: skip
    from ecgaug.names import PHASE6_MORE_PATIENTS, PHASE6_RELATIVE_GENERATORS

    p11 = _load(paths.results_for(PHASE6_MORE_PATIENTS) / "summary.json")
    p12 = _load(paths.results_for(PHASE6_RELATIVE_GENERATORS) / "summary.json")
    if p11 is not None:
        best_key, best = max(p11["conditions"].items(), key=lambda kv: kv[1]["rare_macro_f1"]["point"])
        name = best_key.split("/")[1]
        top = ", the study's highest test score" if best["rare_macro_f1"]["point"] >= _best_test_score() else ""
        vs_none = p11["families"]["secondary"].get(f"cnn: {name} vs none (more patients)")
        text += (f" With the INCART and SVDB patients added to training, {SHORT[name]} reached {_n(best['rare_macro_f1']['point'])}{top}"
                 + (f" (against no augmentation, {_p(vs_none)})" if vs_none is not None else ""))
        if p12 is not None:
            key = max(p12["conditions"], key=lambda k: p12["conditions"][k]["rare_macro_f1"]["point"])
            family = key.split("/")[1]
            c = p12["families"]["confirmatory"][f"cnn: {family}, P12 vs P11 generators"]
            text += (f", and generators retrained on those patients' beats in the form the classifier sees them "
                     f"{'raised' if c['point'] > 0 else 'lowered'} the {SHORT[family]}'s score from {_n(p11['conditions'][key]['rare_macro_f1']['point'])} "
                     f"to {_n(p12['conditions'][key]['rare_macro_f1']['point'])} ({_p(c)})")  # fmt: skip
        text += "."
    if (rr := _record_rr_sentence()) is not None:
        text += " " + rr
    if (ic := _icentia_sentence()) is not None:
        text += " " + ic
    for extra in (_diffusion_sentence(), _af_sentence(), _ensemble_sentence(), _fresh_sentence(), _tuning_sentence()):
        if extra is not None:
            text += " " + extra
    sizes = _model_sizes()
    if sizes:
        text += (" These answers hold for small models (" + " and ".join(f"{_n(v / 1000, 0)}" for v in sizes.values())
                 + " thousand parameters) trained a few times per condition; larger models, more runs and more patients are needed to establish how far augmentation can go.")
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
    files["results_patient_template"] = posthoc_patient_template()
    files["results_relative_cnn"] = posthoc_relative_cnn()
    files["results_mechanism"] = posthoc_mechanism()
    files["results_patient_transfer"] = posthoc_patient_transfer()
    files["results_patient_adaptation"] = posthoc_patient_adaptation()
    files["results_more_patients"] = posthoc_more_patients()
    files["results_relative_generators"] = posthoc_relative_generators()
    files["results_record_rr"] = posthoc_record_rr()
    files["results_icentia"] = posthoc_icentia()
    files["results_diffusion"] = posthoc_diffusion()
    files["results_af_rule"] = posthoc_af_rule()
    files["results_diffusion_long"] = posthoc_diffusion_long()
    files["results_seed_ensemble"] = posthoc_seed_ensemble()
    files["results_fresh_patients"] = posthoc_fresh_patients()
    files["results_rule_tuning"] = posthoc_rule_tuning()
    files["results_physics_prior"] = posthoc_physics_prior()
    files["results_bigger_classifier"] = posthoc_bigger_classifier()
    files["results_physics_fitted"] = posthoc_physics_fitted()
    files["table_posthoc"] = posthoc_table()
    files["outlook"] = outlook(heading=False)
    files["discussion"] = discussion()
    files["conclusion"] = conclusion()
    files["table_hypotheses"] = hypotheses_table(verdicts)
    files["abstract_results"] = abstract_sentence(summary, verdicts, audit) + "\n"
    return files
