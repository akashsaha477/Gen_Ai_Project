"""The course report's own short texts (``paper/report.tex``): a summary, an example of the system's output, the follow-up
analyses told by theme, and a short conclusion. Every number comes from a result file, through the same helpers as the
paper (:mod:`ecgaug.paper_text`); nothing is typed by hand (docs/RULES.md rule 5). ``python -m ecgaug.paper`` writes them
as ``paper/generated/report_*.tex``. The paper keeps the full account of every analysis; the report tells it in brief.
"""

from __future__ import annotations

import numpy as np

from ecgaug import literature, paths, workflow
from ecgaug import paper_text as pt
from ecgaug.names import FINAL_EVAL, PHASE5_GATE, POST_HOC

REBALANCING = ("class_weighted", "random_oversampling", "smote")


def _load(experiment: str, name: str = "summary.json") -> dict | None:
    return pt._load(paths.results_for(experiment) / name)


def _first_lower(text: str) -> str:
    return text[:1].lower() + text[1:]


def _tally(families: list[dict | None]) -> tuple[int, int, int]:
    """Holm-tested comparisons pooled over families: (detectably higher, detectably lower, total)."""
    items = [c for f in families if f for c in f.values()]
    return (sum(bool(c["reject_holm"]) and c["point"] > 0 for c in items), sum(bool(c["reject_holm"]) and c["point"] < 0 for c in items), len(items))


def _tally_text(up: int, down: int) -> str:
    if not up and not down:
        return "none changed detectably"
    found = [f"{up} rose" if up else "", f"{down} fell" if down else ""]
    return " and ".join(f for f in found if f) + " detectably"


def posthoc_reads() -> int:
    """Logged reads of the test records after the final evaluation (``docs/TEST_ACCESS_LOG.md``, kind ``posthoc``)."""
    log = paths.docs_dir() / "TEST_ACCESS_LOG.md"
    if not log.is_file():
        return 0
    return sum(1 for line in log.read_text().splitlines() if line.startswith("| 20") and "| posthoc |" in line)


def _audit_range(audit: dict) -> tuple[float, float, float]:
    """P4: the cnn without augmentation with beats split at random (lowest and highest over the audit's splits) and across patients."""
    nones = [audit["audit_split0"]["none"]["mean"], *(c["intra"]["mean"] for c in audit["checks"] if c["condition"] == "none")]
    return min(nones), max(nones), audit["inter_patient"]["none"]["point"]


def _zhu() -> dict:
    return next(r for r in literature.rows() if r["key"] == "zhu2022")


def _near(ours: float, theirs: float) -> str:
    return "close to" if abs(ours - theirs) < 0.02 else ("above" if ours > theirs else "below")


def summary() -> str:
    """The report's summary: the system, the question, the pre-registered answer, what moved the score and how far to trust it."""
    final = _load(FINAL_EVAL)
    if final is None:
        return pt._missing("summary: after the final evaluation") + "\n"
    conf = final["families"]["confirmatory"]
    favour = sum(bool(c["reject_holm"]) and c["point"] > 0 for c in conf.values())
    best = max(e["rare_macro_f1"]["point"] for e in final["conditions"].values())
    none = final["conditions"]["cnn/none"]["rare_macro_f1"]["point"]
    text = ("We built a system that takes one ECG beat and returns its class, normal (N), supraventricular ectopic (S), ventricular ectopic "
            "(V) or fusion (F), a saliency map of the parts of the beat behind the prediction, and a short note for a clinician that is checked "
            "against the model output. With it we asked whether synthetic beats of the rare classes from a conditional VAE or a WGAN-GP help a "
            "model find them in patients it has never seen. Every choice was fixed on validation patients by rules written down in advance, and "
            "the 22 test patients of the MIT-BIH Arrhythmia Database were read once. Generative augmentation was detectably better than no "
            f"augmentation in {favour} of {len(conf)} pre-registered comparisons, and the best pre-registered model reached a rare-class macro-F1 "
            f"(the mean of S and F F1) of {pt._n(best)}.")  # fmt: skip
    audit = _load("phase6_audit_check")
    if audit is not None:
        lo, hi, inter = _audit_range(audit)
        text += (f" The cnn without augmentation, {pt._n(inter)} on new patients, scored {pt._n(lo)} to {pt._n(hi)} with the same patients in "
                 "training and test.")
    p7, p18, p19 = _load("phase6_relative_cnn"), _load("phase6_seed_ensemble"), _load("phase6_fresh_patients")
    if p7 is not None and p18 is not None:
        ens = p18["ensemble"]["with_af_veto"]["cnn/cvae"]
        zhu = _zhu()["rare_macro_f1"]
        text += (" Follow-up analyses, each registered before it ran, found what moves the score: each beat seen relative to the patient's own "
                 f"normal beat ({pt._n(none)} to {pt._n(p7['conditions']['cnn/none']['rare_macro_f1']['point'])}), prematurity judged against the "
                 "recording's mean heart period, no S calls during atrial fibrillation, and ten runs averaged; together they reached "
                 f"{pt._n(ens['rare_macro_f1']['point'])} on the test records, {_near(ens['rare_macro_f1']['point'], zhu)} the best published "
                 f"inter-patient result we could check ({pt._n(zhu)}).")  # fmt: skip
        if p19 is not None:
            fresh = p19["ensemble"]["with_af_veto"]["cnn/cvae"]["f1_S"]["point"]
            text += (f" That score is optimistic after repeated reads of the test records: on {p19['patients']:,} unused patients the pipeline's "
                     f"S F1 was {pt._n(fresh)}, against {pt._n(ens['f1_S']['point'])} on the test records.")
    return text + (" Patient variation, not class rarity, limits these models; more training patients with rare beats, larger models and more "
                   "runs are needed to establish how much augmentation can add.\n")


EXAMPLE_CLASSES = ("S", "V")


def _example_box(case: dict, note: dict, variant: str) -> str:
    p = case["payload"]
    rows = [("predicted class", p["predicted_class"]),
            ("probabilities", ", ".join(f"{k} {v:.2f}" for k, v in p["probabilities"].items())),
            ("low confidence", "yes" if p["low_confidence"] else "no"),
            ("saliency share", ", ".join(f"{s['name']} {s['mass']:.2f}" for s in p["top_segments"]))]  # fmt: skip
    body = " \\\\\n".join(f"{a} & {pt._tex(str(b))}" for a, b in rows)
    return ("\\fbox{\\begin{minipage}[t]{0.46\\linewidth}\\footnotesize\n"
            f"\\textbf{{Input}} (validation beat labelled {case['true_class']} by the cardiologists)\\par\\smallskip\n"
            "\\begin{tabular}{@{}p{0.3\\linewidth}p{0.64\\linewidth}@{}}\n" + body + "\n\\end{tabular}\\par\\medskip\n"
            f"\\textbf{{Note}} (generator {variant}; {note['unsupported_claims']} unsupported claims)\\par\\smallskip\n"
            f"\\emph{{{pt._tex(note['note'])}}}\n\\end{{minipage}}}}")  # fmt: skip


def example_output() -> str:
    """The system's output for two validation beats: for S and for V, the first case (in the order of the phase 5 cases) that the model
    classified correctly without the low-confidence flag; the object the chosen note generator received and the note it wrote."""
    gate = workflow.manifest().get(PHASE5_GATE)
    cases = _load("phase5_clinical_notes", "cases.json")
    if gate is None or cases is None:
        return pt._missing("example output: after the phase 5 gate") + "\n"
    variant = gate["notes_variant"]
    notes = {n["case"]: n for n in (_load("phase5_clinical_notes", f"{variant}_notes.json") or [])}
    picks = [next((c for c in cases["cases"] if c["payload"]["predicted_class"] == c["true_class"] == cls and not c["payload"]["low_confidence"]
                   and c["payload"]["record_anon_id"] in notes), None) for cls in EXAMPLE_CLASSES]  # fmt: skip
    if any(p is None for p in picks):
        return pt._missing("example output: a correctly classified confident S and V case with a note") + "\n"
    boxes = "\\hfill\n".join(_example_box(c, notes[c["payload"]["record_anon_id"]], variant) for c in picks)
    return ("\\begin{figure}[htbp]\\centering\n" + boxes + "\n"
            "\\caption{The system's output for two validation beats, the first S and the first V beat of the note cases that the model classified "
            "correctly with confidence. The note generator receives only the input object, never the signal; the fact checker compares every claim "
            "of the note with it, and a template note replaces any note that fails. Saliency shares are the fractions of the saliency map inside the "
            "P, QRS, ST and T windows of Figure~\\ref{fig:medical}; the two largest are passed on.}\\label{fig:example}\n\\end{figure}\n")  # fmt: skip


def posthoc_story() -> str:
    """The follow-up analyses (P1-P23) by theme rather than in order, with their key numbers; the post-hoc table has the rest."""
    final = _load(FINAL_EVAL)
    if final is None:
        return pt._missing("follow-up analyses: after the final evaluation") + "\n"
    paras = []
    p19 = _load("phase6_fresh_patients")
    unused = f"{p19['patients']:,} Icentia11k patients that no analysis had used" if p19 else "patients that no analysis had used"
    paras.append(f"After the final evaluation, {len(POST_HOC)} follow-up analyses (P1 to P{len(POST_HOC)}) asked why augmentation did not help and "
                 "what would. Each was added to the pre-registration (versions 4 to 21) before it ran, and every read of the test records is logged: "
                 f"{posthoc_reads()} reads after the final evaluation. Later scores on the test records are therefore optimistic and do not replace "
                 f"the pre-registered results, so two analyses scored the best pipeline on {unused}. Table~\\ref{{tab:posthoc}} lists every analysis "
                 "with its numbers, and Figure~\\ref{fig:progress} follows the test score through the main changes.")  # fmt: skip

    seeds, audit = _load("phase6_extra_seeds"), _load("phase6_audit_check")
    ext = [_load("phase6_external_validation", f"{db}_summary.json") for db in ("incartdb", "svdb")]
    if seeds and audit and all(ext):
        up, down, total = _tally([e["confirmatory"] for e in ext])
        lo, hi, inter = _audit_range(audit)
        changed = len(seeds["changed"])
        paras.append("\\paragraph{The pre-registered answer holds.} With five seeds instead of three, "
                     + ("no pre-registered verdict changed" if not changed else f"{changed} pre-registered verdicts changed") + " (P3). "
                     "On two other databases, St Petersburg INCART and the MIT-BIH Supraventricular Arrhythmia Database, generative augmentation was "
                     f"detectably better than no augmentation in {up} and worse in {down} of {total} comparisons (P2). With beats split at random, so "
                     f"that the same patients are in training and test, the same cnn reached {pt._n(lo)} to {pt._n(hi)} against {pt._n(inter)} across "
                     "patients (P4): much of what such splits measure is how well a model recognises patients it has seen.")  # fmt: skip

    p7, attr = _load("phase6_relative_cnn"), _load("phase6_augmentation_mechanism", "attribution.json")
    _, p9 = pt._p9()
    p10, p11, p14 = _load("phase6_patient_adaptation"), _load("phase6_more_patients"), _load("phase6_icentia")
    p14_data = _load("phase6_icentia", "data.json")
    if p7 and attr and p9 and p10 and p11 and p14 and p14_data:
        base = final["conditions"]["cnn/none"]["rare_macro_f1"]["point"]
        rel = p7["conditions"]["cnn/none"]["rare_macro_f1"]["point"]
        rebal = max(final["conditions"][f"cnn/{c}"]["rare_macro_f1"]["point"] - base for c in REBALANCING)
        s_cls = attr["classes"]["S"]
        top, real = s_cls["real_training"]["top_record"], s_cls["real_training"]["top_share"]
        syn = float(np.mean([e["shares"].get(top, 0.0) for c in ("cvae", "wgan_gp") for e in s_cls.get(f"{c}_bank", {}).values()]))
        up, down, total = _tally([p9["families"]["confirmatory"], p10["families"]["adaptation"], p11["families"]["confirmatory"], p14["families"]["confirmatory"]])
        same_patients = syn >= real - 0.05
        paras.append("\\paragraph{The patient limits the model more than the class balance.} Giving the cnn each beat relative to the patient's own "
                     f"median beat, with nothing else changed, raised its test score without augmentation from {pt._n(base)} to {pt._n(rel)} (P7)"
                     + (", more than any rebalancing method did" if rel - base > rebal else "") + ". The synthetic S beats came from the same few "
                     f"patients as the real ones: {pt._n(syn, 2)} of them were nearest to a beat of record {top}, against {pt._n(real, 2)} of the real "
                     "training S beats (P8)" + (", so the generators added beats but not patients" if same_patients else "") + ". Methods that act on the patient were then tested: moving "
                     "the deviation of each rare beat onto other patients' normal beats (P9), normalising each patient inside the network (P10), and "
                     f"adding the INCART and SVDB patients (P11) or {len(p14_data['patients'])} Icentia11k patients (P14) to training. Of their "
                     f"{total} test comparisons, {_tally_text(up, down)}" + ("; at this scale, adding patients did not by itself raise the score." if not up else "."))  # fmt: skip

    rr, af = pt._record_rr_sentence(), pt._af_sentence()
    if rr and af:
        paras.append("\\paragraph{Timing identifies S beats, as it does for a cardiologist.} An S beat is recognised mainly by coming early for the "
                     f"patient's own rhythm (Section~\\ref{{sec:medical}}). In P13, {_first_lower(rr)} In P16, {_first_lower(af)}")  # fmt: skip

    gens = {"P15": _load("phase6_diffusion"), "P17": _load("phase6_diffusion_long"), "P21": _load("phase6_physics_prior", "test_summary.json"),
            "P23": _load("phase6_physics_fitted", "test_summary.json")}  # fmt: skip
    p22, p22t = _load("phase6_bigger_classifier"), _load("phase6_bigger_classifier", "test_summary.json")
    if all(gens.values()) and p22 and p22t:
        from ecgaug import architecture as arch

        sizes = {r["name"]: r["parameters"] for r in arch.inventory()}
        ratio = sizes["ResNet1D"] / sizes["Shallow CNN (the cnn)"]
        up, down, total = _tally([g["families"]["confirmatory"] for g in gens.values()])
        s_up, _, s_n = _tally([p22["families"]["secondary"]])
        r_up, r_down, r_n = _tally([p22t["families"]["confirmatory"]])
        paras.append("\\paragraph{Better generators and a larger classifier did not change the answer.} A class-conditional diffusion model (P15), "
                     "the same model trained five times longer (P17), the physics loss of SimGANs, which scores generated beats with an ECG simulator "
                     "(P21), and a physics loss fitted to each generated beat (P23) were each compared on the test records with the generator they "
                     f"replace or with no augmentation; of these {total} comparisons, {_tally_text(up, down)}. A ResNet with {ratio:.1f} times "
                     f"the cnn's parameters (P22) raised supraventricular F1 on the unused patients in {s_up} of {s_n} comparisons; of its {r_n} comparisons on "
                     f"the test records, {_tally_text(r_up, r_down)}.")  # fmt: skip

    ens, fresh, tuned = pt._ensemble_sentence(), pt._fresh_sentence(), pt._tuning_sentence()
    p18 = _load("phase6_seed_ensemble")
    if ens and fresh and tuned and p18 and p19:
        best = p18["ensemble"]["with_af_veto"]["cnn/cvae"]
        zhu = _zhu()["rare_macro_f1"]
        on_new = p19["ensemble"]["with_af_veto"]["cnn/cvae"]["f1_S"]["point"]
        p20 = _load("phase6_rule_tuning")
        chosen = p20["conditions"]["chosen"]["cnn/cvae"]["f1_S"]["point"] if p20 else None
        text = ("\\paragraph{The best pipeline, and how far to trust it.} In P18, " + _first_lower(ens)
                + f" The mean of its S and F F1 is {_near(best['rare_macro_f1']['point'], zhu)} the best published inter-patient result we could "
                f"check, {pt._n(zhu)} for Zhu et al.~\\cite{{zhu2022}} (Section~\\ref{{sec:all-results}}). The pipeline was designed after the test "
                f"records had been read many times, so this score is optimistic. In P19, {_first_lower(fresh)} Icentia11k has no fusion label, so only S "
                f"and V can be scored there. In P20, {_first_lower(tuned)}")  # fmt: skip
        news = [v for v in (on_new, chosen) if v is not None]
        if news and max(news) < best["f1_S"]["point"]:
            text += (" On new patients, then, the pipeline's supraventricular F1 ("
                     + " and ".join(pt._n(v) for v in news) + f") stays well below the {pt._n(best['f1_S']['point'])} measured on the test records.")
        paras.append(text)
    return "\n\n".join(paras) + "\n"


def conclusion() -> str:
    """A short conclusion: the pre-registered answer, what moved the score and why, what the system is, and what is needed next."""
    final, p7, p18 = _load(FINAL_EVAL), _load("phase6_relative_cnn"), _load("phase6_seed_ensemble")
    if final is None:
        return pt._missing("conclusion: after the final evaluation") + "\n"
    conf = final["families"]["confirmatory"]
    favour = sum(bool(c["reject_holm"]) and c["point"] > 0 for c in conf.values())
    text = ("Under a strict inter-patient protocol, with every choice fixed in advance, class-conditional generative augmentation "
            + ("did not detectably improve" if not favour else "improved") + f" rare-arrhythmia detection on MIT-BIH ({favour} of {len(conf)} "
            "confirmatory comparisons in its favour), and the follow-up analyses did not change that answer. The synthetic beats came from the same "
            "few training patients as the real ones, so they added beats but not the variety of new patients.")  # fmt: skip
    if p7 is not None and p18 is not None:
        text += (" What did move the score uses the same cues a cardiologist uses: each beat seen against the patient's own normal beat "
                 f"({pt._n(final['conditions']['cnn/none']['rare_macro_f1']['point'])} to {pt._n(p7['conditions']['cnn/none']['rare_macro_f1']['point'])}), "
                 "prematurity judged against the recording's mean heart period, and no supraventricular calls during atrial fibrillation; with ten runs "
                 f"averaged the pipeline reached {pt._n(p18['ensemble']['with_af_veto']['cnn/cvae']['rare_macro_f1']['point'])}, a score that is "
                 "optimistic after repeated reads of the test records and lower on new patients.")  # fmt: skip
    return text + (" The system delivers what the course asked for, a class, a saliency map and a checked note for every beat; it is a course "
                   "project and not a medical device. Larger models, more runs and above all more training patients with supraventricular and "
                   "fusion beats are needed to establish how much augmentation can add.\n")


def build() -> dict[str, str]:
    """Every report text: name -> LaTeX (written as ``paper/generated/<name>.tex``)."""
    return {"report_summary": summary(), "report_example": example_output(), "report_architecture": pt.architecture_section(flowing=True),
            "report_posthoc": posthoc_story(), "report_conclusion": conclusion()}
