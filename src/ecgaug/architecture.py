"""Every model the project trained or tested, for the report's architecture section and flow figure.

``inventory()`` returns one row per model or component: its group, name, role, structure, trainable parameters (counted
by building the model from its config, never typed), where it ran and what happened to it (choices read from the gate
records in ``results/manifest.json`` and scores from the result files). ``best_pipeline()`` gives the post-hoc pipeline
with the highest test score so far (P18) and that score. Nothing here trains or reads test data.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ecgaug import paths, workflow

GROUPS = ("classifier", "generator", "decision", "explanation", "note")
SIDE = 3  # RR side features: rr_prev, rr_next, rr_ratio
CLASSES = 4


def _model_parameters(model: dict, n_side: int = SIDE) -> int:
    from ecgaug.models import build_model, count_parameters

    return int(count_parameters(build_model(model, CLASSES, n_side)))


def _generator_parameters(config: Path, setting: int = 0) -> int:
    from ecgaug.experiments import generator_runs as gr
    from ecgaug.generators import build_generator

    cfg = gr.load_config(paths.repo_root() / config)
    _, g = gr.settings_grid(cfg)[setting]
    return int(build_generator(cfg["family"], gr.generator_settings(cfg, g), 180, 0).n_parameters())


def _diffusion_parameters() -> int:
    from ecgaug.generators import build_generator

    d = yaml.safe_load((paths.repo_root() / "configs/phase6_diffusion/diffusion.yaml").read_text())
    return int(build_generator("diffusion", {**d["training"], **d["generator"], "classes": d["classes"]}, 180, 0).n_parameters())


def _json(path: str) -> dict | None:
    p = paths.repo_root() / path
    return json.loads(p.read_text()) if p.is_file() else None


def _phase2(variant: str) -> dict:
    return _json(f"results/phase2_baseline_classifiers/{variant}.json") or {}


def _gate(name: str) -> dict:
    return workflow.manifest().get(name) or {}


def _row(group: str, name: str, role: str, structure: str, parameters: int | None, where: str, outcome: str, chosen: bool = False) -> dict:
    return {"group": group, "name": name, "role": role, "structure": structure, "parameters": parameters, "where": where, "outcome": outcome, "chosen": chosen}


def classifiers() -> list[dict]:
    """The deep classifiers (phase 2's four, the probe) and the other classifiers that were fitted."""
    p2 = _gate("phase2_gate")
    chosen = {p2.get("cnn", {}).get("variant"): "cnn", p2.get("rnn", {}).get("variant"): "rnn"}

    def phase2_outcome(v: str) -> str:
        d = _phase2(v)
        score = f"validation S F1 {d['mean']:.3f} (SD {d['sd']:.3f}) at phase 2" if d else "phase 2"
        return f"{score}; chosen as the {chosen[v]}" if v in chosen else f"{score}; not chosen"

    specs = {"A": {"name": "shallow_cnn", "channels": [32, 64, 128], "kernel": 7, "side_hidden": 16},
             "B": {"name": "resnet1d", "channels": [32, 64, 128, 128], "kernel": 7, "side_hidden": 16},
             "C": {"name": "bilstm", "hidden": 64, "layers": 2, "side_hidden": 16},
             "D": {"name": "cnn_bilstm", "channels": [32, 64], "kernel": 7, "hidden": 64, "side_hidden": 16}}  # fmt: skip
    resnet = "re-tested in P18's pipeline as P22"
    label = {"none": "no augmentation", "cvae": "cVAE", "wgan_gp": "WGAN-GP"}

    def diffs(summary: dict, family: str) -> str:
        return ", ".join(f"{label[n.split(' vs ')[0].split('/')[1]]} " + f"{c['point']:+.3f}".replace("-", "−") + (" (detectable)" if c["reject_holm"] else "")
                         for n, c in summary["families"][family].items())  # fmt: skip

    p22, p22_test = _json("results/phase6_bigger_classifier/summary.json"), _json("results/phase6_bigger_classifier/test_summary.json")
    if p22:
        resnet += f": against the cnn, S F1 on P19's patients {diffs(p22, 'secondary')}"
    if p22_test:
        resnet += f"; test rare-class macro-F1 {diffs(p22_test, 'confirmatory')}"
    return [
        _row("classifier", "Probe CNN", "fixed probe", "3 conv blocks (16, 32, 64 channels, kernel 7), global average pooling, RR side branch, linear head",
             _model_parameters({"name": "probe_cnn", "channels": [16, 32, 64], "kernel": 7, "side_hidden": 16}),
             "phase 1 (input choice); quality metrics (TSTR, feature network)", "a measuring instrument, not a candidate"),
        _row("classifier", "Shallow CNN (the cnn)", "beat classifier", "3 conv-BN-ReLU-max-pool blocks (32, 64, 128, kernel 7), global average pooling; RR side branch (batch norm, 16 units); linear head",
             _model_parameters(specs["A"]), "phase 2 variant A; every later experiment", phase2_outcome("A"), chosen=True),
        _row("classifier", "ResNet1D", "beat classifier", "stem conv, 4 residual stages (32, 64, 128, 128, kernel 7, two conv-BN each with a shortcut), global average pooling, RR side branch, linear head",
             _model_parameters(specs["B"]), "phase 2 variant B; P22", phase2_outcome("B") + "; " + resnet),
        _row("classifier", "BiLSTM (the rnn)", "beat classifier", "2-layer bidirectional LSTM (64 units), last states and mean output concatenated, RR side branch, linear head",
             _model_parameters(specs["C"]), "phase 2 variant C; full grid; P5, P13", phase2_outcome("C"), chosen=True),
        _row("classifier", "CNN-BiLSTM", "beat classifier", "2 conv blocks (32, 64, kernel 7), 1 bidirectional LSTM layer (64 units), RR side branch, linear head",
             _model_parameters(specs["D"]), "phase 2 variant D", phase2_outcome("D")),
        _row("classifier", "PTB-XL pretraining", "initialization", "the probe CNN pretrained on PTB-XL beats with their recording's five diagnostic superclasses, then fine-tuned on MIT-BIH",
             _model_parameters({"name": "probe_cnn", "channels": [16, 32, 64], "kernel": 7, "side_hidden": 16}), "phase 2 (two arms: RR features, 125 Hz)",
             "from-scratch training kept in both arms"),
        _row("classifier", "Per-patient batch norm", "test-time adaptation", "frozen cnn or rnn whose batch-norm statistics are recomputed from each patient's beats", None,
             "P10", "no new parameters; post hoc"),
        _row("classifier", "RR linear discriminants", "reference", "LDA on the three RR features, and on them plus every 10th sample of the beat (de Chazal-style)", None,
             "P1", "fixed references beside the cnn; post hoc"),
        _row("classifier", "Logistic regression, gradient boosting", "design check", "models of the RR intervals compared with rules on training-side patients", None,
             "P14", "descriptive check only"),
    ]


def generators() -> list[dict]:
    """The phase 3 variants, the diffusion model, the physics losses and the augmentation without a learned model."""
    g3 = _gate("phase3a_gate")
    win = {f: (g3.get(f"phase3_{f}") or {}).get("winner") for f in ("cvae", "wgan_gp")}
    cvae = {"A": ("A_conv_concat.yaml", "conv encoder and decoder, class one-hot concatenated to the input and the latent code, latent 16, KL weight annealed over 30% of steps"),
            "B": ("B_conv_low_beta.yaml", "as A with a smaller KL weight (0.1 and 0.25)"),
            "C": ("C_conv_spectral_loss.yaml", "as A plus an L1 loss between FFT magnitudes"),
            "D": ("D_residual_film.yaml", "residual encoder and decoder with FiLM class conditioning in every block")}  # fmt: skip
    wgan = {"A": ("A_concat_embedding.yaml", "label embedding concatenated to the noise (64) and as a class channel to the critic; 5 critic steps, gradient penalty 10"),
            "B": ("B_projection_critic.yaml", "projection critic"),
            "C": ("C_auxiliary_classifier.yaml", "auxiliary class head on the critic (AC-WGAN-GP)"),
            "D": ("D_per_class_unconditional.yaml", "one unconditional WGAN-GP per class S, V, F")}  # fmt: skip
    rows = []
    for family, label, table in (("cvae", "cVAE", cvae), ("wgan_gp", "WGAN-GP", wgan)):
        for v, (cfg, structure) in table.items():
            ok = v == win[family]
            rows.append(_row("generator", f"{label} {v}", "class-conditional generator", structure, _generator_parameters(Path(f"configs/phase3_{family}") / cfg),
                             f"phase 3; {'P12 (retrained on patient-relative beats), P21, P23' if ok else 'pilots'}",
                             "chosen at the phase 3 gate" if ok else "not chosen at the phase 3 gate", chosen=ok))  # fmt: skip
    rows += [
        _row("generator", "Diffusion (1D U-Net)", "class-conditional generator", "1D U-Net (64, 128, 256 channels, kernel 5), 1,000-step cosine schedule, DDIM sampling in 50 steps, classifier-free guidance 1.5, EMA weights",
             _diffusion_parameters(), "P15; P17 (five times longer)", "post hoc; one of P18's six conditions"),
        _row("generator", "Physics loss, drawn parameters", "generator objective", "McSharry et al.'s ECG equation, Euler residual with 16 parameters drawn from the class (SimGANs)", None,
             "P21", "barely moved the generators; post hoc"),
        _row("generator", "Physics loss, fitted heights", "generator objective", "the simulator's best fit with the class's wave timing, heights and baseline solved per beat", None,
             "P23", "post hoc"),
        _row("generator", "SMOTE, random oversampling, class weights", "baselines", "interpolation between same-class beats; repeated real beats; class-weighted cross-entropy", None,
             "phase 4 grid; P5-P18", "the pre-registered comparison conditions"),
        _row("generator", "Patient transfer, fusion beats", "augmentation without a learned model", "a real deviation placed on other patients' templates (P9); a fraction of a real V deviation as F (P21)", None,
             "P9; P21", "fusion beats stopped by their pre-registered rule"),
    ]
    return rows


def decision_explanation_note() -> list[dict]:
    p5 = _gate("phase5_gate")
    method = p5.get("explainability", {})
    names = {"gradcam": "Grad-CAM", "integrated_gradients": "integrated gradients", "smoothgrad": "SmoothGrad", "occlusion": "occlusion"}
    chosen_methods = ", ".join(f"{names.get(v, v)} ({k})" for k, v in method.items())
    notes = {"A": "deterministic template", "B": "LLM restating the JSON input under a constrained prompt", "C": "B plus two examples and a fixed JSON schema",
             "D": "B plus the fact checker and up to two regenerations"}  # fmt: skip
    rows = [
        _row("decision", "Ten-run ensemble", "decision", "mean class probability of ten cnn runs, then argmax", None, "P18", "post hoc"),
        _row("decision", "Huang's SVEB stage", "rule", "a beat called N becomes S when its RR interval is below 0.8 of the recording's mean", None, "P13; P20 (threshold chosen on Icentia11k patients)", "post hoc"),
        _row("decision", "AF veto", "rule with a fitted detector", "logistic regression on five RR-irregularity features of the 64 intervals around a beat, fitted on the MIT-BIH Atrial Fibrillation Database; vetoes the stage", None,
             "P16; P20", "post hoc"),
        _row("explanation", "Saliency", "explanation", "Grad-CAM, integrated gradients, SmoothGrad, occlusion; shares of saliency in the P, QRS, ST and T windows", None,
             "phase 5", f"chosen per architecture: {chosen_methods}" if chosen_methods else "phase 5", chosen=True),
    ]
    for v, text in notes.items():
        rows.append(_row("note", f"Note generator {v}", "clinical note", text + ("" if v == "A" else "; Gemini 3.5 Flash-Lite"), None, "phase 5",
                         "chosen at the phase 5 gate" if v == p5.get("notes_variant") else ("the fallback whenever the fact check fails" if v == "A" else "not chosen"),
                         chosen=v == p5.get("notes_variant")))  # fmt: skip
    return rows


def inventory() -> list[dict]:
    return classifiers() + generators() + decision_explanation_note()


def best_pipeline() -> dict:
    """The post-hoc pipeline with the highest test rare-class macro-F1 so far: P18's cVAE ensemble with the stage and veto."""
    s = _json("results/phase6_seed_ensemble/summary.json")
    score = s["ensemble"]["with_af_veto"]["cnn/cvae"]["rare_macro_f1"] if s else None
    final = _json("results/final_eval/summary.json")
    primary = None
    if final:
        conds = final.get("conditions", {})
        primary = {k: v["rare_macro_f1"]["point"] for k, v in conds.items() if isinstance(v, dict) and "rare_macro_f1" in v}
    return {"score": score, "rule_threshold": s.get("rule_threshold") if s else None, "af_threshold": s.get("af_threshold") if s else None, "primary": primary}
