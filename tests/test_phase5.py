"""Phase 5: saliency methods and their scores, notes and the fact checker, demo, error analysis, the phase 5 gate."""

import json
import shutil

import numpy as np
import pytest
import torch
from torch import nn

from conftest import REPO, variant_config
from ecgaug import paths, workflow
from ecgaug.experiments import phase5_clinical_notes as notes_runner
from ecgaug.experiments import phase5_explainability as explain_runner
from ecgaug.explain.evaluate import faithfulness, interpolate_out, segment_masses, stability
from ecgaug.explain.methods import applies, saliency
from ecgaug.models import build_model
from ecgaug.report import notes

SETTINGS = {"ig_steps": 16, "smoothgrad_samples": 8, "smoothgrad_sd": 0.1, "occlusion_window_ms": 14, "occlusion_stride_ms": 7, "fs": 360, "batch_size": 16}


class Region(nn.Module):
    """A 'model' whose class-0 logit is the signal in samples 80-100 and nothing else (scaled so the softmax does not saturate)."""

    def forward(self, x, side=None):
        s = 0.05 * x[:, 80:100].sum(dim=1)
        return torch.stack([s, -s, torch.zeros_like(s), torch.zeros_like(s)], dim=1)


def _beats(n=8, length=180, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, length)).astype(np.float32)
    x[:, 80:100] += 3.0  # class 0 is predicted
    return x


@pytest.mark.parametrize("method", ["integrated_gradients", "smoothgrad", "occlusion"])
def test_methods_find_the_region_the_model_uses(method):
    maps = np.abs(saliency(method, Region(), _beats(), None, SETTINGS))
    inside = maps[:, 80:100].sum(axis=1) / maps.sum(axis=1)
    assert maps.shape == (8, 180) and inside.mean() > 0.8


def test_gradcam_needs_a_convolution_and_returns_a_map_per_sample():
    cnn = build_model({"name": "shallow_cnn", "channels": [4, 8, 8], "kernel": 5, "side_hidden": 2}, 4, 3)
    rnn = build_model({"name": "bilstm", "hidden": 4, "layers": 1, "side_hidden": 2}, 4, 3)
    assert applies("gradcam", cnn) and not applies("gradcam", rnn)
    maps = saliency("gradcam", cnn, _beats(), np.zeros((8, 3), dtype=np.float32), SETTINGS)
    assert maps.shape == (8, 180) and (maps >= 0).all()
    with pytest.raises(ValueError, match="convolutional"):
        saliency("gradcam", rnn, _beats(), np.zeros((8, 3), dtype=np.float32), SETTINGS)


def test_faithfulness_rewards_maps_that_point_at_what_the_model_uses():
    x = _beats()
    good = np.zeros_like(x)
    good[:, 80:100] = 1.0
    bad = np.zeros_like(x)
    bad[:, :20] = 1.0
    fractions = [0.0, 0.1, 0.2, 0.3]
    gain_good = faithfulness(Region(), x, None, good, fractions, 2, 0)["gain"].mean()
    gain_bad = faithfulness(Region(), x, None, bad, fractions, 2, 0)["gain"].mean()
    assert gain_good > 0.02 and gain_bad < 0 and gain_good > gain_bad


def test_interpolation_stability_and_segment_masses():
    np.testing.assert_allclose(interpolate_out(np.array([0.0, 5.0, 2.0, 3.0]), np.array([1, 2])), [0.0, 1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    maps = rng.standard_normal((5, 180))
    assert stability([maps, maps]) == pytest.approx(1.0)
    assert abs(stability([maps, rng.standard_normal((5, 180))])) < 0.2
    masses = segment_masses(np.ones((2, 180)), pre_samples=90, fs=360.0)
    assert masses["QRS"][0] == pytest.approx(36 / 180) and sum(m[0] for m in masses.values()) <= 1.0


@pytest.mark.parametrize("letter", ["A", "B"])
def test_explainability_smoke(letter):
    out = explain_runner.smoke(explain_runner.load_config(variant_config(REPO / "configs" / "phase5_explainability", letter)))
    assert out["applicable"] == ({"cnn": True, "rnn": False} if letter == "A" else {"cnn": True, "rnn": True})


# --------------------------------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------------------------------

PAYLOAD = notes.note_input("case-01", {"N": 0.31, "S": 0.52, "V": 0.12, "F": 0.05}, {"P": 0.41, "QRS": 0.33, "ST": 0.16, "T": 0.1}, "model-7", 0.6)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls, self.model = list(replies), 0, "fake"

    def complete(self, system, messages):
        self.calls += 1
        text = self.replies.pop(0) if self.replies else "still wrong"
        return {"text": text, "usage": {"input_tokens": 10, "output_tokens": 5}, "latency_s": 0.01}


def test_template_note_is_grounded_and_complete():
    text = notes.template_note(PAYLOAD)
    report = notes.check_note(text, PAYLOAD)
    assert report["unsupported_claims"] == 0 and report["coverage_fraction"] == 1.0
    assert notes.REVIEW_REQUIRED in text and PAYLOAD["low_confidence"]
    confident = notes.note_input("x", {"N": 0.9, "S": 0.05, "V": 0.03, "F": 0.02}, {"QRS": 0.6, "T": 0.4}, "m", 0.6)
    assert notes.REVIEW_REQUIRED not in notes.template_note(confident)


def test_fact_checker_catches_unsupported_claims():
    good = notes.template_note(PAYLOAD)
    for bad, issue in (
        (good.replace("0.52", "0.85"), "number not in the input"),
        (good + " The ST segment is raised.", "segment not among"),
        (good + " Consistent with atrial fibrillation.", "claim beyond the input"),
        (good.replace(notes.DISCLAIMER, ""), "disclaimer missing"),
        (good.replace(notes.REVIEW_REQUIRED, ""), "low-confidence case"),
    ):
        assert any(issue in i for i in notes.check_note(bad, PAYLOAD)["issues"]), issue
    assert notes.check_note(good.replace("0.52", "52%"), PAYLOAD)["unsupported_claims"] == 0  # a percentage of an input value is fine


def test_llm_variants_with_a_fake_client():
    good = notes.template_note(PAYLOAD)
    out = notes.generate("B", PAYLOAD, FakeClient([good]))
    assert out["variant_used"] == "B" and out["note"] == good and out["input_tokens"] == 10
    answer = json.dumps({"prediction": "The beat is classified as a supraventricular ectopic beat (S) with probability 0.52.",
                         "attention": "The model relied most on the P wave (41%) and the QRS complex (33%).", "review": notes.REVIEW_REQUIRED})  # fmt: skip
    c = notes.generate("C", PAYLOAD, FakeClient([answer]))
    assert notes.check_note(c["note"], PAYLOAD)["unsupported_claims"] == 0 and c["note"].endswith(notes.DISCLAIMER)
    fixed = notes.generate("D", PAYLOAD, FakeClient(["probability 0.99", good]))
    assert fixed["variant_used"] == "D" and fixed["attempts"] == 2 and fixed["note"] == good
    client = FakeClient(["wrong 0.99", "wrong 0.98", "wrong 0.97"])
    fallback = notes.generate("D", PAYLOAD, client)
    assert fallback["fell_back"] and fallback["variant_used"] == "A" and client.calls == 3
    assert notes.generate("B", PAYLOAD, None)["fell_back"]


def test_llm_client_comes_from_the_environment_and_hides_the_key(monkeypatch):
    for name in ("GEMINI_API_KEY", "ECGAUG_LLM_PROVIDER", "ECGAUG_LLM_MODEL", "ECGAUG_LLM_THINKING"):
        monkeypatch.delenv(name, raising=False)
    assert notes.client_from_env() is None
    monkeypatch.setenv("GEMINI_API_KEY", "g-secret-value")
    client = notes.client_from_env()
    assert client.provider == "gemini" and client.model == notes.DEFAULT_MODELS["gemini"] and "secret" not in repr(client)
    monkeypatch.setenv("ECGAUG_LLM_MODEL", "gemini-other")
    assert notes.client_from_env().model == "gemini-other"
    monkeypatch.setenv("ECGAUG_LLM_PROVIDER", "other")
    with pytest.raises(ValueError, match="not built"):
        notes.client_from_env()


def test_gemini_request_and_reply_map_to_the_shared_format():
    client = notes.LLMClient("gemini-test", "g-secret-value", "gemini")
    request = client._request("system text", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    body = json.loads(request.data)
    assert "g-secret-value" not in request.full_url and request.get_header("X-goog-api-key") == "g-secret-value"
    assert [c["role"] for c in body["contents"]] == ["user", "model"] and body["systemInstruction"]["parts"][0]["text"] == "system text"
    assert body["generationConfig"] == {"maxOutputTokens": 400, "thinkingConfig": {"thinkingLevel": notes.GEMINI_THINKING}}
    reply = {"candidates": [{"content": {"parts": [{"text": "hidden", "thought": True}, {"text": "The note."}]}}],
             "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5, "thoughtsTokenCount": 3}}  # fmt: skip
    assert client._parse(reply) == ("The note.", {"input_tokens": 12, "output_tokens": 8})


def test_llm_client_retries_timeouts_and_stops_on_a_daily_quota(monkeypatch):
    import io
    import urllib.error

    client = notes.LLMClient("gemini-test", "g-secret-value", "gemini", max_waits=2)
    monkeypatch.setattr(notes.time, "sleep", lambda s: None)
    ok = json.dumps({"candidates": [{"content": {"parts": [{"text": "A note."}]}}], "usageMetadata": {}}).encode()
    replies = [TimeoutError("read timed out"), io.BytesIO(ok)]

    def fake_urlopen(request, timeout):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(notes.urllib.request, "urlopen", fake_urlopen)
    assert client.complete("s", [{"role": "user", "content": "q"}])["text"] == "A note."
    daily = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b'{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}'))
    monkeypatch.setattr(notes.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(daily))
    with pytest.raises(notes.LLMQuotaError, match="daily request quota"):
        client.complete("s", [{"role": "user", "content": "q"}])


def test_cases_are_stratified_with_enough_low_confidence():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 100)
    top = rng.uniform(0.3, 1.0, size=400)
    idx = notes_runner.choose_cases(y, top, n_cases=50, min_low=10, threshold=0.6, seed=0)
    assert len(idx) == 50 and len(set(idx)) == 50 and (top[idx] < 0.6).sum() >= 10
    assert min(np.bincount(y[idx], minlength=4)) >= 10


def test_variant_c_note_has_the_disclaimer_once():
    answer = {"prediction": "The beat is S.", "attention": "", "review": f"{notes.REVIEW_REQUIRED} {notes.DISCLAIMER}"}
    note = notes.render_c(answer)
    assert note.count(notes.DISCLAIMER) == 1 and note.endswith(notes.DISCLAIMER) and notes.REVIEW_REQUIRED in note


def test_notes_runner_records_llm_variants_as_not_run_without_a_key(tmp_root):
    cfg = notes_runner.load_config(variant_config(REPO / "configs" / "phase5_clinical_notes", "B"))
    cases = [{"payload": PAYLOAD}]
    assert notes_runner.run_variant(cfg, cases, None, [])["not_run"]
    result = notes_runner.run_variant(cfg, cases, FakeClient([notes.template_note(PAYLOAD)]), [])
    assert result["passes_hard_gate"] and result["variant_used"]["B"] == 1
    a = notes_runner.run_variant(notes_runner.load_config(variant_config(REPO / "configs" / "phase5_clinical_notes", "A")), cases, None, [])
    assert a["unsupported_claims"] == 0
    sheet = notes_runner.rating_sheet()
    assert sheet.is_file() and "variant" not in sheet.read_text().splitlines()[0]


def test_demo_smoke():
    from ecgaug import demo

    out = demo.smoke()
    assert out["unsupported_claims"] == 0 and out["claims_not_matching"] == 0 and out["prediction"] in "NSVF"
    assert out["note_by"] == "offline template (no LLM), because no LLM key is set (GEMINI_API_KEY)"


def test_demo_prints_the_offline_analysis_then_the_review_then_the_match(capsys):
    from ecgaug import demo

    demo.smoke(show=True)
    text = capsys.readouterr().out
    parts = ["1. Offline analysis by our models (no LLM)", "2. Review: the offline template (no LLM)", "3. The review against the offline analysis"]
    assert all(p in text for p in parts) and text.index(parts[0]) < text.index(parts[1]) < text.index(parts[2])
    assert "Explainability" in text and "QRS" in text and "0 unsupported claims" in text and "{" not in text
    assert all(f"   {label}:" in text for label in ("Findings", "Impression", "Confidence", "Next step"))
    assert "Rhythm strip" in text and "reference" in text and "RR (ms)" in text


OUT = {"record": 118, "beat": 112, "prediction": "S", "probabilities": {"N": 0.23, "S": 0.62, "V": 0.13, "F": 0.02}, "low_confidence": False,
       "threshold": 0.6, "segment_masses": {"P": 0.1836, "QRS": 0.2353, "ST": 0.196, "T": 0.3146}, "top_segments": ["T", "QRS"],
       "rr": {"before_s": 0.631, "after_s": 0.908, "ratio": 0.781}, "model_id": "run-x",
       "offline_runs": [{"prediction": "S"}] * 6 + [{"prediction": "N"}] * 30}  # fmt: skip
GOOD = {"findings": "The beat came 631 ms after the previous one, 78% of the mean of the previous 10 RR intervals, earlier than the preceding beats; "
                    "the next followed after 908 ms. Attention lay on the T wave (31%) and the QRS complex (24%).",
        "impression": "Supraventricular ectopic beat (S), probability 0.62; next most likely normal beat (N), 0.23.",
        "confidence": "Above the low-confidence threshold of 0.6; 6 of 36 saved runs predict S."}  # fmt: skip


def test_clinician_review_template_and_checker():
    from ecgaug.report import review

    facts = review.review_facts(OUT)
    assert facts["rr_before_ms"] == 631 and facts["saved_runs_same_class"] == 6 and facts["segment_shares"]["T"] == 0.31
    offline = review.template_review(facts)
    assert review.check_review(offline, facts)["issues"] == [] and offline.endswith(notes.DISCLAIMER) and review.NEXT_STEP in offline
    assert review.check_review(review.render(GOOD), facts)["issues"] == []
    bad = review.render({**GOOD, "findings": "Right bundle branch block morphology, RR 0.7 s.", "impression": "Atrial premature beat."})
    report = review.check_review(bad, facts)
    rows = review.claims(bad, facts)
    assert report["unsupported_claims"] == sum(not r["matches"] for r in rows) >= 4
    assert {"bundle", "block", "atrial", "0.7"} == {r["claim"] for r in rows if not r["matches"]}
    unnamed = review.render({**GOOD, "impression": "Normal beat (N), probability 0.23.", "confidence": "Above the low-confidence threshold of 0.6."})
    assert review.check_review(unnamed, facts)["issues"] == ["the predicted class S is not named"]
    assert review.render({**GOOD, "findings": "Findings: " + GOOD["findings"] + " " + notes.DISCLAIMER}).count(notes.DISCLAIMER) == 1
    with pytest.raises(ValueError):
        review.render({"findings": "only one field"})


def test_demo_clinician_review_rewrites_then_falls_back_with_the_reason():
    from ecgaug import demo
    from ecgaug.report import review

    facts = review.review_facts(OUT)
    offline = review.template_review(facts)
    good, bad = json.dumps(GOOD), json.dumps({**GOOD, "impression": "Atrial premature beat with aberrant conduction."})
    assert demo.clinician_review(facts, FakeGemini([good])) == (review.render(GOOD), "Gemini fake (clinician review), passed the fact checker")
    text, by = demo.clinician_review(facts, FakeGemini([bad, "no json here", good]))
    assert text == review.render(GOOD) and by.endswith("after 2 rewrites")
    text, by = demo.clinician_review(facts, FakeGemini([bad, bad, bad]))
    assert text == offline and "failed the fact checker 3 times" in by and "atrial" in by
    assert demo.clinician_review(facts, None) == (offline, "offline template (no LLM), because no LLM key is set (GEMINI_API_KEY)")
    text, by = demo.clinician_review(facts, GeminiDown([]))
    assert text == offline and "call failed" in by and "quota" in by


def test_demo_lists_beats_and_has_one_example_per_class():
    from ecgaug import demo
    from ecgaug.experiments.data_source import smoke_data

    with smoke_data() as (split, _):
        text = demo.list_beats(split.val)
        _, candidates = demo.usable_beats(split.val[0])
    assert "first S positions" in text and f"{split.val[0]:<8}{len(candidates):>7}" in text
    assert set(demo.SHOWCASE) == set(demo.CLASSES)
    with pytest.raises(SystemExit):
        demo.main(["S", "--record", "118"])


def test_note_claims_follow_the_fact_checker():
    template = notes.template_note(PAYLOAD)
    wrong = "Supraventricular ectopic beat (S), probability 0.7, with atrial fibrillation; the model relied most on the T wave (10%)."
    for note in (template, wrong, template.replace(notes.DISCLAIMER, "")):
        rows = notes.note_claims(note, PAYLOAD)
        assert sum(not r["matches"] for r in rows) == notes.check_note(note, PAYLOAD)["unsupported_claims"]
    rows = {r["claim"]: r for r in notes.note_claims(wrong, PAYLOAD)}
    assert not rows["0.7"]["matches"] and not rows["fibrillation"]["matches"] and not rows["T wave"]["matches"]
    assert rows["S"]["offline"] == "the predicted class" and rows["S"]["matches"]
    assert "ventricular ectopic beat" not in {r["claim"] for r in notes.note_claims("A supraventricular ectopic beat (S).", PAYLOAD)}


def test_demo_finds_the_beat_in_every_saved_run():
    from ecgaug import demo

    chosen = {"cnn": {"none": {"run_ids": ["phase4_full_grid-cnn.none-s0-abc1234-202611030000", "phase4_full_grid-cnn.none-s1-abc1234-202611030000"],
                               "strength": None}}}  # fmt: skip
    paths.results_dir().mkdir(parents=True, exist_ok=True)
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase4_gate": {"chosen": chosen}}))
    for k, run_id in enumerate(chosen["cnn"]["none"]["run_ids"]):
        run_dir = paths.runs_dir() / "phase4_full_grid" / run_id
        run_dir.mkdir(parents=True)
        probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2 + 0.5 * k, 0.6 - 0.5 * k, 0.1]], dtype=np.float32)
        np.savez(run_dir / "preds_val.npz", probs=probs, y=np.array([0, 1]), record=np.array([118, 118]), r_idx=np.array([100, 500]))
    rows = demo.offline_runs(118, 500)
    assert [(r["seed"], r["prediction"]) for r in rows] == [(0, "V"), (1, "S")]
    assert demo.offline_runs(118, 999) == [] and demo.offline_runs(201, 100) == []
    assert demo.describe_model(rows[1]["run_id"]) == {"arch": "cnn", "condition": "none", "strength": None, "seed": 1}


class FakeGemini(FakeClient):
    provider = "gemini"


class GeminiDown(FakeGemini):
    def complete(self, system, messages):
        raise notes.LLMQuotaError("gemini fake: the daily request quota is used up; try tomorrow or another model")


def test_demo_note_says_who_wrote_it_and_why_the_template_replaced_it(monkeypatch):
    from ecgaug import demo

    template = notes.template_note(PAYLOAD)
    grounded = template.replace("Beat classification:", "Classification of this beat:")
    assert notes.check_note(grounded, PAYLOAD)["issues"] == []
    assert demo.write_note("A", PAYLOAD, None) == (template, "template (variant A)")
    note, by = demo.write_note("C", PAYLOAD, None)
    assert note == template and "GEMINI_API_KEY" in by
    assert demo.write_note("B", PAYLOAD, FakeGemini([grounded])) == (grounded, "Gemini fake (variant B), passed the fact checker")
    note, by = demo.write_note("B", PAYLOAD, FakeGemini(["Atrial fibrillation with probability 0.99."]))
    assert note == template and "failed the fact checker" in by
    note, by = demo.write_note("D", PAYLOAD, FakeGemini([]))  # wrong every time: D gives way after its retries
    assert note == template and "failed the fact checker" in by
    note, by = demo.write_note("C", PAYLOAD, GeminiDown([]))
    assert note == template and "call failed" in by and "quota" in by
    assert demo.chosen_notes_variant() == "A"  # before the phase 5 gate
    paths.results_dir().mkdir(parents=True, exist_ok=True)
    (paths.results_dir() / "manifest.json").write_text(json.dumps({"phase5_gate": {"notes_variant": "C"}}))
    assert demo.chosen_notes_variant() == "C" and demo.chosen_notes_variant("template") == "A"


def test_error_analysis_from_saved_predictions(monkeypatch):
    from ecgaug.report import error_analysis

    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 10 + [2] * 5 + [3] * 5)
    probs = rng.dirichlet(np.ones(4), size=len(y))
    fake = {"probs": probs, "y": y, "record": np.repeat([118, 201], 25), "r_idx": np.arange(50), "rr_ratio": rng.uniform(0.5, 1.5, 50)}
    monkeypatch.setattr(error_analysis, "_predictions", lambda run_id, split: fake)
    out = error_analysis.analyse({"cnn": {"none": {"run_ids": ["a", "b"]}}}, "test")
    entry = out["cnn/none"]
    assert np.asarray(entry["confusion_matrix"]).sum() == pytest.approx(len(y)) and set(entry["per_record"]) <= {"118", "201"}
    assert entry["rr_context"]["S called N"]["n"] == int(((y == 1) & (probs.argmax(1) == 0)).sum())


# --------------------------------------------------------------------------------------------------
# the phase 5 gate
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def project(tmp_root):
    shutil.copytree(REPO / "configs", paths.configs_dir())
    return tmp_root


def _explain(letter, method, cost, gains, stabilities):
    per_arch = {arch: ({"applicable": True, "mean_gain": g, "sd_gain": 0.01, "mean_stability": st} if g is not None else {"applicable": False})
                for arch, g, st in zip(("cnn", "rnn"), gains, stabilities)}  # fmt: skip
    folder = paths.results_for("phase5_explainability")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{letter}.json").write_text(json.dumps({"variant": letter, "method": method, "cost": cost, "per_architecture": per_arch, "provisional": False}))


def _notes(letter, unsupported=0, coverage=1.0, not_run=False):
    result = {"variant": letter, "not_run": not_run, "reason": "no key", "unsupported_claims": unsupported, "passes_hard_gate": unsupported == 0,
              "coverage": coverage, "fell_back_to_template": 0, "latency_s_mean": 0.5, "output_tokens": 100, "provisional": False}  # fmt: skip
    folder = paths.results_for("phase5_clinical_notes")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{letter}.json").write_text(json.dumps(result))


def test_phase5_gate(project):
    for letter, method, cost, gains, stab in (("A", "gradcam", 1, (0.10, None), (0.8, None)), ("B", "integrated_gradients", 3, (0.12, 0.08), (0.7, 0.7)),
                                              ("C", "smoothgrad", 2, (0.05, 0.15), (0.5, 0.5)), ("D", "occlusion", 4, (0.02, 0.02), (0.9, 0.9))):  # fmt: skip
        _explain(letter, method, cost, gains, stab)
    _notes("A")
    _notes("B", unsupported=2)
    _notes("C", coverage=0.9)
    _notes("D", coverage=1.0)
    gate = workflow.evaluate_phase5_gate()
    assert gate["methods"]["cnn"]["method"] == "gradcam"  # A and B tie on faithfulness (0.02 apart); A is more stable
    assert gate["methods"]["rnn"]["method"] == "smoothgrad"  # C is best by more than the margin
    assert gate["notes_variant"] == "D" and gate["passed"]
    entry = workflow.approve_phase5_gate(["Ann"])
    assert entry["explainability"]["cnn"] == "gradcam" and "phase 5 gate — " in (paths.docs_dir() / "DECISIONS.md").read_text()


def test_kaggle_artifacts_copy_banks_and_checkpoints(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("kaggle_run", REPO / "kaggle" / "kaggle_run.py")
    kaggle_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_run)
    source, repo = tmp_path / "input", tmp_path / "repo"
    (source / "zip1" / "data" / "synthetic" / "phase3_cvae-A").mkdir(parents=True)
    (source / "zip1" / "data" / "synthetic" / "phase3_cvae-A" / "bank.json").write_text("{}")
    (source / "zip1" / "data" / "synthetic" / "phase3_cvae-A" / "seed0.npz").write_bytes(b"x")
    ckpt = source / "zip1" / "runs" / "phase4_full_grid" / "full_grid" / "cnn" / "run1" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "best.pt").write_bytes(b"w")
    copied = kaggle_run.setup_artifacts(str(source), repo=repo)
    assert "data/synthetic/phase3_cvae-A/seed0.npz" in copied and "runs/phase4_full_grid/full_grid/cnn/run1/checkpoints/best.pt" in copied
    assert kaggle_run.setup_artifacts(str(source), repo=repo) == []  # nothing is copied twice
