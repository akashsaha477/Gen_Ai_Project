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
    for name in ("GEMINI_API_KEY", "LLM_API_KEY", "ECGAUG_LLM_PROVIDER", "ECGAUG_LLM_MODEL", "ECGAUG_LLM_THINKING"):
        monkeypatch.delenv(name, raising=False)
    assert notes.client_from_env() is None
    monkeypatch.setenv("GEMINI_API_KEY", "g-secret-value")
    client = notes.client_from_env()
    assert client.provider == "gemini" and client.model == notes.DEFAULT_MODELS["gemini"] and "secret" not in repr(client)
    monkeypatch.setenv("ECGAUG_LLM_PROVIDER", "otherllm")
    assert notes.client_from_env() is None
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")
    monkeypatch.setenv("ECGAUG_LLM_MODEL", "llm-model")
    client = notes.client_from_env()
    assert client.provider == "otherllm" and client.model == "llm-model" and "secret" not in repr(client)
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
    assert out["unsupported_claims"] == 0 and out["prediction"] in "NSVF"


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
