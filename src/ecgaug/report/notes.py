"""Clinician-readable notes for one classified beat, and the fact checker that guards them (PLAN Phase 5).

The note generator only restates the model's output. Its input is the JSON of docs/RULES.md §5::

    {record_anon_id, predicted_class, probabilities{N,S,V,F}, low_confidence, top_segments[{name, mass}], model_id}

| variant | generator |
|---|---|
| A | deterministic template, no LLM: the permanent fallback |
| B | LLM with the JSON and a constrained system prompt (restate only; fixed disclaimer) |
| C | B plus few-shot examples and a fixed JSON output schema, rendered to a note |
| D | B plus the fact checker and a regenerate loop (at most 2 retries, then the template) |

**Fact checker** (the hard gate: zero unsupported claims): every number in a note must be one of the
input's probabilities or segment masses (as a decimal or a percentage), every class and segment named
must be in the input, no diagnosis outside the four beat classes may appear, the disclaimer must be
present word for word, and a low-confidence case must say that clinician review is required.

**LLM provider** from the environment: ``ECGAUG_LLM_PROVIDER`` (``gemini``, the default, or
``otherllm``), ``ECGAUG_LLM_MODEL`` (default per provider in ``DEFAULT_MODELS``) and the provider's key
(``GEMINI_API_KEY`` or ``LLM_API_KEY``). Gemini's thinking level is ``ECGAUG_LLM_THINKING`` (default
``minimal``): thinking tokens count against the 400-token limit, and at the model's default level they used it all. The key is read from the environment only and is never
printed or written. The request uses the standard library (no new dependency); rate-limit replies (429,
503) are retried with a wait. Without a key, B, C and D cannot run and the template is used.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

CLASS_NAMES = {"N": "normal beat", "S": "supraventricular ectopic beat", "V": "ventricular ectopic beat", "F": "fusion beat"}
SEGMENT_NAMES = {"P": "P wave", "QRS": "QRS complex", "ST": "ST segment", "T": "T wave"}
DISCLAIMER = "This note was generated automatically by a research model. It is not a diagnosis and must be reviewed by a clinician."
REVIEW_REQUIRED = "Clinician review is required."
DEFAULT_MODELS = {"gemini": "gemini-3.5-flash-lite", "otherllm": "llm-model"}  # A23: flash allows 20 free requests a day
GEMINI_THINKING = "minimal"
API_KEYS = {"gemini": "GEMINI_API_KEY", "otherllm": "LLM_API_KEY"}
API_URLS = {"gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "otherllm": "https://api.example.invalid/v1/messages"}  # fmt: skip

# Terms that would be claims beyond the input (diagnoses, causes, advice).
UNSUPPORTED_TERMS = (
    "fibrillation", "flutter", "infarction", "ischemi", "ischaemi", "tachycardia", "bradycardia", "block", "hypertroph",
    "pacemaker", "pacing", "medication", "drug", "treat", "prognos", "syndrome", "cardiomyopathy", "electrolyte", "recommend",
)  # fmt: skip
_SEGMENT_PATTERNS = {"P": r"\bP[- ]?wave\b|\bP segment\b", "QRS": r"\bQRS\b", "ST": r"\bST\b", "T": r"\bT[- ]?wave\b|\bT segment\b"}
_CLASS_PATTERNS = {c: rf"\b{c}\b|{re.escape(name)}" for c, name in CLASS_NAMES.items()}


def note_input(record_anon_id: str, probabilities: dict[str, float], segment_masses: dict[str, float], model_id: str,
               low_confidence_threshold: float, n_segments: int = 2) -> dict:  # fmt: skip
    """The note generator's input JSON (docs/RULES.md §5). Values are rounded as they will be shown."""
    predicted = max(probabilities, key=probabilities.get)
    top = sorted(segment_masses.items(), key=lambda kv: -kv[1])[:n_segments]
    return {
        "record_anon_id": record_anon_id,
        "predicted_class": predicted,
        "probabilities": {c: round(float(p), 2) for c, p in probabilities.items()},
        "low_confidence": bool(max(probabilities.values()) < low_confidence_threshold),
        "top_segments": [{"name": name, "mass": round(float(mass), 2)} for name, mass in top],
        "model_id": model_id,
    }


# --------------------------------------------------------------------------------------------------
# A: template
# --------------------------------------------------------------------------------------------------


def template_note(payload: dict) -> str:
    """Variant A: a fixed sentence pattern filled from the input. Needs no network."""
    c = payload["predicted_class"]
    p = payload["probabilities"]
    others = ", ".join(f"{k} {v:.2f}" for k, v in p.items() if k != c)
    segments = " and ".join(f"the {SEGMENT_NAMES[s['name']]} ({s['mass']:.0%})" for s in payload["top_segments"])
    lines = [
        f"Beat classification: {CLASS_NAMES[c]} ({c}), probability {p[c]:.2f}. Other classes: {others}.",
        f"The model relied most on {segments}." if segments else "",
    ]
    if payload["low_confidence"]:
        lines.append(f"The model is not confident about this beat. {REVIEW_REQUIRED}")
    lines.append(DISCLAIMER)
    return " ".join(line for line in lines if line)


# --------------------------------------------------------------------------------------------------
# fact checker
# --------------------------------------------------------------------------------------------------


def _allowed_numbers(payload: dict) -> list[float]:
    values = list(payload["probabilities"].values()) + [s["mass"] for s in payload["top_segments"]]
    return [float(v) for v in values]


def check_note(note: str, payload: dict) -> dict:
    """Unsupported claims in a note, and which required parts it covers."""
    issues: list[str] = []
    text = note.replace(str(payload["model_id"]), " ").replace(str(payload["record_anon_id"]), " ")
    allowed = _allowed_numbers(payload)
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(%?)", text):
        value = float(match.group(1)) / (100 if match.group(2) else 1)
        tolerance = 0.005 if match.group(2) else 0.0051
        if not any(abs(value - a) <= tolerance for a in allowed):
            issues.append(f"number not in the input: {match.group(0).strip()}")
    for cls, pattern in _CLASS_PATTERNS.items():
        if re.search(pattern, text) and cls not in payload["probabilities"]:
            issues.append(f"class not in the input: {cls}")
    present_segments = {s["name"] for s in payload["top_segments"]}
    for name, pattern in _SEGMENT_PATTERNS.items():
        if re.search(pattern, text) and name not in present_segments:
            issues.append(f"segment not among the input's top segments: {name}")
    lowered = text.lower()
    for term in UNSUPPORTED_TERMS:
        if term in lowered:
            issues.append(f"claim beyond the input: '{term}'")
    if DISCLAIMER not in note:
        issues.append("disclaimer missing")
    if payload["low_confidence"] and REVIEW_REQUIRED not in note:
        issues.append("low-confidence case without 'Clinician review is required.'")
    c = payload["predicted_class"]
    coverage = {
        "predicted_class": bool(re.search(_CLASS_PATTERNS[c], text)),
        "probability": any(abs(float(m.group(1)) / (100 if m.group(2) else 1) - payload["probabilities"][c]) <= 0.0051
                           for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(%?)", text)),
        "segments": all(re.search(_SEGMENT_PATTERNS[s["name"]], text) for s in payload["top_segments"]),
        "disclaimer": DISCLAIMER in note,
        "review_statement": (REVIEW_REQUIRED in note) if payload["low_confidence"] else True,
    }  # fmt: skip
    return {"unsupported_claims": len(issues), "issues": issues, "coverage": coverage, "coverage_fraction": sum(coverage.values()) / len(coverage)}


# --------------------------------------------------------------------------------------------------
# LLM variants
# --------------------------------------------------------------------------------------------------


class LLMQuotaError(RuntimeError):
    """The provider's daily request quota is used up; waiting will not help."""


class LLMClient:
    """Gemini or another provider text generation over HTTPS (standard library only)."""

    def __init__(self, model: str, api_key: str, provider: str = "gemini", max_tokens: int = 400, timeout_s: float = 60.0,
                 max_waits: int = 6, thinking: str | None = GEMINI_THINKING) -> None:  # fmt: skip
        if provider not in API_URLS:
            raise ValueError(f"LLM provider {provider!r} is not built; use one of {sorted(API_URLS)}")
        self.model, self._key, self.provider = model, api_key, provider
        self.max_tokens, self.timeout_s, self.max_waits, self.thinking = max_tokens, timeout_s, max_waits, thinking

    def _request(self, system: str, messages: list[dict]) -> urllib.request.Request:
        if self.provider == "otherllm":
            body = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}
            headers = {"x-api-key": self._key, "api-version": "2023-06-01"}
        else:
            contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
            body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                    "generationConfig": {"maxOutputTokens": self.max_tokens}}  # fmt: skip
            if self.thinking:
                body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking}
            headers = {"x-goog-api-key": self._key}  # in a header, never in the URL
        url = API_URLS[self.provider].format(model=self.model)
        return urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={**headers, "content-type": "application/json"})

    def _parse(self, data: dict) -> tuple[str, dict]:
        if self.provider == "otherllm":
            text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
            return text, data.get("usage", {})
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        meta = data.get("usageMetadata", {})
        output = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
        return text, {"input_tokens": int(meta.get("promptTokenCount", 0)), "output_tokens": output}

    def complete(self, system: str, messages: list[dict]) -> dict:
        for wait in range(self.max_waits + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(self._request(system, messages), timeout=self.timeout_s) as response:
                    data = json.loads(response.read())
                break
            except urllib.error.HTTPError as error:  # rate limit or overload: wait and try again
                body = error.read().decode(errors="replace")
                if error.code == 429 and "PerDay" in body:  # a daily quota does not come back by waiting
                    raise LLMQuotaError(f"{self.provider} {self.model}: the daily request quota is used up; try tomorrow or another model") from error
                if error.code not in (429, 503) or wait == self.max_waits:
                    raise
                time.sleep(float(error.headers.get("retry-after") or min(60, 5 * 2**wait)))
            except (TimeoutError, urllib.error.URLError) as error:  # timeout or dropped connection: try again
                if wait == self.max_waits:
                    raise
                time.sleep(min(60, 5 * 2**wait))
        text, usage = self._parse(data)
        return {"text": text.strip(), "usage": usage, "latency_s": time.perf_counter() - started}

    def __repr__(self) -> str:  # never show the key
        return f"LLMClient(provider={self.provider!r}, model={self.model!r}, thinking={self.thinking!r})"


def client_from_env() -> LLMClient | None:
    """The configured LLM client, or None if the provider's key is not set."""
    provider = os.environ.get("ECGAUG_LLM_PROVIDER", "gemini")
    if provider not in API_KEYS:
        raise ValueError(f"LLM provider {provider!r} is not built; use one of {sorted(API_KEYS)}")
    key = os.environ.get(API_KEYS[provider])
    if not key:
        return None
    thinking = os.environ.get("ECGAUG_LLM_THINKING", GEMINI_THINKING) if provider == "gemini" else None
    return LLMClient(os.environ.get("ECGAUG_LLM_MODEL", DEFAULT_MODELS[provider]), key, provider, thinking=thinking)


SYSTEM_B = (
    "You restate the output of a heartbeat classifier for a clinician, in 2 to 4 plain sentences. "
    "Use only the facts in the JSON you are given. Do not add diagnoses, causes, advice or any number that is not in the JSON. "
    "State the predicted class by its name ({names}) with its probability, and name the segments in top_segments with their mass "
    "(segment names: {segments}). If low_confidence is true, include the sentence '{review}' word for word. "
    "End with this sentence word for word: '{disclaimer}'"
).format(names=", ".join(f"{k} = {v}" for k, v in CLASS_NAMES.items()), segments=", ".join(f"{k} = {v}" for k, v in SEGMENT_NAMES.items()),
         review=REVIEW_REQUIRED, disclaimer=DISCLAIMER)  # fmt: skip

SCHEMA_C = {"prediction": "one sentence: class name, code and probability", "attention": "one sentence: the top segments and their mass",
            "review": "the review sentence if low_confidence is true, else an empty string"}  # fmt: skip


def _few_shot() -> list[dict]:
    examples = [
        note_input("example-1", {"N": 0.92, "S": 0.05, "V": 0.02, "F": 0.01}, {"QRS": 0.48, "T": 0.22, "P": 0.1, "ST": 0.2}, "example", 0.6),
        note_input("example-2", {"N": 0.4, "S": 0.45, "V": 0.1, "F": 0.05}, {"P": 0.35, "QRS": 0.3, "ST": 0.2, "T": 0.15}, "example", 0.6),
    ]
    messages = []
    for payload in examples:
        c = payload["predicted_class"]
        answer = {
            "prediction": f"The beat is classified as a {CLASS_NAMES[c]} ({c}) with probability {payload['probabilities'][c]:.2f}.",
            "attention": "The model relied most on " + " and ".join(f"the {SEGMENT_NAMES[s['name']]} ({s['mass']:.0%})" for s in payload["top_segments"]) + ".",
            "review": REVIEW_REQUIRED if payload["low_confidence"] else "",
        }  # fmt: skip
        messages += [{"role": "user", "content": json.dumps(payload)}, {"role": "assistant", "content": json.dumps(answer)}]
    return messages


def render_c(answer: dict) -> str:
    """The note from variant C's JSON; the disclaimer is added here once, even if the model also wrote it in a field."""
    parts = [str(answer.get(key) or "").replace(DISCLAIMER, "") for key in ("prediction", "attention", "review")]
    return " ".join(p.strip() for p in [*parts, DISCLAIMER] if p.strip())


def generate(variant: str, payload: dict, client: LLMClient | None, max_retries: int = 2) -> dict:
    """A note by variant A-D, with what it cost. LLM variants fall back to the template if there is no client."""
    if variant == "A" or client is None:
        return {"note": template_note(payload), "variant_used": "A", "attempts": 0, "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0,
                "fell_back": variant != "A"}  # fmt: skip
    usage = {"input_tokens": 0, "output_tokens": 0}
    latency = 0.0

    def call(system: str, messages: list[dict]) -> str:
        nonlocal latency
        reply = client.complete(system, messages)
        latency += reply["latency_s"]
        for key in usage:
            usage[key] += int(reply["usage"].get(key, 0))
        return reply["text"]

    if variant == "C":
        system = SYSTEM_B + " Answer with a JSON object only, with these keys: " + json.dumps(SCHEMA_C)
        raw = call(system, [*_few_shot(), {"role": "user", "content": json.dumps(payload)}])
        try:
            note = render_c(json.loads(raw[raw.find("{") : raw.rfind("}") + 1]))
        except ValueError:
            note = raw
        return {"note": note, "variant_used": "C", "attempts": 1, "latency_s": latency, **usage, "fell_back": False}
    messages = [{"role": "user", "content": json.dumps(payload)}]
    note = call(SYSTEM_B, messages)
    attempts = 1
    if variant == "D":
        for _ in range(max_retries):
            report = check_note(note, payload)
            if not report["issues"]:
                break
            messages += [{"role": "assistant", "content": note},
                         {"role": "user", "content": "Rewrite the note. Fix these problems: " + "; ".join(report["issues"])}]  # fmt: skip
            note = call(SYSTEM_B, messages)
            attempts += 1
        if check_note(note, payload)["issues"]:
            return {"note": template_note(payload), "variant_used": "A", "attempts": attempts, "latency_s": latency, **usage, "fell_back": True}
    return {"note": note, "variant_used": variant, "attempts": attempts, "latency_s": latency, **usage, "fell_back": False}
