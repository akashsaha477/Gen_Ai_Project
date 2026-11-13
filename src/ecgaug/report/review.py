"""A clinician-style review of one beat for the demo: written by Gemini from the offline analysis, then checked against it.

Added after the phase 5 gate, for the demo only (decision A44). The gate's note variants (A-D, ``ecgaug.report.notes``)
and their results are unchanged, and ``python -m ecgaug.demo --notes gate`` still writes the gate's choice.

The input (``review_facts``) is the note input of docs/RULES.md §5 plus what the offline analysis adds: every segment's
share of the saliency, the RR intervals around the beat and how many saved runs of our models predict the same class.
Gemini answers in three fields (findings, impression, confidence), in the style of a cardiologist's short report; the
review then ends with a fixed next step and the disclaimer. ``check_review`` applies the fact checker's rules to these
facts: every number must be one of them, every class and segment named must be in them, and no diagnosis, rhythm,
conduction finding, beat origin or waveform description (none of which the facts hold) may appear. A review that fails
is sent back with its problems (at most two rewrites) and then replaced by ``template_review``, the same report filled
in without an LLM.
"""

from __future__ import annotations

import json
import re

from ecgaug.report.notes import _SEGMENT_PATTERNS, CLASS_NAMES, DISCLAIMER, REVIEW_REQUIRED, SEGMENT_NAMES, UNSUPPORTED_TERMS

SEGMENTS = ("P", "QRS", "ST", "T")
FIELDS = ("findings", "impression", "confidence")
NEXT_STEP = f"Next step: {REVIEW_REQUIRED}"
PREVIOUS_RR = 10  # rr_ratio is the RR interval before the beat over the mean of the previous 10 (data contract)
# What a clinician might add that the facts do not hold: rhythm, conduction, beat origin, waveform shape.
NOT_IN_FACTS = ("rhythm", "sinus", "conduction", "bundle", "aberran", "compensatory", "atrial", "junctional", "nodal", "wide", "narrow",
                "inverted", "elevat", "depress", "axis", "amplitude")  # fmt: skip
LABELS = {"rr_before_ms": "RR interval before the beat (ms)", "rr_after_ms": "RR interval after the beat (ms)",
          "rr_before_over_mean_of_previous_10": "RR before / mean of the previous 10", "previous_rr_intervals_averaged": "RR intervals averaged",
          "low_confidence_threshold": "low-confidence threshold", "saved_runs_total": "saved runs", "saved_runs_same_class": "saved runs predicting the class"}  # fmt: skip

SYSTEM = (
    "You are a cardiologist writing a short report on one heartbeat that an automated classifier has analysed. "
    "Use only the facts in the JSON you are given. Answer with a JSON object only, with the keys findings, impression and confidence. "
    "findings: one or two sentences on the timing of the beat (the RR interval before it in ms, its ratio to the mean of the previous 10 RR "
    "intervals, and the RR interval after it in ms) and on where the classifier's attention lay (the segments, with their share of the saliency); "
    "you may say that the beat came earlier than the preceding beats when the ratio is below 1, or later when it is above 1. "
    "impression: one sentence naming the predicted class by its name and code with its probability, and the next most likely class with its probability. "
    "confidence: one sentence comparing the probability with the low-confidence threshold and, if the JSON gives them, saying how many of the "
    "saved runs of the models predict the same class. "
    "Write in the concise, professional style of a clinical report. Class names: {names}. Segment names: {segments}. "
    "Do not name any diagnosis, rhythm, conduction finding, beat origin or condition other than these four classes; do not describe the "
    "waveform's shape, width, amplitude or axis; give no advice; use no number that is not in the JSON."
).format(names=", ".join(f"{k} = {v}" for k, v in CLASS_NAMES.items()), segments=", ".join(f"{k} = {v}" for k, v in SEGMENT_NAMES.items()))


def review_facts(out: dict) -> dict:
    """The review's input, from the demo's offline analysis of one beat (``ecgaug.demo.explain_beat``)."""
    p = out["prediction"]
    facts = {
        "record_anon_id": f"record-{out['record']}-beat-{out['beat']}",
        "predicted_class": p,
        "probabilities": {c: round(float(v), 2) for c, v in out["probabilities"].items()},
        "low_confidence": bool(out["low_confidence"]),
        "low_confidence_threshold": out["threshold"],
        "segment_shares": {s: round(float(out["segment_masses"].get(s, 0.0)), 2) for s in SEGMENTS},
        "top_segments": list(out["top_segments"]),
        "rr_before_ms": round(out["rr"]["before_s"] * 1000),
        "rr_after_ms": round(out["rr"]["after_s"] * 1000),
        "rr_before_over_mean_of_previous_10": round(out["rr"]["ratio"], 2),
        "previous_rr_intervals_averaged": PREVIOUS_RR,
        "model_id": out["model_id"],
    }
    if out["offline_runs"]:
        facts["saved_runs_total"] = len(out["offline_runs"])
        facts["saved_runs_same_class"] = sum(r["prediction"] == p for r in out["offline_runs"])
    return facts


def _values(facts: dict) -> dict[str, float]:
    values = {f"probability of {c}": float(v) for c, v in facts["probabilities"].items()}
    values.update({f"{SEGMENT_NAMES[s]} share": float(v) for s, v in facts["segment_shares"].items()})
    values.update({label: float(facts[key]) for key, label in LABELS.items() if key in facts})
    for key in ("rr_before_ms", "rr_after_ms"):  # the same interval in seconds
        values[LABELS[key].replace("(ms)", "(s)")] = facts[key] / 1000
    return values


def render(answer: dict) -> str:
    """The review from Gemini's three fields: Findings, Impression, Confidence, the fixed next step and the disclaimer."""
    if not isinstance(answer, dict) or not all(isinstance(answer.get(k), str) and answer[k].strip() for k in FIELDS):
        raise ValueError(f"the answer needs the text fields {', '.join(FIELDS)}")
    text = {k: re.sub(rf"^\s*{k}\s*:\s*", "", " ".join(answer[k].replace(DISCLAIMER, "").split()), flags=re.IGNORECASE) for k in FIELDS}
    return "\n".join([f"Findings: {text['findings']}", f"Impression: {text['impression']}", f"Confidence: {text['confidence']}", NEXT_STEP, DISCLAIMER])


def template_review(facts: dict) -> str:
    """The same report filled in without an LLM: the fallback, and the review when no key is set."""
    p, probs, shares = facts["predicted_class"], facts["probabilities"], facts["segment_shares"]
    q = max((c for c in probs if c != p), key=probs.get)
    ratio = facts["rr_before_over_mean_of_previous_10"]
    timing = "earlier than" if ratio < 1 else "later than" if ratio > 1 else "in step with"
    where = " and ".join(f"the {SEGMENT_NAMES[s]} ({shares[s]:.0%})" for s in facts["top_segments"])
    confidence = f"The probability is {'below' if facts['low_confidence'] else 'at or above'} the low-confidence threshold of {facts['low_confidence_threshold']:.2f}"
    if "saved_runs_total" in facts:
        confidence += f"; {facts['saved_runs_same_class']} of the {facts['saved_runs_total']} saved runs of our models predict {p} for this beat"
    return render({
        "findings": f"The beat came {facts['rr_before_ms']} ms after the previous one, {ratio:.2f} of the mean of the previous {PREVIOUS_RR} RR "
                    f"intervals ({timing} the preceding beats), and the next beat followed after {facts['rr_after_ms']} ms. "
                    f"The classifier's attention lay mostly on {where}.",
        "impression": f"{CLASS_NAMES[p].capitalize()} ({p}), probability {probs[p]:.2f}; the next most likely class is {CLASS_NAMES[q]} ({q}), {probs[q]:.2f}.",
        "confidence": confidence + ".",
    })  # fmt: skip


def claims(text: str, facts: dict) -> list[dict]:
    """Each claim of a review next to the fact it restates, in reading order; the rows with ``matches`` False are its problems."""
    body = text.replace(str(facts["model_id"]), " ").replace(str(facts["record_anon_id"]), " ")
    values = _values(facts)
    rows = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(%?)", body):
        value = float(m.group(1)) / (100 if m.group(2) else 1)
        tolerance = 0.005 if m.group(2) else 0.0051
        names = [name for name, v in values.items() if abs(value - v) <= tolerance]
        rows.append({"at": m.start(), "claim": m.group(0).strip(), "offline": " or ".join(names) or "no such value", "matches": bool(names)})
    for cls, name in CLASS_NAMES.items():  # whole words: "ventricular" inside "supraventricular" is not class V
        if found := re.search(rf"\b{cls}\b|(?i:\b{re.escape(name)})", body):
            what = "the predicted class" if cls == facts["predicted_class"] else f"class {cls}, probability {facts['probabilities'].get(cls, 0):.2f}"
            rows.append({"at": found.start(), "claim": found.group(0), "offline": what, "matches": cls in facts["probabilities"]})
    for name, pattern in _SEGMENT_PATTERNS.items():
        if found := re.search(pattern, body):
            share = facts["segment_shares"].get(name)
            rows.append({"at": found.start(), "claim": found.group(0), "offline": f"{share:.0%} of the saliency" if share is not None else "no such segment",
                         "matches": share is not None})  # fmt: skip
    lowered = body.lower()
    rows += [{"at": lowered.find(term), "claim": term, "offline": "beyond the offline analysis", "matches": False}
             for term in (*UNSUPPORTED_TERMS, *NOT_IN_FACTS) if term in lowered]  # fmt: skip
    rows.sort(key=lambda r: r["at"])
    rows.append({"at": len(body), "claim": "next step", "offline": "present word for word" if NEXT_STEP in text else "missing", "matches": NEXT_STEP in text})
    rows.append({"at": len(body), "claim": "disclaimer", "offline": "present word for word" if DISCLAIMER in text else "missing", "matches": DISCLAIMER in text})
    return rows


def check_review(text: str, facts: dict) -> dict:
    """The review's problems: every claim that does not match the facts, and the predicted class if it is not named."""
    rows = claims(text, facts)
    p = facts["predicted_class"]
    named = bool(re.search(rf"\b{p}\b|(?i:\b{re.escape(CLASS_NAMES[p])})", text))
    issues = [f"{r['claim']}: {r['offline']}" for r in rows if not r["matches"]] + ([] if named else [f"the predicted class {p} is not named"])
    return {"unsupported_claims": sum(not r["matches"] for r in rows), "issues": issues, "coverage": {"predicted_class": named}}


def write_review(facts: dict, client, max_rewrites: int = 2) -> dict:
    """Gemini's review, sent back with its problems at most ``max_rewrites`` times; ``review`` is None if it never passes."""
    messages = [{"role": "user", "content": json.dumps(facts)}]
    issues: list[str] = []
    for attempt in range(1, max_rewrites + 2):
        raw = client.complete(SYSTEM, messages)["text"]
        try:
            text = render(json.loads(raw[raw.find("{") : raw.rfind("}") + 1]))
            issues = check_review(text, facts)["issues"]
        except ValueError as err:
            issues = [f"the answer was not the JSON object asked for ({err})"]
        if not issues:
            return {"review": text, "attempts": attempt, "issues": []}
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": "Rewrite the JSON object. Fix these problems: " + "; ".join(issues)}]
    return {"review": None, "attempts": max_rewrites + 1, "issues": issues}
