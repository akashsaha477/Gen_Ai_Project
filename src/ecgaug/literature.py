"""Published MIT-BIH results the report compares with, copied as the papers report them (percent), with where each
number was read. Only values that were checked against the paper or its abstract are here; F1 that a paper does not
report is computed from its sensitivity and positive predictivity by ``f1``. Our own numbers never come from here.

``protocol`` is ``inter-patient`` for de Chazal et al.'s DS1/DS2 split (train on DS1, test on DS2, the protocol of this
project) and ``intra-patient`` for a random split of beats, where the same patients are in training and test.
"""

from __future__ import annotations

CLASSES = ("N", "S", "V", "F")

PUBLISHED: tuple[dict, ...] = (
    {"key": "dechazal2004", "name": "de Chazal et al. 2004", "method": "linear discriminants, morphology and RR features, two leads", "protocol": "inter-patient",
     "se": {"S": 75.9, "V": 77.7}, "ppv": {"S": 38.5, "V": 81.9}, "f1": {},
     "source": "abstract, IEEE Trans Biomed Eng 51(7):1196-1206 (2004), doi:10.1109/TBME.2004.827359"},
    {"key": "huang2014", "name": "Huang et al. 2014", "method": "random projections, RR intervals, ensemble of SVMs", "protocol": "inter-patient",
     "se": {"S": 91.1, "V": 93.9}, "ppv": {"S": 42.2, "V": 90.9}, "f1": {},
     "source": "abstract, BioMedical Engineering OnLine 13:90 (2014), doi:10.1186/1475-925X-13-90"},
    {"key": "zhu2022", "name": "Zhu et al. 2022", "method": "RR-interval waveforms and a two-stage CNN", "protocol": "inter-patient",
     "se": {}, "ppv": {}, "f1": {"N": 98.26, "S": 68.61, "V": 95.99, "F": 47.75},
     "source": "abstract, Physiological Measurement (2022)"},
    {"key": "mousavi2019", "name": "Mousavi et al. 2019", "method": "CNN with a sequence-to-sequence model over beat sequences", "protocol": "inter-patient",
     "se": {"S": 88.94, "V": 99.94}, "ppv": {"S": 92.57, "V": 99.50}, "f1": {},
     "source": "abstract, arXiv:1812.07421 (2019)", "note": "far above every other inter-patient report; its protocol was not checked here"},
    {"key": "kwak2024diffusion", "name": "Kwak and Jung 2024", "method": "MobileNetV2 with conditional-diffusion augmentation (model 4)", "protocol": "intra-patient",
     "se": {"N": 95.68, "S": 94.68, "V": 98.88, "F": 89.74}, "ppv": {"N": 98.93, "S": 94.68, "V": 98.20, "F": 90.91},
     "f1": {"N": 98.79, "S": 94.68, "V": 98.54, "F": 90.32},
     "source": "Table 4, PeerJ Computer Science 10:e2299 (2024); MIT-BIH beats split at random 80/10/10"},
)


def f1(se: float | None, ppv: float | None) -> float | None:
    """F1 from sensitivity and positive predictivity (same units), or None if either is missing."""
    if se is None or ppv is None or se + ppv == 0:
        return None
    return 2 * se * ppv / (se + ppv)


def rows() -> list[dict]:
    """Each published result on a 0-1 scale: per class ``se``, ``ppv`` and ``f1`` (reported, else computed; ``f1_computed``
    names the classes whose F1 was computed) and the mean of S and F F1 where both exist."""
    out = []
    for p in PUBLISHED:
        r = {"key": p["key"], "name": p["name"], "method": p["method"], "protocol": p["protocol"], "source": p["source"], "note": p.get("note", ""),
             "se": {c: v / 100 for c, v in p["se"].items()}, "ppv": {c: v / 100 for c, v in p["ppv"].items()}, "f1": {}, "f1_computed": []}  # fmt: skip
        for c in CLASSES:
            if c in p["f1"]:
                r["f1"][c] = p["f1"][c] / 100
            elif (v := f1(p["se"].get(c), p["ppv"].get(c))) is not None:
                r["f1"][c] = v / 100
                r["f1_computed"].append(c)
        r["rare_macro_f1"] = (r["f1"]["S"] + r["f1"]["F"]) / 2 if "S" in r["f1"] and "F" in r["f1"] else None
        out.append(r)
    return out
