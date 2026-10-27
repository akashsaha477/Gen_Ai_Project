"""The paper's generated results sentences: wording follows the pre-registered criteria, never the other way round."""

from ecgaug import paper_text as T


def _cmp(point, reject, lo=None, hi=None, p=0.01):
    return {"point": point, "ci_low": point - 0.02 if lo is None else lo, "ci_high": point + 0.02 if hi is None else hi,
            "p_holm": p if reject else 0.4, "reject_holm": reject}  # fmt: skip


def _entry(value, f=0.3, s=0.2, strength=0.25):
    e = {"point": value, "ci_low": value - 0.05, "ci_high": value + 0.05}
    return {"rare_macro_f1": e, "f1_F": {"point": f}, "f1_S": {"point": s}, "strength": strength}


def _summary(cvae_reject=True, cvae_point=0.05, flip_without_202=False):
    confirmatory = {"cnn: cvae vs none": _cmp(cvae_point, cvae_reject), "cnn: wgan_gp vs none": _cmp(0.01, False)}
    without = {k: dict(v) for k, v in confirmatory.items()}
    if flip_without_202:
        without["cnn: cvae vs none"]["reject_holm"] = not cvae_reject
    return {
        "conditions": {"cnn/none": {**_entry(0.30, f=0.30, s=0.20), "strength": None},
                       "cnn/cvae": _entry(0.35, f=0.40, s=0.21), "cnn/wgan_gp": _entry(0.31, f=0.31, s=0.25)},
        "families": {"confirmatory": confirmatory,
                     "secondary": {"cnn: cvae vs smote": _cmp(0.03, True), "cnn: cvae vs class_weighted": _cmp(-0.02, False),
                                   "cnn: cvae vs random_oversampling": _cmp(0.04, True), "cnn: cvae vs wgan_gp": _cmp(0.04, False)}},
        "sensitivity": {"without_record_202": {"confirmatory": without}},
        "phase4_gate": {"best_generative": {"cnn": "cvae"}},
    }  # fmt: skip


def test_a_significant_gain_is_called_an_improvement_and_supports_h1():
    text, verdicts = T.rq1(_summary())
    assert "cVAE augmentation (strength 0.25) improved it" in text and "WGAN-GP augmentation (strength 0.25) did not detectably change it" in text
    assert verdicts["H1"][0] == "supported" and "unchanged" in text
    assert verdicts["H1c"][0] == "partly supported"  # beats SMOTE and random oversampling, not class-weighted loss
    assert verdicts["H1b"][0] == "mixed"  # F gain larger than S gain for cVAE, not for WGAN-GP


def test_without_a_significant_test_nothing_is_called_an_improvement():
    text, verdicts = T.rq1(_summary(cvae_reject=False))
    assert ") improved it" not in text and "no generative condition improved" in text and verdicts["H1"][0] == "not supported"
    text, _ = T.rq1(_summary(cvae_reject=True, cvae_point=-0.05))
    assert "cVAE augmentation (strength 0.25) lowered it" in text and "$-$0.050" in text


def test_the_sensitivity_analysis_reports_a_changed_conclusion():
    text, _ = T.rq1(_summary(flip_without_202=True))
    assert "different for cnn: cvae vs none" in text


def test_missing_results_stay_visible_placeholders():
    assert T.rq1(None)[0].startswith("\\missing{") and T.rq2(None)[0].startswith("\\missing{") and T.rq3(None)[0].startswith("\\missing{")
    assert "\\missing{pending}" in T.hypotheses_table({})


def test_audit_and_rq3_wording():
    audit = {"architecture": "cnn", "split": {"records": 44, "records_in_train_and_test": 44}, "inflation_none": -0.02,
             "conditions": {"none": {"intra": {"mean": 0.28, "sd": 0.01}, "inter": {"point": 0.30, "ci_low": 0.2, "ci_high": 0.4}}},
             "gains": {"cvae": {"intra": {"mean": 0.10}, "inter": {"point": 0.02, "ci_low": -0.03, "ci_high": 0.07}}}}  # fmt: skip
    text, verdicts = T.rq2(audit)
    assert "lowered by 0.020" in text and verdicts["H2"][0] == "not consistent"
    quality = {"n_checkpoints": 8, "n_runs": 2, "provisional": False, "metrics": {
        "quality_dtw": {"group": "fidelity and diversity", "rho": -0.6, "ci_low": -0.8, "ci_high": -0.3, "excludes_zero": True},
        "memorization_fraction_mean": {"group": "fidelity and diversity", "rho": float("nan"), "ci_low": None, "ci_high": None, "excludes_zero": None},
        "quality_tstr": {"group": "tstr", "rho": 0.5, "ci_low": 0.1, "ci_high": 0.8, "excludes_zero": True}}}  # fmt: skip
    text, verdicts = T.rq3(quality)
    assert "dtw ($\\rho=-0.60$" in text and "memorization fraction (undefined)" in text and "TSTR, analysed apart" in text
    assert verdicts["H3"] == ("supported", "1 of 2 metrics with an interval excluding zero")
