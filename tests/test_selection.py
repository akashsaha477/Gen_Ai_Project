"""The pre-registered selection rule: tie margin, tie-breaks, literal application."""

import math

import pytest

from ecgaug.eval.selection import FALLBACK, PRIMARY, Candidate, select_winner, selection_value, tie_break_value, tie_margin

METRICS = {"rare_macro_f1": 0.4, "per_class": {"S": {"f1": 0.5}, "V": {"f1": 0.8}, "F": {"f1": 0.3}}}


def test_selection_values():
    assert selection_value(METRICS, PRIMARY) == 0.4 and tie_break_value(METRICS, PRIMARY) is None
    assert selection_value(METRICS, FALLBACK) == 0.5 and tie_break_value(METRICS, FALLBACK) == 0.8
    with pytest.raises(ValueError):
        selection_value(METRICS, "accuracy")


def test_tie_margin_is_the_larger_of_the_floor_and_the_pooled_sd():
    assert tie_margin(0.0, 0.0) == 0.02
    assert tie_margin(0.01, 0.01) == 0.02
    assert tie_margin(0.03, 0.05) == pytest.approx(math.sqrt((0.03**2 + 0.05**2) / 2))


def test_clear_winner_is_not_overturned_by_simplicity():
    out = select_winner([Candidate("A", 0.50, 0.005, complexity=360), Candidate("B", 0.40, 0.005, complexity=63)])
    assert out["winner"] == "A" and out["steps"][0]["tied"] == ["A"]


def test_tie_within_the_floor_goes_to_the_simpler_variant():
    out = select_winner([Candidate("A", 0.50, 0.0, complexity=180), Candidate("D", 0.485, 0.0, complexity=63)])
    assert out["winner"] == "D" and out["best_by_mean"] == "A"


def test_tie_margin_grows_with_seed_spread():
    wide = [Candidate("A", 0.50, 0.06, complexity=180), Candidate("D", 0.45, 0.06, complexity=63)]
    assert select_winner(wide)["winner"] == "D"  # 0.05 apart, pooled SD 0.06: tied
    tight = [Candidate("A", 0.50, 0.01, complexity=180), Candidate("D", 0.45, 0.01, complexity=63)]
    assert select_winner(tight)["winner"] == "A"  # 0.05 apart, margin 0.02: not tied


def test_fallback_breaks_ties_on_v_f1_before_simplicity():
    candidates = [
        Candidate("B", 0.50, 0.0, complexity=360, tie_mean=0.80, tie_sd=0.0),
        Candidate("D", 0.49, 0.0, complexity=63, tie_mean=0.70, tie_sd=0.0),
    ]
    out = select_winner(candidates)
    assert out["winner"] == "B"
    assert [s["step"] for s in out["steps"]] == ["selection metric", "tie-break metric", "simplicity, then wall-clock"]


def test_fallback_with_both_metrics_tied_goes_to_the_simpler_variant():
    candidates = [
        Candidate("B", 0.50, 0.0, complexity=360, tie_mean=0.80, tie_sd=0.0),
        Candidate("D", 0.49, 0.0, complexity=63, tie_mean=0.79, tie_sd=0.0),
    ]
    assert select_winner(candidates)["winner"] == "D"


def test_equal_complexity_falls_back_to_wall_clock_then_name():
    a = Candidate("A", 0.50, 0.0, complexity=180, wall_clock_s=90)
    b = Candidate("B", 0.50, 0.0, complexity=180, wall_clock_s=60)
    assert select_winner([a, b])["winner"] == "B"
    assert select_winner([Candidate("B", 0.5, 0.0, 180), Candidate("A", 0.5, 0.0, 180)])["winner"] == "A"


def test_only_candidates_tied_with_the_best_are_eligible():
    candidates = [Candidate("A", 0.50, 0.0, 360), Candidate("B", 0.485, 0.0, 180), Candidate("C", 0.47, 0.0, 63)]
    assert select_winner(candidates)["winner"] == "B"  # C is 0.03 below the best, outside the margin


def test_undefined_metric_stops_the_rule():
    with pytest.raises(ValueError, match="undefined"):
        select_winner([Candidate("A", float("nan"), 0.0, 1)])
    with pytest.raises(ValueError):
        select_winner([])
