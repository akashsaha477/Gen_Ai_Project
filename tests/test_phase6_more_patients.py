"""P11: the validation units and the cap on N beats per record (hand-made cases)."""

import numpy as np

from ecgaug.experiments import phase6_more_patients as mp


def test_every_fifth_unit_in_sorted_order_goes_to_validation():
    assert mp.validation_units(np.array([7, 3, 1, 9, 5, 2, 8, 4, 6, 10, 3]), 5) == {5, 10}


def test_cap_keeps_every_rare_beat_and_at_most_cap_normal_beats_per_record():
    y = np.array([0] * 10 + [1, 2] + [0] * 3 + [3])
    records = np.array([1] * 12 + [2] * 4)
    keep = mp.cap_normal(y, records, cap=4, seed=0)
    assert keep[10] and keep[11] and keep[15]  # S, V and F beats are all kept
    assert ((records == 1) & (y == 0) & keep).sum() == 4  # record 1 had 10 N beats
    assert ((records == 2) & (y == 0) & keep).sum() == 3  # record 2 had only 3
    assert np.array_equal(keep, mp.cap_normal(y, records, cap=4, seed=0))  # the same seed gives the same beats
