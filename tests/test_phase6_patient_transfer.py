"""P9 patient transfer: the transferred beat, the RR scaling and the choice of target patients (hand-made cases)."""

import numpy as np

from ecgaug.experiments import phase6_patient_transfer as tr


def test_transfer_adds_the_deviation_to_the_target_template_and_scales_rr():
    deviation = np.array([[0.0, 1.0, 0.0, -1.0]])
    target = np.array([[1.0, 1.0, 1.0, 1.0]])
    x, rr = tr.transfer(deviation, np.array([[0.5, 1.0, 0.6]]), np.array([1.0]), target, np.array([0.8]))
    expected = np.array([1.0, 2.0, 1.0, 0.0])
    expected = (expected - expected.mean()) / expected.std()
    assert np.allclose(x[0], expected, atol=1e-6)
    assert np.allclose(rr[0], [0.4, 0.8, 0.6])  # rr_prev and rr_next scaled by 0.8 / 1.0, the ratio unchanged


def test_targets_are_n_beats_of_another_record():
    records = np.array([1, 1, 2, 2, 3, 3])
    y = np.array([0, 1, 0, 1, 0, 2])
    rng = np.random.default_rng(0)
    out = tr.draw_targets(rng, np.array([1] * 50 + [2] * 50), records, y)
    assert set(y[out]) == {0}
    assert not np.any(records[out[:50]] == 1) and not np.any(records[out[50:]] == 2)
    assert set(records[out[:50]]) == {2, 3}  # both other patients are used
