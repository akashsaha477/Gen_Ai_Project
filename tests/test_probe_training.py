"""The probe CNN and the training loop."""

import numpy as np
import pytest
import torch

from ecgaug.eval.selection import FALLBACK, PRIMARY
from ecgaug.models.probe import ProbeCNN, count_parameters
from ecgaug.train.loop import TrainConfig, fit_classifier, set_seed


@pytest.mark.parametrize("length", [63, 180, 360])
@pytest.mark.parametrize("n_side", [0, 3])
def test_probe_output_shape(length, n_side):
    model = ProbeCNN(n_side=n_side).eval()
    side = torch.randn(5, n_side) if n_side else None
    assert model(torch.randn(5, length), side).shape == (5, 4)
    assert model(torch.randn(5, 1, length), side).shape == (5, 4)


def test_probe_is_small_and_checks_its_side_input():
    assert count_parameters(ProbeCNN()) < 30_000
    assert count_parameters(ProbeCNN(n_side=3)) > count_parameters(ProbeCNN())
    with pytest.raises(ValueError, match="side"):
        ProbeCNN(n_side=3)(torch.randn(4, 180))
    with pytest.raises(ValueError, match="side"):
        ProbeCNN()(torch.randn(4, 180), torch.randn(4, 3))


def _separable(n_per_class=120, length=180, seed=0, classes=(0, 1, 2, 3)):
    """Four classes that differ by the position of a bump: easy to learn."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in classes:
        x = 0.2 * rng.standard_normal((n_per_class, length))
        x[:, 30 + 30 * c : 50 + 30 * c] += 2.0
        xs.append(x)
        ys.append(np.full(n_per_class, c))
    order = rng.permutation(n_per_class * len(classes))
    return {"X": np.concatenate(xs)[order].astype(np.float32), "y": np.concatenate(ys)[order].astype(np.int8)}


def _fit(seed=0, lr=3e-3, max_epochs=6, patience=5, selection=PRIMARY, val=None):
    set_seed(seed)
    model = ProbeCNN()
    cfg = TrainConfig(lr=lr, max_epochs=max_epochs, patience=patience, batch_size=64)
    return fit_classifier(model, _separable(seed=1), val or _separable(seed=2), cfg, selection, seed=seed)


def test_training_learns_a_separable_problem():
    fit = _fit()
    assert fit.best_value > 0.95
    assert fit.val_probs.shape == (480, 4)
    np.testing.assert_allclose(fit.val_probs.sum(axis=1), 1.0, atol=1e-5)
    assert fit.best_metrics["rare_macro_f1"] == pytest.approx(fit.best_value)
    assert len(fit.history) == fit.epochs_run and fit.history[0]["epoch"] == 1


def test_training_is_deterministic_for_a_seed():
    a, b, c = _fit(seed=3), _fit(seed=3), _fit(seed=4)
    np.testing.assert_array_equal(a.val_probs, b.val_probs)
    assert [h["train_loss"] for h in a.history] == [h["train_loss"] for h in b.history]
    assert not np.array_equal(a.val_probs, c.val_probs)


def test_early_stopping_keeps_the_best_epoch(monkeypatch):
    # validation values per epoch: the best is epoch 3, then three epochs without improvement
    values = iter([0.5, 0.4, 0.6, 0.3, 0.6, 0.1, 0.9, 0.9])
    monkeypatch.setattr("ecgaug.train.loop.selection_value", lambda metrics, name: next(values))
    fit = _fit(max_epochs=20, patience=3)
    assert fit.best_epoch == 3 and fit.best_value == 0.6
    assert fit.epochs_run == 6 and fit.stopped_early  # an equal value at epoch 5 is not an improvement
    assert [h["val_selection"] for h in fit.history] == [0.5, 0.4, 0.6, 0.3, 0.6, 0.1]


def test_training_runs_to_the_epoch_cap_when_it_keeps_improving(monkeypatch):
    values = iter([0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr("ecgaug.train.loop.selection_value", lambda metrics, name: next(values))
    fit = _fit(max_epochs=4, patience=2)
    assert fit.best_epoch == 4 and fit.epochs_run == 4 and not fit.stopped_early


def test_side_input_is_used_when_given():
    rng = np.random.default_rng(0)
    train, val = _separable(seed=1), _separable(seed=2)
    for part in (train, val):
        part["side"] = rng.standard_normal((len(part["y"]), 3)).astype(np.float32)
    set_seed(0)
    fit = fit_classifier(ProbeCNN(n_side=3), train, val, TrainConfig(lr=3e-3, max_epochs=2, batch_size=64), FALLBACK, seed=0)
    assert fit.val_probs.shape == (480, 4)


def test_undefined_selection_metric_is_an_error_not_a_zero():
    val_without_s = _separable(seed=2, classes=(0, 2, 3))
    with pytest.raises(ValueError, match="undefined"):
        _fit(selection=FALLBACK, val=val_without_s)
