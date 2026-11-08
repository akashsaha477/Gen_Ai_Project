"""The diffusion generator: the cosine schedule, the network's shapes, the shared interface, determinism, and that training
plus DDIM sampling with guidance reproduce each class's waveform (a hand-made two-class case)."""

import numpy as np
import pytest
import torch

from ecgaug.data.records import CLASS_INDEX
from ecgaug.generators import build_generator
from ecgaug.generators.diffusion import UNet1D, cosine_alpha_bar

SMALL = {"timesteps": 200, "sample_steps": 25, "guidance": 2.0, "p_uncond": 0.1, "ema_decay": 0.99, "channels": [16, 32, 32],
         "kernel": 5, "embed_dim": 32, "groups": 8, "lr": 2e-3, "weight_decay": 0.0, "batch_size": 64, "classes": ["N", "S", "V", "F"]}  # fmt: skip


def test_cosine_schedule_falls_from_one_towards_zero():
    abar = cosine_alpha_bar(1000)
    assert abar.shape == (1000,) and 0.99 < float(abar[0]) <= 1.0 and float(abar[-1]) < 1e-3
    assert bool((abar[1:] < abar[:-1]).all())


def test_unet_keeps_the_beat_length_and_starts_by_predicting_zero():
    net = UNet1D(180, 4, [16, 32, 64], 5, 32, 8)
    x, t, y = torch.randn(3, 180), torch.tensor([0, 10, 999]), torch.tensor([0, 2, 4])
    out = net(x, t, y)
    assert out.shape == (3, 180) and torch.allclose(out, torch.zeros_like(out))
    with pytest.raises(ValueError):
        UNet1D(182, 4, [16, 32, 64], 5, 32, 8)


def _waves(n: int, length: int, seed: int) -> dict:
    """N: flat; S: a positive bump; V: a wide negative wave; F: a small bump. Tiny noise."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1, 1, length)
    shapes = {"N": 0 * grid, "S": 2.0 * np.exp(-((grid / 0.15) ** 2)), "V": -2.0 * np.exp(-((grid / 0.4) ** 2)), "F": 1.0 * np.exp(-((grid / 0.2) ** 2))}
    x, y = [], []
    for c, s in shapes.items():
        x.append(s + 0.05 * rng.standard_normal((n, length)))
        y += [CLASS_INDEX[c]] * n
    m = len(y)
    return {"X": np.concatenate(x).astype(np.float32), "y": np.array(y, np.int8), "record": np.full(m, 100, np.int16), "split": np.array(["train"] * m)}, shapes


def test_interface_determinism_and_finite_beats():
    data, _ = _waves(16, 32, 0)
    gen = build_generator("diffusion", SMALL, 32, seed=0)
    fit = gen.fit(data, steps=3, seed=0, fractions=(1.0,), log_every=1)
    assert not fit.diverged and fit.steps_done == 3 and len(fit.checkpoints) == 1
    a, b = gen.sample("S", 5, seed=7), gen.sample("S", 5, seed=7)
    assert a.shape == (5, 32) and np.isfinite(a).all() and np.array_equal(a, b)
    assert not np.array_equal(a, gen.sample("S", 5, seed=8))
    assert gen.n_parameters() == sum(p.numel() for p in gen.denoiser.parameters())  # the moving average is not trained
    restored = build_generator("diffusion", SMALL, 32, seed=1)
    restored.load_state_dict(gen.state_dict())
    assert np.array_equal(restored.sample("V", 4, seed=3), gen.sample("V", 4, seed=3))


def test_training_and_guided_sampling_reproduce_each_class_waveform():
    torch.manual_seed(0)
    data, shapes = _waves(64, 32, 1)
    gen = build_generator("diffusion", SMALL, 32, seed=0)
    fit = gen.fit(data, steps=600, seed=0, fractions=(1.0,), log_every=100)
    assert not fit.diverged
    assert fit.history[-1]["loss"] < fit.history[0]["loss"]
    for cls in ("S", "V", "F"):
        mean = gen.sample(cls, 64, seed=11).mean(axis=0)
        errors = {c: float(np.abs(mean - s).mean()) for c, s in shapes.items()}
        assert min(errors, key=errors.get) == cls, (cls, errors)  # nearer its own class waveform than any other
        assert errors[cls] < 0.35, (cls, errors)
