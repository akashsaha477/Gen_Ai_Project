"""Phase 3 generators: interface, train-only rule, determinism, checkpoints, divergence, banks."""

import json

import numpy as np
import pytest

from ecgaug import paths
from ecgaug.data.contract import FIELDS
from ecgaug.data.records import CLASS_INDEX
from ecgaug.generators import GENERATED_CLASSES, TrainOnlyError, build_generator
from ecgaug.generators.bank import BankError, bank_counts, export_bank, load_bank, sample_bank

L = 63
CVAE = {"classes": ["N", "S", "V", "F"], "batch_size": 16, "lr": 1e-3, "architecture": "conv", "channels": [8, 16, 16],
        "kernel": 5, "latent_dim": 4, "beta": 1.0, "kl_anneal_fraction": 0.3, "spectral_weight": 0.0}  # fmt: skip
WGAN = {"classes": ["N", "S", "V", "F"], "batch_size": 16, "lr": 1e-4, "adam_betas": [0.0, 0.9], "n_critic": 2,
        "gradient_penalty": 10.0, "noise_dim": 8, "embed_dim": 4, "generator_channels": [16, 16, 8],
        "critic_channels": [8, 16, 16], "kernel": 5, "conditioning": "concat", "auxiliary_weight": 1.0}  # fmt: skip
SETTINGS = [
    ("cvae", CVAE), ("cvae", {**CVAE, "spectral_weight": 1.0}), ("cvae", {**CVAE, "architecture": "residual_film"}),
    *[("wgan_gp", {**WGAN, "conditioning": c}) for c in ("concat", "projection", "auxiliary", "per_class")],
]  # fmt: skip


def training_part(n=200, seed=0):
    """Contract-format training beats: each class a bump of its own width, plus noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(4), n // 4).astype(np.int8)
    t = np.linspace(-1, 1, L)
    x = np.exp(-0.5 * (t[None, :] / (0.05 + 0.05 * y[:, None])) ** 2) + 0.05 * rng.standard_normal((len(y), L))
    return {
        "X": x.astype(np.float32), "y": y, "record": np.full(len(y), 101, dtype=np.int16), "split": np.full(len(y), "train", dtype="<U5"),
        "r_idx": np.arange(len(y), dtype=np.int32), "rr_prev": rng.uniform(0.5, 1.2, len(y)).astype(np.float32),
        "rr_next": rng.uniform(0.5, 1.2, len(y)).astype(np.float32), "rr_ratio": rng.uniform(0.6, 1.4, len(y)).astype(np.float32),
        "is_synthetic": np.zeros(len(y), dtype=bool), "source": np.full(len(y), "real", dtype="<U4"),
    }  # fmt: skip


IDS = ["cvae-conv", "cvae-spectral", "cvae-film", "wgan-concat", "wgan-projection", "wgan-auxiliary", "wgan-per-class"]


@pytest.mark.parametrize("family,settings", SETTINGS, ids=IDS)
def test_every_variant_trains_checkpoints_and_samples_deterministically(family, settings):
    part = training_part()
    calls = []
    gen = build_generator(family, settings, L, seed=3)
    fit = gen.fit(part, steps=8, seed=3, on_checkpoint=lambda k, step, g: calls.append((k, step)) or {"seen": k}, log_every=2)
    assert not fit.diverged and fit.steps_done == 8
    assert calls == [(1, 2), (2, 4), (3, 6), (4, 8)] and [c["seen"] for c in fit.checkpoints] == [1, 2, 3, 4]
    assert fit.history and all(np.isfinite(v) for row in fit.history for v in row.values())
    beats = gen.sample("F", 5, seed=1)
    assert beats.shape == (5, L) and beats.dtype == np.float32 and np.isfinite(beats).all()
    np.testing.assert_array_equal(beats, gen.sample("F", 5, seed=1))
    assert not np.array_equal(beats, gen.sample("F", 5, seed=2))
    again = build_generator(family, settings, L, seed=3)
    again.fit(part, steps=8, seed=3, log_every=2)
    np.testing.assert_array_equal(beats, again.sample("F", 5, seed=1))


def test_generators_refuse_beats_from_outside_the_training_split():
    part = training_part()
    part["split"][:3] = "val"
    with pytest.raises(TrainOnlyError, match="not from the training split"):
        build_generator("cvae", CVAE, L, seed=0).fit(part, steps=2, seed=0)
    part = training_part()
    with pytest.raises(TrainOnlyError, match="validation records"):
        build_generator("cvae", CVAE, L, seed=0).fit(part, steps=2, seed=0, val_records=(101,))
    del part["split"]
    with pytest.raises(TrainOnlyError):
        build_generator("cvae", CVAE, L, seed=0).fit(part, steps=2, seed=0)


def test_batches_are_balanced_and_per_class_gans_take_turns():
    gen = build_generator("wgan_gp", {**WGAN, "conditioning": "per_class"}, L, seed=0)
    seen = []

    def spy(draw, step, total):
        seen.append(sorted(set(draw()[1].tolist())))
        return {"loss": 0.0}

    gen.train_step = spy
    gen.fit(training_part(), steps=6, seed=0)
    order = [CLASS_INDEX[c] for c in GENERATED_CLASSES]
    assert seen == [[order[i % 3]] for i in range(6)]
    assert gen.sample_classes() == GENERATED_CLASSES
    with pytest.raises(ValueError, match="cannot sample class 'N'"):
        gen.sample("N", 2, seed=0)

    cvae = build_generator("cvae", CVAE, L, seed=0)
    counts = []
    cvae.train_step = lambda draw, step, total: counts.append(np.bincount(draw()[1].numpy(), minlength=4).tolist()) or {"loss": 0.0}
    cvae.fit(training_part(), steps=2, seed=0)
    assert counts == [[4, 4, 4, 4], [4, 4, 4, 4]]


def test_divergence_stops_training_and_is_reported():
    gen = build_generator("cvae", CVAE, L, seed=0)
    original = gen.train_step

    def explode(draw, step, total):
        out = original(draw, step, total)
        return {**out, "loss": float("nan")} if step == 3 else out

    gen.train_step = explode
    calls = []
    fit = gen.fit(training_part(), steps=8, seed=0, on_checkpoint=lambda k, s, g: calls.append(k) or {})
    assert fit.diverged and fit.steps_done == 2 and "step 3" in fit.divergence
    assert calls == [1]  # the checkpoint at step 2 was reached, nothing after


def test_bank_has_contract_fields_rr_from_training_and_is_never_overwritten():
    part = training_part()
    gen = build_generator("cvae", CVAE, L, seed=0)
    gen.fit(part, steps=4, seed=0)
    counts = bank_counts(part["y"])
    assert counts == {"S": 0, "V": 0, "F": 0}  # every class already as large as N here
    bank = sample_bank(gen, part, {"S": 7, "V": 0, "F": 5}, seed=2, source="phase3_cvae-A")
    assert set(bank) == set(FIELDS) and len(bank["y"]) == 12
    assert (bank["split"] == "train").all() and bank["is_synthetic"].all() and (bank["record"] == -1).all()
    assert (bank["source"] == "phase3_cvae-A").all()
    np.testing.assert_allclose(bank["X"].mean(axis=1), 0, atol=1e-5)
    np.testing.assert_allclose(bank["X"].std(axis=1), 1, atol=1e-3)
    for cls in ("S", "F"):  # every RR triplet is one real training beat's triplet of the same class
        real = {tuple(r) for r in np.stack([part[f][part["y"] == CLASS_INDEX[cls]] for f in ("rr_prev", "rr_next", "rr_ratio")], 1)}
        synth = np.stack([bank[f][bank["y"] == CLASS_INDEX[cls]] for f in ("rr_prev", "rr_next", "rr_ratio")], 1)
        assert all(tuple(r) in real for r in synth)
    again = sample_bank(gen, part, {"S": 7, "V": 0, "F": 5}, seed=2, source="phase3_cvae-A")
    np.testing.assert_array_equal(bank["X"], again["X"])

    entry = export_bank(bank, "phase3_cvae-A", 0, {"run_id": "x", "train_only_check": "passed"})
    assert entry["counts"] == {"S": 7, "V": 0, "F": 5}
    loaded = load_bank("phase3_cvae-A", 0, expected_sha256=entry["sha256"])
    np.testing.assert_array_equal(loaded["X"], bank["X"])
    with pytest.raises(BankError, match="never overwritten"):
        export_bank(bank, "phase3_cvae-A", 0, {})
    with pytest.raises(BankError, match="does not match the frozen manifest"):
        load_bank("phase3_cvae-A", 0, expected_sha256="0" * 64)
    path = paths.repo_root() / "data" / "synthetic" / "phase3_cvae-A" / "seed0.npz"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(BankError, match="does not match bank.json"):
        load_bank("phase3_cvae-A", 0)
    with pytest.raises(BankError, match="audit"):
        load_bank("audit-phase3_cvae-A", 0)
    meta = json.loads((path.parent / "bank.json").read_text())
    assert "PREREGISTRATION §5" in meta["rr_features"] and "0" in meta["seeds"]


def test_bank_size_is_enough_to_balance_every_minority_class_to_n():
    y = np.array([0] * 100 + [1] * 10 + [2] * 30 + [3] * 5)
    assert bank_counts(y) == {"S": 90, "V": 70, "F": 95}
