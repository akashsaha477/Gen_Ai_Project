"""P12: the generators' input scale, the bank size, the patient-relative bank and the grid (hand-made cases)."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecgaug.data.records import CLASS_INDEX
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.paths import repo_root


class ConstantGenerator:
    """Stand-in generator: every beat of class c is the constant (index of c + 1) / 10."""

    def sample(self, cls: str, n: int, seed: int) -> np.ndarray:
        return np.full((n, 4), (CLASS_INDEX[cls] + 1) / 10, dtype=np.float32)


def _train() -> dict:
    y = np.array([0, 0, 0, 0, 1, 2, 3], dtype=np.int8)
    n = len(y)
    return {"X": np.zeros((n, 4), dtype=np.float32), "y": y, "record": np.arange(n, dtype=np.int16), "split": np.array(["train"] * n),
            "r_idx": np.arange(n, dtype=np.int32), "rr_prev": np.arange(n, dtype=np.float32) + 0.5, "rr_next": np.arange(n, dtype=np.float32) + 0.6,
            "rr_ratio": np.arange(n, dtype=np.float32) + 0.7, "is_synthetic": np.zeros(n, bool), "source": np.array(["real"] * n)}  # fmt: skip


def test_scale_is_the_sd_of_every_training_sample():
    x = np.array([[1.0, -1.0], [3.0, -3.0]])
    assert rg.scale_of(x) == pytest.approx(np.sqrt(5.0))


def test_bank_holds_what_the_largest_strength_adds():
    y = np.array([0] * 100 + [1] * 10 + [2] * 60 + [3] * 2)
    assert rg.bank_counts(y, 0.5) == {"S": 40, "V": 0, "F": 48}  # up to 0.5 x 100 N beats; V already has 60


def test_relative_bank_is_scaled_back_and_not_zscored_per_beat():
    train = _train()
    bank = rg.sample_relative_bank(ConstantGenerator(), train, {"S": 3, "V": 1, "F": 2}, seed=0, scale=2.0, source="p12")
    y = np.asarray(bank["y"])
    assert [int((y == CLASS_INDEX[c]).sum()) for c in ("S", "V", "F")] == [3, 1, 2]
    s = np.asarray(bank["X"])[y == CLASS_INDEX["S"]]
    assert np.allclose(s, 0.4)  # (1 + 1) / 10 x scale 2; a per-beat z-score would have made it 0
    assert np.asarray(bank["is_synthetic"]).all() and set(np.asarray(bank["source"]).tolist()) == {"p12"}
    f = y == CLASS_INDEX["F"]
    assert np.allclose(np.asarray(bank["rr_prev"])[f], 6.5)  # copied from the one real F beat (index 6)


def test_incart_records_become_their_patients_and_other_records_stay():
    units = rg.patient_units(np.array([1, 2, 3, 101, 805]), incart={1: 7, 2: 7, 3: 9})
    assert units.tolist() == [1007, 1007, 1009, 101, 805]


def test_grid_trains_the_cnn_on_both_new_banks_at_the_phase4_strengths_unscreened(monkeypatch):
    monkeypatch.delenv("ECGAUG_ROOT")  # read-only: the real configs and v6's data record
    cfg = rg.load_config(repo_root() / "configs/phase6_relative_generators/relative_generators.yaml")
    grid = rg.relative_grid(cfg)
    assert list(grid["architectures"]) == ["cnn"] and grid["policy"] == {"kind": "fraction_of_n"}
    assert {c: v["bank"] for c, v in grid["conditions"].items()} == {"cvae": "phase6_relative_generators/phase3_cvae-D",
                                                                      "wgan_gp": "phase6_relative_generators/phase3_wgan_gp-A"}  # fmt: skip
    assert all(v["strengths"] == rg.strengths() for v in grid["conditions"].values())


def test_config_must_name_both_families(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment: phase6_relative_generators\nmore_patients_config: x\ngenerators: {cvae: y}\nmechanism_beats: 1\n")
    with pytest.raises(ValueError):
        rg.load_config(bad)


@pytest.mark.parametrize("family", ["cvae", "wgan_gp"])
def test_train_generators_end_to_end_on_fake_beats(family, monkeypatch):
    """Two steps of the real generator on fake patient-relative beats: a checkpoint, a bank at the generator's scale and the record."""
    from types import SimpleNamespace

    from ecgaug.experiments import generator_runs as gr
    from ecgaug.generators import bank as banks

    real = Path(__file__).resolve().parents[1]
    files = {"cvae": "configs/phase3_cvae/D_residual_film.yaml", "wgan_gp": "configs/phase3_wgan_gp/A_concat_embedding.yaml"}
    g = gr.load_config(real / files[family])
    g["training"]["final_steps"], g["final_seeds"] = 2, [0]
    cfg = {"generators": files, "gen": {family: g}}
    monkeypatch.setattr(rg, "strengths", lambda: [0.1, 0.25, 0.5])
    rng = np.random.default_rng(0)
    n, length = 64, 256
    y = np.array([0] * 40 + [1] * 8 + [2] * 8 + [3] * 8, dtype=np.int8)
    split = np.array(["train"] * n)
    split[::8] = "val"  # every class keeps training beats
    arrays = {"X": (0.3 * rng.standard_normal((n, length))).astype(np.float32), "y": y, "record": np.where(split == "val", 900, 100).astype(np.int16),
              "split": split, "r_idx": np.arange(n, dtype=np.int32), "rr_prev": np.full(n, 0.8, np.float32), "rr_next": np.full(n, 0.8, np.float32),
              "rr_ratio": np.ones(n, np.float32), "is_synthetic": np.zeros(n, bool), "source": np.array(["real"] * n)}  # fmt: skip
    data = SimpleNamespace(arrays=arrays, spec=SimpleNamespace(n_samples=length), info={"content_sha256": "fake"})
    record = rg.train_generators(cfg, family, data, "cpu")
    entry = record["seeds"]["0"]
    assert not entry["diverged"] and entry["steps_done"] == 2
    train_y = y[split == "train"]
    assert entry["bank_counts"] == rg.bank_counts(train_y, 0.5)
    bank = banks.load_bank(rg.generator_id(cfg, family), 0, expected_sha256=entry["bank_sha256"])
    assert np.asarray(bank["X"]).shape[1] == length and np.asarray(bank["is_synthetic"]).all()
    meta = json.loads((banks.bank_dir(rg.generator_id(cfg, family)) / banks.BANK_FILE).read_text())
    assert "not" not in meta["normalization"] and "scale" in meta["normalization"]
    assert rg.generators_record(family)["scale"] == pytest.approx(rg.scale_of(arrays["X"][split == "train"]))
    from ecgaug import runs

    run_dir = runs.find_run_dir(entry["run_id"])
    assert all((run_dir / f).is_file() for f in runs.REQUIRED_RUN_FILES) and (run_dir / "checkpoints" / "final.pt").is_file()
