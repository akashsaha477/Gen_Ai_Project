# Rare-arrhythmia detection with generative ECG augmentation

Does class-conditional generative augmentation (cVAE, WGAN-GP) improve detection of rare arrhythmia
classes on MIT-BIH when training and test patients are kept apart? A 30-day project (2026-10-07 to
2026-11-05) run phase by phase with Claude Code.

Start with `CLAUDE.md` (rules and data protocol), then `docs/PLAN.md`, `docs/PAPER_TRACK.md` and
`docs/PREREGISTRATION.md`. Live state is in `docs/PROGRESS.md` and `docs/DECISIONS.md`.

## Setup

```
uv sync                              # creates .venv with the versions pinned in uv.lock
uv run pytest                        # leakage, test lock, contract, metrics, bootstrap, runner
uv run python -m ecgaug.data.raw     # download MIT-BIH from PhysioNet and verify checksums
uv run python -m ecgaug.data.counts  # per-record class counts for DS1
```

Open Claude Code in this directory (not its parent) so `CLAUDE.md` and the `/phase`, `/variant`,
`/gate`, `/status`, `/final-eval` and `/paper` commands load. See `KICKOFF_PROMPT.md` for the phase loop.

## The test set is locked

DS2 is read only through `ecgaug.data.load_test()`, which raises unless `ALLOW_TEST=1`, and only
`/final-eval` sets that. Counting DS2 labels and building `test.npz` need `ALLOW_TEST_BUILD=1`. Every
access is appended to `docs/TEST_ACCESS_LOG.md`. Do not set either flag by hand outside those steps.

## Running a variant

```
uv run python -m ecgaug.experiments.m1 --config configs/m1/A.yaml --check   # DS1 inputs, no training
uv run python -m ecgaug.experiments.m1 --config configs/m1/A.yaml --smoke   # fake data, end to end
uv run python -m ecgaug.experiments.m1 --config configs/m1/A.yaml           # the experiment
```

The experiment refuses to start until the pre-registration is in force, the validation split is
approved and the tree is committed. To run on Kaggle, follow `docs/KAGGLE.md`.

## Troubleshooting

- **`KeyError: '_items'` on `import matplotlib.pyplot` (macOS):** the system font listing returns
  nothing on some macOS versions. Call `ecgaug.eval.plotting.import_pyplot()` once; it builds
  matplotlib's font cache without that step, after which normal imports work.
