# Brief: Module 1, variant D (short window at 125 Hz)

## Definition

Variant A resampled to 125 Hz (63 samples, 31 before and 32 after the R peak).

Only the input representation differs between the four variants. The beat set, labels, split, model,
seeds and budget are identical, and the test suite checks that the beat set is the same for all four.

## Interface and what already exists

- The representation is defined by `configs/m1_input_representation/D_short_window_125hz.yaml` and executed by shared, tested code:
  `RepresentationSpec` and `transform_record` in `src/ecgaug/data/representation.py`.
- The runner is `src/ecgaug/experiments/input_representation.py`. It builds the candidate dataset from DS1 records only,
  trains the probe, logs every run and writes the result file.
- There is no variant-specific code to write. Your job in `/variant 1 D` is to check that the config
  matches the definition above, run it, sanity-check the outputs and write the notes.
- If shared code needs a change, stop and describe it so it can go through `main`.

## Budget (identical for A–D, from `configs/m1_input_representation/shared_budget.yaml`)

| item | value |
|---|---|
| seeds | 0, 1, 2 |
| learning-rate grid | 1e-3 and 3e-4, both trained for every seed |
| epochs | at most 40, early stopping with patience 5 on the selection metric |
| optimiser | AdamW, weight decay 0.01, batch 256, unweighted cross-entropy |
| model | probe CNN (fixed) |
| device | CPU by default; a different device is recorded in `env.json` |

A variant file that tries to override any of this is rejected by the runner.

## How to run

```
git switch -c variant/m1-D main
uv sync
uv run python -m ecgaug.experiments.input_representation --config configs/m1_input_representation/D_short_window_125hz.yaml --check    # DS1 inputs, no training
uv run python -m ecgaug.experiments.input_representation --config configs/m1_input_representation/D_short_window_125hz.yaml --smoke    # fake data, end to end
uv run python -m ecgaug.experiments.input_representation --config configs/m1_input_representation/D_short_window_125hz.yaml            # the experiment
```

On Kaggle: follow `kaggle/README.md` with `GIT_REF = "variant/m1-D"` and
`CONFIGS = ["configs/m1_input_representation/D_short_window_125hz.yaml"]`, then `python kaggle/ingest_runs.py <zip>` locally.

The experiment needs the approved validation split (`configs/validation_split/approved.yaml`) and a committed tree. Wall-clock has not been
measured yet; the first run reports it.

## What gets logged

- `runs/<run_id>/` for each of the 6 runs (2 learning rates × 3 seeds): `config.yaml`, `env.json`,
  `train_log.csv`, `metrics_val.json`, `preds_val.npz`, `checkpoints/best.pt`.
- `results/runs_index.csv`: one row per run.
- `results/validation/m1_input_representation/D.json`: per-seed metrics, mean ± SD, run IDs, wall-clock, both learning
  rates, budget deviations. Written by the runner; do not edit by hand.
- `results/validation/m1_input_representation/D_notes.md`: you write this (what worked, what failed, surprises, figures).

## Sanity checks before you report

- Input shape is `[n, 63]`, no NaN or infinite values, per-beat mean 0 and SD 1 (`--check` prints these).
- Validation contains every class the selection metric needs.
- Learning curves in `train_log.csv`: training loss falls; note the best epoch and where each seed stopped.
- If a seed stops at the patience limit with a selection value of 0, report it plainly. Do not rerun
  with other settings.
- No test data: the runner cannot read DS2, and you must not try.

## Selection rule (PREREGISTRATION §4–5)

Seed-mean validation selection metric; variants within the tie margin max(0.02, pooled seed SD) of the
best are tied; under the fallback metric ties are broken by V F1; then fewer input values per beat wins.
Variant D has **63** input values per beat (D 63, A 180, C 183, B 360). The rule is applied at
`/gate 1` by `ecgaug.eval.selection.select_winner`, not by you.

## Notes specific to D

- The continuous record is resampled with a polyphase filter (`scipy.signal.resample_poly`, factor 25/72) before windowing, so the anti-aliasing filter never sees a window edge. The R index maps to `round(r * 125 / 360)`.
- D has the fewest input values per beat, so it wins any tie under the Module 1 rule.
