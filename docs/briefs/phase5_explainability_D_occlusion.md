# Brief: phase5_explainability, variant D (occlusion)

## Definition

Occlusion with a sliding 50 ms window (stride 25 ms), zero baseline.

Settings of this variant (everything else is shared):

```yaml
method: occlusion
cost: 4
```

## Interface and what already exists

- Config: `configs/phase5_explainability/D_occlusion.yaml`, merged with the shared `configs/phase5_explainability/shared_settings.yaml` by the runner, which rejects a variant that
  tries to change anything shared (CLAUDE.md rule 7).
- Runner: `python -m ecgaug.experiments.phase5_explainability` (`src/ecgaug/experiments/phase5_explainability.py`).
- There is no variant-specific code to write. Your job: check the config matches the definition above, run it,
  sanity-check the outputs and write the notes file. If shared code needs a change, describe it so it goes through `main`.

## Shared settings (identical for A-D)

```yaml
experiment: phase5_explainability
conditions:
- none
- cvae
- wgan_gp
beats_per_class: 200
sample_seed: 0
deletion_fractions:
- 0.0
- 0.1
- 0.2
- 0.3
- 0.4
- 0.5
random_orders: 3
synthetic_beats_per_class: 200
method_settings:
  ig_steps: 50
  smoothgrad_samples: 25
  smoothgrad_sd: 0.1
  occlusion_window_ms: 50
  occlusion_stride_ms: 25
  batch_size: 64
```

## How to run

```
uv run python -m ecgaug.experiments.phase5_explainability --config configs/phase5_explainability/D_occlusion.yaml --smoke      # fake data, end to end
uv run python -m ecgaug.experiments.phase5_explainability --config configs/phase5_explainability/D_occlusion.yaml              # the official run
```

On Kaggle: import `kaggle/phase5_explainability.ipynb`, set `VARIANTS = ["D"]`, `GIT_NAME`, `GIT_EMAIL`, then Run All. Attach the artifacts dataset (the grid's zips) and set `ARTIFACTS_DIR`.

`python -m ecgaug.workflow status` says whether the gates this run needs are approved; an official run refuses
otherwise and says why.

## What gets logged

- Run folders under `runs/phase5_explainability/D_occlusion/` (config, environment, logs, metrics; weights and predictions stay out of git).
- `results/phase5_explainability/D.json`: written by the runner; do not edit by hand.
- `results/phase5_explainability/D_notes.md` (if the runner creates it): you write it.
- Figures under `figures/phase5_explainability/D/`.

## Sanity checks before you report

- Needs the phase 4 gate's checkpoints on disk (ingest the grid's Kaggle zips).
- The class-average saliency figure: does the map sit on the QRS for V beats, near the P wave for S beats?
- Faithfulness gain should be above 0; a negative gain means the map points away from what the model uses.
- No test data: nothing here can read DS2, and you must not try.

## Selection rule

Per architecture: the mean faithfulness gain over the three models (none, cVAE, WGAN-GP); ties within the margin by stability, then the cheaper method. Applied at the phase 5 gate.
