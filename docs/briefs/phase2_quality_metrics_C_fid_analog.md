# Brief: phase2_quality_metrics, variant C (fid analog)

## Definition

FID-analog: Frechet distance between Gaussians fitted to frozen feature-network features, relative to real-vs-real.

Settings of this variant (everything else is shared):

```yaml
family: fid_analog
```

## Interface and what already exists

- Config: `configs/phase2_quality_metrics/C_fid_analog.yaml`, merged with the shared `configs/phase2_quality_metrics/metric_settings.yaml` by the runner, which rejects a variant that
  tries to change anything shared (CLAUDE.md rule 7).
- Runner: `python -m ecgaug.experiments.phase2_quality_metrics` (`src/ecgaug/experiments/phase2_quality_metrics.py`).
- There is no variant-specific code to write. Your job: check the config matches the definition above, run it,
  sanity-check the outputs and write the notes file. If shared code needs a change, describe it so it goes through `main`.

## Shared settings (identical for A-D)

```yaml
experiment: phase2_quality_metrics
resamples: 3
degradations:
  noise:
  - 0.1
  - 0.3
  - 1.0
  label_shuffle:
  - 0.25
  - 0.5
  - 1.0
  time_shift:
  - 5
  - 15
  - 40
  smoothing:
  - 3
  - 9
  - 27
pass_rule:
  spearman_min: 0.8
  cv_max: 0.1
metrics:
  dtw_max_beats: 200
  dtw_candidates: 20
  dtw_neighbours: 5
  dtw_window_fraction: 0.1
  spectral_nperseg: 64
  fid_max_beats: 1000
  tstr_classifier:
    model:
      name: probe_cnn
      channels:
      - 16
      - 32
      - 64
      kernel: 7
      side_hidden: 16
    lr: 0.001
    max_epochs: 10
    patience: 3
    batch_size: 256
    weight_decay: 0.01
    seed: 0
feature_net:
  model:
    name: probe_cnn
    channels:
    - 16
    - 32
    - 64
    kernel: 7
    side_hidden: 16
  lr: 0.001
  max_epochs: 10
  patience: 3
  batch_size: 256
  weight_decay: 0.01
  seed: 0
```

## How to run

```
uv run python -m ecgaug.experiments.phase2_quality_metrics --config configs/phase2_quality_metrics/C_fid_analog.yaml --smoke      # fake data, end to end
uv run python -m ecgaug.experiments.phase2_quality_metrics --config configs/phase2_quality_metrics/C_fid_analog.yaml --preview    # before the gates it needs; provisional
uv run python -m ecgaug.experiments.phase2_quality_metrics --config configs/phase2_quality_metrics/C_fid_analog.yaml              # the official run
```

On Kaggle: import `kaggle/phase2_quality_metrics.ipynb`, set `VARIANTS = ["C"]`, `GIT_NAME`, `GIT_EMAIL`, then Run All.

`python -m ecgaug.workflow status` says whether the gates this run needs are approved; an official run refuses
otherwise and says why.

## What gets logged

- Run folders under `runs/phase2_quality_metrics/C_fid_analog/` (config, environment, logs, metrics; weights and predictions stay out of git).
- `results/phase2_quality_metrics/C.json`: written by the runner; do not edit by hand.
- `results/phase2_quality_metrics/C_notes.md` (if the runner creates it): you write it.
- Figures under `figures/phase2_quality_metrics/C/`.

## Sanity checks before you report

- The unchanged held-out beats must score best; the figure `sanity.png` shows the score against severity for each degradation.
- Look at the CV of each degraded set over the 3 resamples; a metric that jumps between resamples is not usable.
- Variant C also writes the frozen feature network (`results/phase2_quality_metrics/feature_net_<hash>.pt`); commit it.
- No test data: nothing here can read DS2, and you must not try.

## Selection rule

A family passes if it ranks the real held-out beats best, gets worse with every degradation (Spearman >= 0.8) and is stable (median CV < 10%). The passing families form the composite quality rank, fixed at the phase 2 gate.
