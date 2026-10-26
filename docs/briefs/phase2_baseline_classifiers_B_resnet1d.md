# Brief: phase2_baseline_classifiers, variant B (resnet1d)

## Definition

1D-ResNet, stem convolution and four residual stages, global average pooling, linear head.

Settings of this variant (everything else is shared):

```yaml
role: cnn
model:
  name: resnet1d
  channels:
  - 32
  - 64
  - 128
  - 128
  kernel: 7
  side_hidden: 16
```

## Interface and what already exists

- Config: `configs/phase2_baseline_classifiers/B_resnet1d.yaml`, merged with the shared `configs/phase2_baseline_classifiers/shared_budget.yaml` by the runner, which rejects a variant that
  tries to change anything shared (CLAUDE.md rule 7).
- Runner: `python -m ecgaug.experiments.phase2_baseline_classifiers` (`src/ecgaug/experiments/phase2_baseline_classifiers.py`).
- There is no variant-specific code to write. Your job: check the config matches the definition above, run it,
  sanity-check the outputs and write the notes file. If shared code needs a change, describe it so it goes through `main`.

## Shared settings (identical for A-D)

```yaml
experiment: phase2_baseline_classifiers
seeds:
- 0
- 1
- 2
lr_grid:
- 0.001
- 0.0003
max_epochs: 40
patience: 5
min_epochs: 0
batch_size: 256
weight_decay: 0.01
device: auto
```

## How to run

```
uv run python -m ecgaug.experiments.phase2_baseline_classifiers --config configs/phase2_baseline_classifiers/B_resnet1d.yaml --smoke      # fake data, end to end
uv run python -m ecgaug.experiments.phase2_baseline_classifiers --config configs/phase2_baseline_classifiers/B_resnet1d.yaml --preview    # before the gates it needs; provisional
uv run python -m ecgaug.experiments.phase2_baseline_classifiers --config configs/phase2_baseline_classifiers/B_resnet1d.yaml              # the official run
```

On Kaggle: import `kaggle/phase2_baseline_classifiers.ipynb`, set `VARIANTS = ["B"]`, `GIT_NAME`, `GIT_EMAIL`, then Run All.

`python -m ecgaug.workflow status` says whether the gates this run needs are approved; an official run refuses
otherwise and says why.

## What gets logged

- Run folders under `runs/phase2_baseline_classifiers/B_resnet1d/` (config, environment, logs, metrics; weights and predictions stay out of git).
- `results/phase2_baseline_classifiers/B.json`: written by the runner; do not edit by hand.
- `results/phase2_baseline_classifiers/B_notes.md` (if the runner creates it): you write it.
- Figures under `figures/phase2_baseline_classifiers/B/`.

## Sanity checks before you report

- Learning curves in `train_log.csv`: training loss falls; note the best epoch and where each seed stopped.
- Validation confusion matrix (figure `training_and_confusion.png`): where do S beats go?
- If a seed keeps epoch 1 with S F1 0, report it plainly (docs/PHASE1_REPORT.md §6); do not rerun with other settings.
- No test data: nothing here can read DS2, and you must not try.

## Selection rule

Seed-mean validation selection metric; ties within max(0.02, pooled SD) go to fewer parameters. The `cnn` is chosen from A and B, the `rnn` from C and D, at the phase 2 gate (`python -m ecgaug.workflow phase2-gate`).
