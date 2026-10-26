# Brief: phase4_augmentation_policy, variant C (multiple of real)

## Definition

Add a fixed multiple of each minority class's real training count.

Settings of this variant (everything else is shared):

```yaml
policy:
  kind: multiple_of_real
pilot_strength: 2.0
grid_strengths:
- 1.0
- 2.0
- 4.0
complexity: 2
```

## Interface and what already exists

- Config: `configs/phase4_augmentation_policy/C_multiple_of_real.yaml`, merged with the shared `configs/phase4_augmentation_policy/pilot_settings.yaml` by the runner, which rejects a variant that
  tries to change anything shared (CLAUDE.md rule 7).
- Runner: `python -m ecgaug.experiments.phase4_augmentation_policy` (`src/ecgaug/experiments/phase4_augmentation_policy.py`).
- There is no variant-specific code to write. Your job: check the config matches the definition above, run it,
  sanity-check the outputs and write the notes file. If shared code needs a change, describe it so it goes through `main`.

## Shared settings (identical for A-D)

```yaml
experiment: phase4_augmentation_policy
classifier: cnn
seed: 0
banks:
  cvae: phase3_gate/phase3_cvae
  wgan_gp: phase3_gate/phase3_wgan_gp
max_n_recall_drop: 0.01
preview_classifier:
  model:
    name: probe_cnn
    channels:
    - 16
    - 32
    - 64
    kernel: 7
    side_hidden: 16
  lr: 0.001
  max_epochs: 40
  patience: 5
  batch_size: 256
  weight_decay: 0.01
```

## How to run

```
uv run python -m ecgaug.experiments.phase4_augmentation_policy --config configs/phase4_augmentation_policy/C_multiple_of_real.yaml --smoke      # fake data, end to end
uv run python -m ecgaug.experiments.phase4_augmentation_policy --config configs/phase4_augmentation_policy/C_multiple_of_real.yaml --preview    # before the gates it needs; provisional
uv run python -m ecgaug.experiments.phase4_augmentation_policy --config configs/phase4_augmentation_policy/C_multiple_of_real.yaml              # the official run
```

On Kaggle: import `kaggle/phase4_augmentation_policy.ipynb`, set `VARIANTS = ["C"]`, `GIT_NAME`, `GIT_EMAIL`, then Run All. Attach the banks dataset and set `BANKS_DATASET_DIR`.

`python -m ecgaug.workflow status` says whether the gates this run needs are approved; an official run refuses
otherwise and says why.

## What gets logged

- Run folders under `runs/phase4_augmentation_policy/C_multiple_of_real/` (config, environment, logs, metrics; weights and predictions stay out of git).
- `results/phase4_augmentation_policy/C.json`: written by the runner; do not edit by hand.
- `results/phase4_augmentation_policy/C_notes.md` (if the runner creates it): you write it.
- Figures under `figures/phase4_augmentation_policy/C/`.

## Sanity checks before you report

- The reference run (no augmentation) is shared by all variants: check its run ID is the same in every result file.
- N recall of each bank's run against the reference: the limit is 0.01.
- How many beats each run added (`n_added` in the run's config.yaml).
- No test data: nothing here can read DS2, and you must not try.

## Selection rule

Variants that lower validation N recall by more than 0.01 (mean over both banks) are out; the best mean validation S F1 over both banks wins; ties within 0.02 go to the simpler policy (A 1, B 2, C 2, D 3). Applied at the phase 4a gate, whose approval writes the full grid.
