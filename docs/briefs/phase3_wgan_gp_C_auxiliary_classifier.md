# Brief: phase3_wgan_gp, variant C (auxiliary classifier)

## Definition

AC-WGAN-GP, an auxiliary class head on the critic trained on real beats; the generator is also rewarded for beats classified as the intended class.

Settings of this variant (everything else is shared):

```yaml
generator:
  conditioning: auxiliary
```

## Interface and what already exists

- Config: `configs/phase3_wgan_gp/C_auxiliary_classifier.yaml`, merged with the shared `configs/phase3_wgan_gp/generator_settings.yaml` by the runner, which rejects a variant that
  tries to change anything shared (CLAUDE.md rule 7).
- Runner: `python -m ecgaug.experiments.phase3_wgan_gp` (`src/ecgaug/experiments/phase3_wgan_gp.py`).
- There is no variant-specific code to write. Your job: check the config matches the definition above, run it,
  sanity-check the outputs and write the notes file. If shared code needs a change, describe it so it goes through `main`.

## Shared settings (identical for A-D)

```yaml
experiment: phase3_wgan_gp
family: wgan_gp
classes:
- N
- S
- V
- F
training:
  batch_size: 128
  lr: 0.0001
  adam_betas:
  - 0.0
  - 0.9
  n_critic: 5
  gradient_penalty: 10.0
  pilot_steps: 5000
  final_steps: 20000
  log_every: 50
generator_defaults:
  conditioning: concat
  noise_dim: 64
  embed_dim: 16
  generator_channels:
  - 128
  - 64
  - 32
  critic_channels:
  - 32
  - 64
  - 128
  kernel: 7
  auxiliary_weight: 1.0
pilot_seeds:
- 0
- 1
final_seeds:
- 0
- 1
- 2
checkpoint_fractions:
- 0.25
- 0.5
- 0.75
- 1.0
pilot:
  policy_fraction_of_n: 0.5
  classifier: cnn
  classifier_seed: 0
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
diagnostics:
  max_beats: 1000
  memorization_percentile: 5
  c2st_folds: 5
hard_gates:
  diversity_min_ratio: 0.7
  memorization_max_fraction: 0.1
bank:
  size: balance_to_n
```

## How to run

```
uv run python -m ecgaug.experiments.phase3_wgan_gp --config configs/phase3_wgan_gp/C_auxiliary_classifier.yaml --smoke      # fake data, end to end
uv run python -m ecgaug.experiments.phase3_wgan_gp --config configs/phase3_wgan_gp/C_auxiliary_classifier.yaml --preview    # before the gates it needs; provisional
uv run python -m ecgaug.experiments.phase3_wgan_gp --config configs/phase3_wgan_gp/C_auxiliary_classifier.yaml              # the official run
uv run python -m ecgaug.experiments.phase3_wgan_gp --config configs/phase3_wgan_gp/C_auxiliary_classifier.yaml --final      # only if the phase 3a gate chose this variant
```

On Kaggle: import `kaggle/phase3_wgan_gp.ipynb`, set `VARIANTS = ["C"]`, `GIT_NAME`, `GIT_EMAIL`, then Run All.

`python -m ecgaug.workflow status` says whether the gates this run needs are approved; an official run refuses
otherwise and says why.

## What gets logged

- Run folders under `runs/phase3_wgan_gp/C_auxiliary_classifier/` (config, environment, logs, metrics; weights and predictions stay out of git).
- `results/phase3_wgan_gp/C.json`: written by the runner; do not edit by hand.
- `results/phase3_wgan_gp/C_notes.md` (if the runner creates it): you write it.
- Figures under `figures/phase3_wgan_gp/C/`.

## Sanity checks before you report

- Losses in `train_log.csv` (figure `01_generator_training.png`): the Wasserstein estimate should settle, the gradient penalty stay small.
- Every checkpoint in `checkpoints.csv` (figure `02_checkpoints.png`): diversity, memorization, C2ST and the pilot score; watch for mode collapse (diversity falling).
- Real versus synthetic beats (figure `03_real_vs_synthetic.png`) and nearest real beats (`05_...`).
- A diverged seed is a result: report it, do not rerun with other settings.
- No test data: nothing here can read DS2, and you must not try.

## Selection rule

As phase3_cvae (hard gates, then the pilot score, then the composite quality rank, then fewer parameters). Fallback if no WGAN-GP variant passes by Day 12: the team restricts the classes or adopts variant D, and records the failure.
