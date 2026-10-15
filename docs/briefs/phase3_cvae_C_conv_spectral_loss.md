# Brief: phase3_cvae, variant C (conv spectral loss)

## Definition

As A plus a spectral reconstruction term, the L1 distance between FFT magnitudes of the beat and its reconstruction.

Settings of this variant (everything else is shared):

```yaml
generator:
  spectral_weight: 1.0
```

## Interface and what already exists

- Config: `configs/phase3_cvae/C_conv_spectral_loss.yaml`, merged with the shared `configs/phase3_cvae/generator_settings.yaml` by the runner, which rejects a variant that
  tries to change anything shared (CLAUDE.md rule 7).
- Runner: `python -m ecgaug.experiments.phase3_cvae` (`src/ecgaug/experiments/phase3_cvae.py`).
- There is no variant-specific code to write. Your job: check the config matches the definition above, run it,
  sanity-check the outputs and write the notes file. If shared code needs a change, describe it so it goes through `main`.

## Shared settings (identical for A-D)

```yaml
experiment: phase3_cvae
family: cvae
classes:
- N
- S
- V
- F
training:
  batch_size: 128
  lr: 0.001
  pilot_steps: 5000
  final_steps: 20000
  log_every: 50
generator_defaults:
  architecture: conv
  channels:
  - 32
  - 64
  - 128
  kernel: 7
  latent_dim: 16
  beta: 1.0
  kl_anneal_fraction: 0.3
  spectral_weight: 0.0
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
uv run python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/C_conv_spectral_loss.yaml --smoke      # fake data, end to end
uv run python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/C_conv_spectral_loss.yaml --preview    # before the gates it needs; provisional
uv run python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/C_conv_spectral_loss.yaml              # the official run
uv run python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/C_conv_spectral_loss.yaml --final      # only if the phase 3a gate chose this variant
```

On Kaggle: import `kaggle/phase3_cvae.ipynb`, set `VARIANTS = ["C"]`, `GIT_NAME`, `GIT_EMAIL`, then Run All.

`python -m ecgaug.workflow status` says whether the gates this run needs are approved; an official run refuses
otherwise and says why.

## What gets logged

- Run folders under `runs/phase3_cvae/C_conv_spectral_loss/` (config, environment, logs, metrics; weights and predictions stay out of git).
- `results/phase3_cvae/C.json`: written by the runner; do not edit by hand.
- `results/phase3_cvae/C_notes.md` (if the runner creates it): you write it.
- Figures under `figures/phase3_cvae/C/`.

## Sanity checks before you report

- Losses in `train_log.csv` (figure `01_generator_training.png`): reconstruction falls; KL rises during the warm-up.
- Every checkpoint in `checkpoints.csv` (figure `02_checkpoints.png`): diversity, memorization, C2ST and the pilot score.
- Real versus synthetic beats (figure `03_real_vs_synthetic.png`) and synthetic beats next to their nearest real beat (`05_...`): copies?
- A diverged seed is a result: report it, do not rerun with other settings.
- No test data: nothing here can read DS2, and you must not try.

## Selection rule

Hard gates (no divergence on either pilot seed; diversity >= 0.7 x real and memorization <= 10% for S, V and F on both seeds), then the mean pilot S F1 at the final checkpoint; ties within max(0.02, pooled SD) go to the composite quality rank (V F1 until it exists), then fewer parameters. Applied at the phase 3a gate (`python -m ecgaug.workflow phase3a-gate`).
