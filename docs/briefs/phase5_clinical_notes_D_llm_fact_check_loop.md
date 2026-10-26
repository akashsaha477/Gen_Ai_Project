# Brief: phase5_clinical_notes, variant D (llm fact check loop)

## Definition

As B plus the fact checker and a regenerate loop (at most 2 retries, then the template).

Settings of this variant (everything else is shared):

```yaml
generator: D
```

## Interface and what already exists

- Config: `configs/phase5_clinical_notes/D_llm_fact_check_loop.yaml`, merged with the shared `configs/phase5_clinical_notes/shared_settings.yaml` by the runner, which rejects a variant that
  tries to change anything shared (CLAUDE.md rule 7).
- Runner: `python -m ecgaug.experiments.phase5_clinical_notes` (`src/ecgaug/experiments/phase5_clinical_notes.py`).
- There is no variant-specific code to write. Your job: check the config matches the definition above, run it,
  sanity-check the outputs and write the notes file. If shared code needs a change, describe it so it goes through `main`.

## Shared settings (identical for A-D)

```yaml
experiment: phase5_clinical_notes
n_cases: 50
min_low_confidence: 10
low_confidence_threshold: 0.6
case_seed: 0
model:
  architecture: cnn
  condition: best_generative
saliency_method: integrated_gradients
method_settings:
  ig_steps: 50
  smoothgrad_samples: 25
  smoothgrad_sd: 0.1
  occlusion_window_ms: 50
  occlusion_stride_ms: 25
  batch_size: 64
max_retries: 2
```

## How to run

```
uv run python -m ecgaug.experiments.phase5_clinical_notes --config configs/phase5_clinical_notes/D_llm_fact_check_loop.yaml --smoke      # fake data, end to end
uv run python -m ecgaug.experiments.phase5_clinical_notes --config configs/phase5_clinical_notes/D_llm_fact_check_loop.yaml              # the official run
```

On Kaggle: import `kaggle/phase5_clinical_notes.ipynb`, set `VARIANTS = ["D"]`, `GIT_NAME`, `GIT_EMAIL`, then Run All. For B-D add the `ANTHROPIC_API_KEY` Kaggle secret. Attach the artifacts dataset (the grid's zips) and set `ARTIFACTS_DIR`.

`python -m ecgaug.workflow status` says whether the gates this run needs are approved; an official run refuses
otherwise and says why.

## What gets logged

- Run folders under `runs/phase5_clinical_notes/D_llm_fact_check_loop/` (config, environment, logs, metrics; weights and predictions stay out of git).
- `results/phase5_clinical_notes/D.json`: written by the runner; do not edit by hand.
- `results/phase5_clinical_notes/D_notes.md` (if the runner creates it): you write it.
- Figures under `figures/phase5_clinical_notes/D/`.

## Sanity checks before you report

- Every note is fact-checked: `unsupported_claims` must be 0 over the 50 cases (the hard gate).
- Read ten notes yourself, including low-confidence ones: does each say that clinician review is required?
- After every variant has run: `python -m ecgaug.experiments.phase5_clinical_notes --rating-sheet`, then the blinded ratings.
- No test data: nothing here can read DS2, and you must not try.

## Selection rule

Zero unsupported claims (hard gate); then coverage of the required parts, the team's blinded readability ratings, latency and cost. The template (A) is the permanent fallback. Applied at the phase 5 gate.
