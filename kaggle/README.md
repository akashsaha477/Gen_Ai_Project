# Running the project on Kaggle

There is one notebook per job. Each clones this repository at the branch or commit you name, installs
the environment, checks the raw data, runs the experiment, then commits the results and pushes them
to GitHub, so you only `git pull`. No logic lives in the notebooks.

| Notebook | What it runs | Inputs |
|---|---|---|
| `phase1_input_representation.ipynb` | phase1_input_representation: the probe model on one or more input variants (A–D) | `mitdb-raw` |
| `phase2_baseline_classifiers.ipynb` | the four baseline classifiers (A–D) | `mitdb-raw` |
| `phase2_quality_metrics.ipynb` | the sanity harness of the four quality-metric families (A–D) | `mitdb-raw` |
| `phase2_ptbxl_pretraining.ipynb` | PTB-XL preprocessing, pretraining, MIT-BIH fine-tuning, and from scratch for comparison | `mitdb-raw`, PTB-XL |
| `phase3_cvae.ipynb`, `phase3_wgan_gp.ipynb` | generator pilots (`MODE = "pilot"`), or the chosen generator's full runs and banks (`MODE = "final"`) | `mitdb-raw` |
| `phase4_augmentation_policy.ipynb` | the four augmentation policies with both frozen banks | `mitdb-raw`, banks |
| `phase4_full_grid.ipynb` | the full grid; resumable across sessions | `mitdb-raw`, banks |
| `phase5_explainability.ipynb`, `phase5_clinical_notes.ipynb` | saliency methods and note generators on the phase 4 models | `mitdb-raw`, artifacts (checkpoints) |

Other files in this folder: `kaggle_run.py` (the script every notebook calls), `make_raw_dataset.py`
(builds the MIT-BIH zip to upload once) and `ingest_runs.py` (the backup way to bring results back from
the zip, and the way to bring back weights and banks).

## Weights and banks: the artifacts dataset

Results, logs and figures reach GitHub by themselves. Model weights (`checkpoints/best.pt`) and the
synthetic banks (`data/synthetic/`, about 75 MB per seed) are too big for git: they are only in the zip
in each notebook's Output.

1. After each phase 3 final run and each full-grid session, download the zip and, in your clone,
   `uv run python kaggle/ingest_runs.py <zip>` (brings the weights into `runs/` and the banks into
   `data/synthetic/`; nothing is overwritten).
2. For notebooks that need them (phase 4 and 5), upload the extracted zips once as a private Kaggle
   dataset (e.g. `ecgaug-artifacts`), share it with the team, attach it with Add Input, and set
   `BANKS_DATASET_DIR` (phase 4) or `ARTIFACTS_DIR` (phase 5) to its folder. The runs check every bank
   and checkpoint against the SHA-256 recorded at the gates before using it.

## New parameters

| parameter | notebooks | meaning |
|---|---|---|
| `PREVIEW` | phase 2-4 | `True` runs before the gates the run needs; the result is provisional and no gate accepts it |
| `REPRESENTATION` | phase 2-4 | preview only: which phase1_input_representation letter to use (`""` = the phase 1 gate report's winner) |
| `MODE` | phase 3 | `"pilot"` (seeds 0 and 1, for the phase 3a gate) or `"final"` (the chosen variant, seeds 0-2, banks) |
| `BANKS_DATASET_DIR` | phase 4 | the attached artifacts dataset holding the banks |
| `PREVIEW_BANKS` | phase 4 policy | preview only: which banks to use, e.g. `{"cvae": "phase3_cvae-A", "wgan_gp": "phase3_wgan_gp-B"}` |
| `ARTIFACTS_DIR` | phase 5 | the attached artifacts dataset holding the grid's weights |

The note variants B-D call an LLM: add a Kaggle secret named `LLM_API_KEY` and tick it for the
notes notebook (optional: set `ECGAUG_LLM_MODEL` in the cell). The key is passed to the run and never
printed. Without it, B-D are recorded as not run.

The four input variants, used by both notebooks:

| Letter | Config name | Input per beat |
|---|---|---|
| A | `A_short_window` | 0.5 s window (180 samples) |
| B | `B_long_window_bandpass` | 1 s window, 0.5–40 Hz band-pass (360 samples) |
| C | `C_short_window_rr_features` | A plus three RR-interval features |
| D | `D_short_window_125hz` | A resampled to 125 Hz (63 samples) |

## One-time setup by the repo owner

1. Add the other three members as collaborators on GitHub (the repo is private).
2. Build the raw-data zip on a machine that has the verified download:
   ```
   uv run python -m ecgaug.data.raw           # only if data/raw/mitdb is not there yet
   uv run python kaggle/make_raw_dataset.py   # writes data/kaggle/mitdb-raw.zip (66 MB)
   ```
3. On kaggle.com: Datasets > New Dataset > upload `mitdb-raw.zip`, title `mitdb-raw`, keep it
   **private**, then Settings > Sharing > add the three teammates as viewers.

## One-time setup by each member

1. Kaggle account with a verified phone number (needed for Internet and GPU in notebooks).
2. A GitHub token that can read the repo: GitHub > Settings > Developer settings > Fine-grained
   tokens > only this repository > Contents: **Read and write** (the notebook pushes the results). Copy it once.
3. On kaggle.com: Create > New Notebook > File > Import Notebook > upload the notebook you need from
   this folder.
4. In the notebook:
   - Add-ons > Secrets > add a secret named `GITHUB_TOKEN` with the token, and tick it for this notebook.
   - Add Input > Your Datasets (or Shared with you) > `mitdb-raw`.
   - Settings: Internet **on**.

The token is read from Kaggle Secrets, never printed and never put in a URL. Do not paste it into a cell.

## Every run

1. Commit and push your work first. Kaggle runs what is on GitHub, not what is on your laptop.
2. Edit the parameter cell (the only cell you change):

   | parameter | meaning |
   |---|---|
   | `VARIANTS` | the letters to run, e.g. `["A"]` or `["A", "B", "C", "D"]` |
   | `SMOKE_TEST` | `True` runs the whole pipeline on fake data in about a minute and produces no results |
   | `GIT_REF` | your branch (`variant/phase1_input_representation_A`) or a full commit SHA |
   | `DEVICE` | `"auto"` uses the GPU when the session has one |
   | `ENV_MODE` | `locked` (exact `uv.lock` versions) or `kaggle` (see below) |
   | `RAW_DATASET_DIR` | leave as `/kaggle/input`; `""` downloads MIT-BIH from PhysioNet instead |
   | `PTBXL_SOURCE` | pretraining notebook only: `"download"` or a folder holding a PTB-XL copy |
   | `PREVIEW` | pretraining notebook only: `True` to run before the phase 1 gate or for another variant |
   | `PUSH_RESULTS` | `True` (default) commits the results and pushes them to `GIT_REF` |
   | `GIT_NAME`, `GIT_EMAIL` | your name and GitHub email for that commit (required for the push) |

3. Run All. For long runs use Save Version > Save & Run All (Commit): it keeps running with the
   browser closed.
4. When the run ends, the notebook commits the run folders, result files, figures and runs index
   (never model weights) and pushes them to `GIT_REF`. The last lines of the output say where. In your
   clone: `git pull`.
   - If someone pushed to that branch while you were running, the results go to a new branch
     `<GIT_REF>-kaggle-<time>` instead, and you merge it with a pull request. Nothing is overwritten.
   - Nothing is pushed if a run failed, if a run carries test predictions, or for a smoke test.
5. Backup: the same outputs are in `ecgaug_<sha>_<time>.zip` in the notebook's Output. If the push did not
   happen, download it and run `uv run python kaggle/ingest_runs.py <zip>` in your clone, then commit
   and push. Ingest never overwrites an existing run or result file and refuses test predictions.

First time: set `SMOKE_TEST = True` and run once. If it prints no error, your setup works.

## phase1_input_representation notebook

Each letter is 6 training runs (3 seeds × 2 learning rates) on MIT-BIH DS1 with the approved
validation split. A GPU is optional. Results: `results/phase1_input_representation/<letter>.json`.

## PTB-XL pretraining notebook

One run does all of it for each letter in `VARIANTS`: PTB-XL preprocessing (once per session, reused
for every letter), pretraining on PTB-XL beats, fine-tuning on MIT-BIH, and the same model trained
from scratch for comparison. Turn the GPU on.

Run it **after the phase 1 gate**, with the letter phase 1 gate chose (`python -m ecgaug.workflow status` shows it).
Before that, or for another letter, the run is refused unless you set `PREVIEW = True`; preview
results are marked provisional and do not count.

PTB-XL comes from one of two places:

- `PTBXL_SOURCE = "download"` (default): the session fetches PTB-XL 1.0.3 from PhysioNet (1.8 GB) and
  unpacks only what is needed.
- `PTBXL_SOURCE = "/kaggle/input"`: use a copy you attached with Add Input. Search Kaggle Datasets for
  "PTB-XL" and pick one whose files include `ptbxl_database.csv`, `scp_statements.csv` and a
  `records500/` folder. The folder is found automatically.

What comes back in the zip:

| file | content |
|---|---|
| `results/phase2_ptbxl_data/ptbxl_preprocessing.json` | recordings and beats kept, every exclusion by reason, heart-rate and noise statistics, and how many files match PhysioNet's checksums |
| `results/phase2_ptbxl_pretraining/<letter>.json` | both methods (from scratch, fine-tuned): per-seed validation metrics, mean ± SD, the pretraining AUROC, and which arm the selection rule picks |
| `runs/phase2_ptbxl_pretraining/<variant>/` | `pretrain/` (3 runs), `scratch/` (6 runs) and `finetune/` (6 runs) |

Things to know:

- Until phase 1 gate fixes the input variant, the result is marked `provisional`. Rerun the winning letter
  after the phase 1 gate, and again when Phase 2 has chosen the baseline architecture.
- If an attached copy is not version 1.0.3, the checksum comparison reports mismatches. That is a
  warning, not an error; read the numbers in `ptbxl_preprocessing.json`.

## Environment modes

- `locked` (default): installs exactly the versions in `uv.lock`. Use this for every result that goes
  into a gate. It downloads PyTorch each session, which takes a few minutes.
- `kaggle`: uses Kaggle's preinstalled PyTorch and adds only what is missing. Faster to start, but the
  versions differ from the lockfile. Fine for debugging; flag it as a budget deviation if you use it
  for a result. The actual versions are always recorded in each run's `env.json`.

## Rules that still apply on Kaggle

- A real run is refused (exit code 2) if `configs/validation_split/approved.yaml` is missing or the
  checkout differs from the commit. The message says which.
- Never set `ALLOW_TEST` or `ALLOW_TEST_BUILD` in a notebook. The final evaluation is run once, locally,
  through `/final-eval`.
- One GPU quota per account, renewed weekly (your remaining hours are shown under Settings in the
  notebook). Turn the accelerator off when you do not need it, and ask the team before anything
  estimated above 2 GPU-hours.
- Same seeds, same budget, same commit for every seed of a variant. Do not edit the shared settings
  files (`shared_budget.yaml`, `generator_settings.yaml`, ...) of an experiment.

## When something fails

| symptom | cause and fix |
|---|---|
| `git clone` fails | the `GITHUB_TOKEN` secret is missing, not ticked for this notebook, expired, or you are not a collaborator |
| `no MIT-BIH record files` or a checksum problem | the `mitdb-raw` dataset is not attached or is damaged; re-attach it, or set `RAW_DATASET_DIR = ""` |
| exit code 2, "Refusing to run" | approved split missing, checkout differs from the commit, PTB-XL not available, or (pretraining) phase 1 gate not approved or a different variant chosen; the message says which |
| "The PTB-XL copy ... is incomplete" | the attached copy has no `records500/` folder; attach another or set `PTBXL_SOURCE = "download"` |
| the PTB-XL download is very slow | attach a Kaggle copy and set `PTBXL_SOURCE = "/kaggle/input"` |
| `uv sync` fails or is too slow | Internet is off, or try `ENV_MODE = "kaggle"` for debugging |
| no bundle produced | nothing ran to completion, or it was a smoke test |

## Later phases

Each experiment has its own runner under `ecgaug.experiments` and its own notebook here; the full
grid uses `ecgaug.grid.runner`. The phase 2-5 notebooks were built from the same template as the
phase 1 one and their code is tested locally (smoke runs); they have not run on Kaggle yet.

Status: the runner script was rehearsed in a fresh clone with the raw-data zip as input (locked
environment, checksum check, smoke runs of both runners). The phase2_ptbxl_pretraining experiment was run end to end on
fake data, and its preprocessing and pretraining part on 64 real PTB-XL recordings. Neither notebook
nor a full PTB-XL run has been executed on Kaggle yet, so run times are not known.
