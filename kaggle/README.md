# Running the project on Kaggle

There is one notebook per job. Each clones this repository at the branch or commit you name, installs
the environment, checks the raw data, runs the experiment and produces one zip. You bring the zip back
with `kaggle/ingest_runs.py`. No logic lives in the notebooks.

| Notebook | What it runs | Data |
|---|---|---|
| `module1_input_representation.ipynb` | Module 1: the probe model on one or more input variants (A–D) | MIT-BIH DS1 |
| `ptbxl_pretraining.ipynb` | PTB-XL preprocessing, pretraining, MIT-BIH fine-tuning, and from scratch for comparison | PTB-XL, then MIT-BIH DS1 |

Other files in this folder: `kaggle_run.py` (the script both notebooks call), `make_raw_dataset.py`
(builds the MIT-BIH zip to upload once) and `ingest_runs.py` (brings results back).

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
   tokens > only this repository > Contents: Read-only. Copy it once.
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
   | `GIT_REF` | your branch (`variant/m1-A`) or a full commit SHA |
   | `DEVICE` | `"auto"` uses the GPU when the session has one |
   | `ENV_MODE` | `locked` (exact `uv.lock` versions) or `kaggle` (see below) |
   | `RAW_DATASET_DIR` | leave as `/kaggle/input`; `""` downloads MIT-BIH from PhysioNet instead |
   | `PTBXL_SOURCE` | pretraining notebook only: `"download"` or a folder holding a PTB-XL copy |
   | `PREVIEW` | pretraining notebook only: `True` to run before Gate 1 or for another variant |

3. Run All. For long runs use Save Version > Save & Run All (Commit): it keeps running with the
   browser closed.
4. Download `ecgaug_<sha>_<time>.zip` from the notebook's Output.
5. In your clone:
   ```
   uv run python kaggle/ingest_runs.py ~/Downloads/ecgaug_<sha>_<time>.zip
   git add results runs && git commit -m "m1 A results" && git push
   ```
   Ingest never overwrites an existing run or result file and refuses any run that carries test
   predictions.

First time: set `SMOKE_TEST = True` and run once. If it prints no error, your setup works.

## Module 1 notebook

Each letter is 6 training runs (3 seeds × 2 learning rates) on MIT-BIH DS1 with the approved
validation split. A GPU is optional. Results: `results/validation/m1_input_representation/<letter>.json`.

## PTB-XL pretraining notebook

One run does all of it for each letter in `VARIANTS`: PTB-XL preprocessing (once per session, reused
for every letter), pretraining on PTB-XL beats, fine-tuning on MIT-BIH, and the same model trained
from scratch for comparison. Turn the GPU on.

Run it **after Gate 1**, with the letter Gate 1 chose (`python -m ecgaug.workflow status` shows it).
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
| `results/data/ptbxl_preprocessing.json` | recordings and beats kept, every exclusion by reason, heart-rate and noise statistics, and how many files match PhysioNet's checksums |
| `results/validation/ptbxl_pretraining/<letter>.json` | both arms (from scratch, pretrained): per-seed validation metrics, mean ± SD, the pretraining AUROC, and which arm the selection rule picks |
| `runs/ptbxl_pretraining/<variant>/` | `pretrain/` (3 runs), `scratch/` (6 runs) and `finetune/` (6 runs) |

Things to know:

- Until Gate 1 fixes the input variant, the result is marked `provisional`. Rerun the winning letter
  after Gate 1, and again when Phase 2 has chosen the baseline architecture.
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
- Same seeds, same budget, same commit for every seed of a variant. Do not edit
  `configs/m1_input_representation/shared_budget.yaml`.

## When something fails

| symptom | cause and fix |
|---|---|
| `git clone` fails | the `GITHUB_TOKEN` secret is missing, not ticked for this notebook, expired, or you are not a collaborator |
| `no MIT-BIH record files` or a checksum problem | the `mitdb-raw` dataset is not attached or is damaged; re-attach it, or set `RAW_DATASET_DIR = ""` |
| exit code 2, "Refusing to run" | approved split missing, checkout differs from the commit, PTB-XL not available, or (pretraining) Gate 1 not approved or a different variant chosen; the message says which |
| "The PTB-XL copy ... is incomplete" | the attached copy has no `records500/` folder; attach another or set `PTBXL_SOURCE = "download"` |
| the PTB-XL download is very slow | attach a Kaggle copy and set `PTBXL_SOURCE = "/kaggle/input"` |
| `uv sync` fails or is too slow | Internet is off, or try `ENV_MODE = "kaggle"` for debugging |
| no bundle produced | nothing ran to completion, or it was a smoke test |

## Later phases

Each new module gets its own runner under `ecgaug.experiments` and its own notebook here.

Status: the runner script was rehearsed in a fresh clone with the raw-data zip as input (locked
environment, checksum check, smoke runs of both runners). The pretraining arm was run end to end on
fake data, and its preprocessing and pretraining stage on 64 real PTB-XL recordings. Neither notebook
nor a full PTB-XL run has been executed on Kaggle yet, so run times are not known.
