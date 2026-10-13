# Running the project on Kaggle

Everyone runs the same notebook, `kaggle/run_config.ipynb`. It clones this repository at the
branch or commit you name, installs the environment, checks the raw data, runs your configs and
produces one zip. You bring the zip back with `kaggle/ingest_runs.py`. No logic lives in the notebook.

## One-time setup by the repo owner

1. Add the other three members as collaborators on GitHub (the repo is private).
2. Build the raw-data zip on a machine that has the verified download:
   ```
   uv run python -m ecgaug.data.raw          # only if data/raw/mitdb is not there yet
   uv run python kaggle/make_raw_dataset.py # writes data/kaggle/mitdb-raw.zip (66 MB)
   ```
3. On kaggle.com: Datasets > New Dataset > upload `mitdb-raw.zip`, title `mitdb-raw`, keep it
   **private**, then Settings > Sharing > add the three teammates as viewers.

## One-time setup by each member

1. Kaggle account with a verified phone number (needed for Internet and GPU in notebooks).
2. A GitHub token that can read the repo: GitHub > Settings > Developer settings > Fine-grained
   tokens > only this repository > Contents: Read-only. Copy it once.
3. On kaggle.com: Create > New Notebook > File > Import Notebook > upload
   `kaggle/run_config.ipynb`.
4. In the notebook:
   - Add-ons > Secrets > add a secret named `GITHUB_TOKEN` with the token, and tick it for this notebook.
   - Add Input > Your Datasets (or Shared with you) > `mitdb-raw`.
   - Settings: Internet **on**. Accelerator: GPU only when the phase needs it (Module 1 runs fine on CPU).

The token is read from Kaggle Secrets, never printed and never put in a URL. Do not paste it into a cell.

## Every run

1. Commit and push your work first. Kaggle runs what is on GitHub, not what is on your laptop.
2. Edit the parameter cell:

   | parameter | meaning |
   |---|---|
   | `GIT_REF` | your branch (`variant/m1-A`) or a full commit SHA |
   | `CONFIGS` | one or more config files, run in order |
   | `MODULE_CLI` | the runner of the module (Module 1: `ecgaug.experiments.input_representation`) |
   | `EXTRA_ARGS` | passed to the runner; `["--device", "auto"]` uses the GPU when there is one |
   | `ENV_MODE` | `locked` (exact `uv.lock` versions) or `kaggle` (see below) |
   | `RAW_DATASET_DIR` | leave as `/kaggle/input`; set to `""` to download from PhysioNet instead |

3. Run All. For long runs use Save Version > Save & Run All (Commit): it keeps running with the
   browser closed.
4. Download `ecgaug_<sha>_<time>.zip` from the notebook's Output.
5. In your clone, on your variant branch:
   ```
   uv run python kaggle/ingest_runs.py ~/Downloads/ecgaug_<sha>_<time>.zip
   git add results runs && git commit -m "m1 A results" && git push
   ```
   Ingest never overwrites an existing run or result file and refuses any run that carries test
   predictions.

First time: set `EXTRA_ARGS = ["--smoke"]` and run once. It exercises the whole pipeline on fake data
in about a minute and produces no bundle; if it prints no error, your setup works.

## PTB-XL pretraining arm (preprocessing, pretraining and fine-tuning in one run)

One run does all of it: PTB-XL preprocessing, pretraining on PTB-XL beats, fine-tuning on MIT-BIH, and
the same model trained from scratch for comparison.

1. Give the session a copy of PTB-XL, one of two ways:
   - **Attach a copy:** Add Input > search Kaggle Datasets for "PTB-XL" > pick one whose files include
     `ptbxl_database.csv`, `scp_statements.csv` and a `records500/` folder. Use
     `"--ptbxl-root", "/kaggle/input"`; the folder is found automatically.
   - **Download it:** use `"--ptbxl-download"` instead. The session fetches PTB-XL 1.0.3 from PhysioNet
     (1.8 GB) and unpacks only what is needed. Internet must be on.
2. Keep the `mitdb-raw` dataset attached as well, and turn the GPU on.
3. Parameter cell:
   ```
   GIT_REF = "main"
   CONFIGS = ["configs/ptbxl_pretraining/A_short_window.yaml"]          # or several: A.yaml, B.yaml, C.yaml, D.yaml
   MODULE_CLI = "ecgaug.experiments.ptbxl_pretraining"
   EXTRA_ARGS = ["--device", "auto", "--ptbxl-root", "/kaggle/input"]
   ```
4. Run All, download the zip, `python kaggle/ingest_runs.py <zip>`.

What comes back in the zip:

| file | content |
|---|---|
| `results/data/ptbxl_preprocessing.json` | how many recordings and beats were kept, every exclusion by reason, heart-rate and noise statistics, and how many files match PhysioNet's checksums |
| `results/validation/ptbxl_pretraining/<A-D>.json` | both arms (from scratch, pretrained): per-seed validation metrics, mean ± SD, the pretraining AUROC, and which arm the selection rule picks |
| `runs/` | one folder per run: 3 pretraining runs, 6 from-scratch and 6 fine-tuning runs per config |

Things to know:

- The letter is the input representation (the Module 1 variant). PTB-XL is preprocessed once per session
  and reused for every config in `CONFIGS`.
- Until Gate 1 fixes the representation, the result is marked `provisional`. Rerun the winning letter
  after Gate 1, and again when Phase 2 has chosen the baseline architecture.
- If the attached copy is not version 1.0.3, the checksum comparison reports mismatches. That is a
  warning, not an error; read the numbers in `ptbxl_preprocessing.json`.
- First time, add `"--smoke"` to `EXTRA_ARGS`: the whole arm runs on fake data in under a minute.

## Environment modes

- `locked` (default): installs exactly the versions in `uv.lock`. Use this for every result that goes
  into a gate. It downloads PyTorch each session, which takes a few minutes.
- `kaggle`: uses Kaggle's preinstalled PyTorch and adds only what is missing. Faster to start, but the
  versions differ from the lockfile. Fine for debugging; flag it as a budget deviation if you use it
  for a result. The actual versions are always recorded in each run's `env.json`.

## Rules that still apply on Kaggle

- A real run is refused (exit code 2) if `configs/validation_split/approved.yaml` is missing or the checkout
  differs from the commit. The message says which.
- Never set `ALLOW_TEST` or `ALLOW_TEST_BUILD` in a notebook. The final evaluation is run once, locally,
  through `/final-eval`.
- One GPU quota per account, renewed weekly (your remaining hours are shown under Settings in the
  notebook). Turn the accelerator off when you do not need it, and ask the team before anything
  estimated above 2 GPU-hours.
- Same seeds, same budget, same commit for every seed of a variant. Do not edit `configs/m1_input_representation/shared_budget.yaml`.

## When something fails

| symptom | cause and fix |
|---|---|
| `git clone` fails | the `GITHUB_TOKEN` secret is missing, not ticked for this notebook, expired, or you are not a collaborator |
| `no MIT-BIH record files` or a checksum problem | the `mitdb-raw` dataset is not attached or is damaged; re-attach it, or set `RAW_DATASET_DIR = ""` |
| exit code 2, "Refusing to run" | approved split missing, or the checkout differs from the commit |
| `uv sync` fails or is too slow | Internet is off, or try `ENV_MODE = "kaggle"` for debugging |
| no bundle produced | nothing ran to completion (or it was a `--smoke` or `--check` run) |

## Later phases

The same notebook serves every phase: each module gets its own runner under `ecgaug.experiments`, and
you change only `MODULE_CLI` and `CONFIGS`. Runners so far: `ecgaug.experiments.input_representation` (Module 1) and
`ecgaug.experiments.ptbxl_pretraining` (PTB-XL pretraining arm). This file is updated when a phase adds one.

Status: the runner script was rehearsed in a fresh clone with the raw-data zip as input (locked
environment, checksum check, two smoke runs). The pretraining arm was run end to end on fake data, and
its preprocessing and pretraining stage on 64 real PTB-XL recordings. Neither the notebook nor a full
PTB-XL run has been executed on Kaggle yet, so run times are not known.
