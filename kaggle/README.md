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
   | `MODULE_CLI` | the runner of the module (Module 1: `ecgaug.experiments.m1`) |
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

## Environment modes

- `locked` (default): installs exactly the versions in `uv.lock`. Use this for every result that goes
  into a gate. It downloads PyTorch each session, which takes a few minutes.
- `kaggle`: uses Kaggle's preinstalled PyTorch and adds only what is missing. Faster to start, but the
  versions differ from the lockfile. Fine for debugging; flag it as a budget deviation if you use it
  for a result. The actual versions are always recorded in each run's `env.json`.

## Rules that still apply on Kaggle

- A real run is refused (exit code 2) until `docs/PREREGISTRATION.md` is in force and the validation
  split is approved. The message says which one is missing.
- Never set `ALLOW_TEST` or `ALLOW_TEST_BUILD` in a notebook. The final evaluation is run once, locally,
  through `/final-eval`.
- One GPU quota per account, renewed weekly (your remaining hours are shown under Settings in the
  notebook). Turn the accelerator off when you do not need it, and ask the team before anything
  estimated above 2 GPU-hours.
- Same seeds, same budget, same commit for every seed of a variant. Do not edit `configs/m1/_shared.yaml`.

## When something fails

| symptom | cause and fix |
|---|---|
| `git clone` fails | the `GITHUB_TOKEN` secret is missing, not ticked for this notebook, expired, or you are not a collaborator |
| `no MIT-BIH record files` or a checksum problem | the `mitdb-raw` dataset is not attached or is damaged; re-attach it, or set `RAW_DATASET_DIR = ""` |
| exit code 2, "Refusing to run" | pre-registration not in force, split not approved, or the checkout differs from the commit |
| `uv sync` fails or is too slow | Internet is off, or try `ENV_MODE = "kaggle"` for debugging |
| no bundle produced | nothing ran to completion (or it was a `--smoke` or `--check` run) |

## Later phases

The same notebook serves every phase: each module gets its own runner under `ecgaug.experiments`, and
you change only `MODULE_CLI` and `CONFIGS`. This file is updated when a phase adds a runner.

Status: the runner script was rehearsed in a fresh clone with the raw-data zip as input (locked
environment, checksum check, two smoke runs). The notebook itself has not yet been executed on Kaggle.
