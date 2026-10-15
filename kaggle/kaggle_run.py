"""Run one or more configs inside a Kaggle session (or any fresh checkout) and bundle the outputs.

Called by the notebooks in this folder after they have cloned the repository. Uses only the
standard library, so it runs with the system Python before the project environment exists.

    python kaggle/kaggle_run.py --config configs/phase1_input_representation/A_short_window.yaml -- --device cuda
    python kaggle/kaggle_run.py --config configs/phase1_input_representation/A_short_window.yaml configs/phase1_input_representation/B_long_window_bandpass.yaml --raw-dir /kaggle/input/mitdb-raw

Steps: install the environment, get and verify the raw data, run each config through its package
CLI, and zip runs/ plus the variant result files for kaggle/ingest_runs.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Kept in step with ecgaug.names (this script runs before the package is installed; a test checks it).
EXPERIMENTS = ("phase1_input_representation", "phase2_ptbxl_pretraining", "phase3_cvae", "phase3_wgan_gp")
EXTRA_RESULT_FILES = ("results/phase2_ptbxl_data/ptbxl_preprocessing.json",)
BANKS = "data/synthetic"  # synthetic banks of final generator runs: in the zip, never pushed (too big for git)


def sh(cmd: list[str], check: bool = True) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=REPO, check=check).returncode


def setup_env(mode: str) -> list[str]:
    """Install the project and return the command prefix that runs a module in it.

    'locked' installs exactly uv.lock (reference environment). 'kaggle' keeps Kaggle's preinstalled
    packages and adds only what is missing; faster, but versions differ from the lockfile and are
    recorded in each run's env.json.
    """
    if mode == "locked":
        sh([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
        sh(["uv", "sync", "--frozen", "--no-dev"])
        return ["uv", "run", "--frozen", "--no-dev", "python", "-m"]
    if mode == "kaggle":
        sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", "."])
        sh([sys.executable, "-m", "pip", "install", "--quiet", "wfdb", "pyyaml"])
        return [sys.executable, "-m"]
    if mode == "none":
        return [sys.executable, "-m"]
    raise ValueError(f"unknown env mode {mode!r}")


def find_raw(raw_dir: Path) -> Path:
    """The folder under ``raw_dir`` that holds the record files (Kaggle datasets may nest them)."""
    hits = sorted(raw_dir.rglob("100.dat"))
    if not hits:
        raise FileNotFoundError(f"no MIT-BIH record files (100.dat) under {raw_dir}")
    return hits[0].parent


def setup_data(run: list[str], raw_dir: str | None) -> None:
    """Copy attached raw files or download them, then verify every file against PhysioNet."""
    target = REPO / "data" / "raw" / "mitdb"
    if raw_dir:
        source = find_raw(Path(raw_dir))
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.suffix in (".hea", ".dat", ".atr"):
                shutil.copy2(path, target / path.name)
        sh([*run, "ecgaug.data.raw", "--skip-download"])
    else:
        sh([*run, "ecgaug.data.raw"])


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() or "0" * 40


def bundle(out_dir: Path, manifest: dict, repo: Path = REPO) -> Path | None:
    """Zip runs/, each experiment's results/ folder, figures, the PTB-XL report and any synthetic banks;
    None if nothing ran."""
    staging = out_dir / "_bundle"
    shutil.rmtree(staging, ignore_errors=True)
    found = False
    for rel in ("runs", *(f"results/{name}" for name in EXPERIMENTS)):
        if (repo / rel).is_dir() and any((repo / rel).rglob("*")):
            shutil.copytree(repo / rel, staging / rel)
            found = True
    if found and (repo / "figures").is_dir():  # figures made at the end of each run
        shutil.copytree(repo / "figures", staging / "figures")
    for rel in EXTRA_RESULT_FILES:  # small reports that exist only where the full data was processed
        if found and (repo / rel).is_file():
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / rel, staging / rel)
    if found and (repo / BANKS).is_dir():
        shutil.copytree(repo / BANKS, staging / BANKS)
    if not found:
        return None
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M")
    archive = shutil.make_archive(str(out_dir / f"ecgaug_{manifest['git_sha'][:7]}_{stamp}"), "zip", staging)
    shutil.rmtree(staging)
    return Path(archive)


RESULT_PATHS = ("runs", "figures", "results/runs_index.csv", *EXTRA_RESULT_FILES)


def push_results(branch: str, message: str, name: str, email: str, repo: Path = REPO) -> str | None:
    """Commit the run outputs in the clone and push them to GitHub; return the branch pushed to.

    This replaces downloading the zip and running ingest_runs.py. Only run outputs are committed
    (run folders without weights, result files, figures, the runs index), and nothing is pushed if a
    run carries test predictions. The push goes to ``branch`` if that is a fast-forward; otherwise to a
    new branch ``<branch>-kaggle-<time>`` so nobody's work is overwritten. Returns None if there was
    nothing to commit.
    """
    if any((repo / "runs").rglob("preds_test*")):
        raise RuntimeError("A run carries test predictions; nothing is pushed. Stop and report this to the team.")
    paths = [p for p in (*RESULT_PATHS, *(f"results/{e}" for e in EXPERIMENTS)) if (repo / p).exists()]
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode == 0:
        return None
    subprocess.run(["git", "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "--quiet", "-m", message],
                   cwd=repo, check=True)  # fmt: skip
    if subprocess.run(["git", "push", "--quiet", "origin", f"HEAD:refs/heads/{branch}"], cwd=repo).returncode == 0:
        return branch
    fallback = f"{branch}-kaggle-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M')}"
    subprocess.run(["git", "push", "--quiet", "origin", f"HEAD:refs/heads/{fallback}"], cwd=repo, check=True)
    return fallback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True, help="one or more config files, run in order")
    parser.add_argument("--cli", default="ecgaug.experiments.phase1_input_representation", help="package CLI that runs a config")
    parser.add_argument("--env", default="locked", choices=["locked", "kaggle", "none"])
    parser.add_argument("--raw-dir", default=None, help="folder with the raw MIT-BIH files (else download)")
    parser.add_argument("--out", default="/kaggle/working", help="where the bundle zip is written")
    parser.add_argument("--push-to", default=None, metavar="BRANCH", help="commit the outputs and push them to this branch")
    parser.add_argument("--git-name", default="", help="author name for the results commit")
    parser.add_argument("--git-email", default="", help="author email for the results commit")
    parser.add_argument("extra", nargs="*", help="arguments after -- are passed to the CLI")
    args = parser.parse_args(argv)

    run = setup_env(args.env)
    setup_data(run, args.raw_dir)

    results = {}
    for config in args.config:
        code = sh([*run, args.cli, "--config", config, *args.extra], check=False)
        results[config] = code
        if code != 0:
            print(f"!! {config} exited with code {code}; stopping here", flush=True)
            break

    manifest = {"git_sha": git_sha(), "cli": args.cli, "env": args.env, "extra": args.extra, "exit_codes": results}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = bundle(out_dir, manifest)
    ok = all(code == 0 for code in results.values()) and len(results) == len(args.config)
    if archive:
        print(f"bundle (backup copy of the outputs): {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    else:
        print("no run outputs to bundle")
    if args.push_to and archive and ok:
        if not (args.git_name and args.git_email):
            print("not pushed: set GIT_NAME and GIT_EMAIL; download the zip and run kaggle/ingest_runs.py instead")
            return 1
        names = " ".join(Path(c).stem for c in args.config)
        experiment = Path(args.config[0]).parent.name
        branch = push_results(args.push_to, f"add {experiment} results: {names}", args.git_name, args.git_email)
        if branch:
            print(f"results committed and pushed to branch '{branch}'. In your clone: git pull origin {branch}")
            if branch != args.push_to:
                print(f"('{args.push_to}' had moved on since this run started, so a new branch was used; merge it with a pull request)")
    elif archive:
        print("next: download the zip and run `python kaggle/ingest_runs.py <zip>` in your clone")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
