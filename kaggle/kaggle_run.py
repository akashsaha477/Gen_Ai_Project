"""Run one or more configs inside a Kaggle session (or any fresh checkout) and bundle the outputs.

Called by kaggle/run_config.ipynb after it has cloned the repository. Uses only the
standard library, so it runs with the system Python before the project environment exists.

    python kaggle/kaggle_run.py --config configs/m1/A.yaml -- --device cuda
    python kaggle/kaggle_run.py --config configs/m1/A.yaml configs/m1/B.yaml --raw-dir /kaggle/input/mitdb-raw

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
EXTRA_RESULT_FILES = ("results/data/ptbxl_preprocessing.json",)


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
    """Zip runs/, results/validation/ and the PTB-XL report into ``out_dir``; None if no run produced output."""
    staging = out_dir / "_bundle"
    shutil.rmtree(staging, ignore_errors=True)
    found = False
    for rel in ("runs", "results/validation"):
        if (repo / rel).is_dir() and any((repo / rel).rglob("*")):
            shutil.copytree(repo / rel, staging / rel)
            found = True
    for rel in EXTRA_RESULT_FILES:  # small reports that exist only where the full data was processed
        if found and (repo / rel).is_file():
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / rel, staging / rel)
    if not found:
        return None
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M")
    archive = shutil.make_archive(str(out_dir / f"ecgaug_{manifest['git_sha'][:7]}_{stamp}"), "zip", staging)
    shutil.rmtree(staging)
    return Path(archive)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True, help="one or more config files, run in order")
    parser.add_argument("--cli", default="ecgaug.experiments.m1", help="package CLI that runs a config")
    parser.add_argument("--env", default="locked", choices=["locked", "kaggle", "none"])
    parser.add_argument("--raw-dir", default=None, help="folder with the raw MIT-BIH files (else download)")
    parser.add_argument("--out", default="/kaggle/working", help="where the bundle zip is written")
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
    if archive:
        print(f"bundle: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
        print("next: download it and run `python kaggle/ingest_runs.py <zip>` in your clone")
    else:
        print("no run outputs to bundle")
    return 0 if all(code == 0 for code in results.values()) and len(results) == len(args.config) else 1


if __name__ == "__main__":
    raise SystemExit(main())
