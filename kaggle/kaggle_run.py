"""Run one or more configs inside a Kaggle session (or any fresh checkout) and bundle the outputs.

Called by the notebooks in this folder after they have cloned the repository. Uses only the
standard library, so it runs with the system Python before the project environment exists.

    python kaggle/kaggle_run.py --config configs/phase1_input_representation/A_short_window.yaml -- --device cuda
    python kaggle/kaggle_run.py --config configs/phase1_input_representation/A_short_window.yaml configs/phase1_input_representation/B_long_window_bandpass.yaml --raw-dir /kaggle/input

Steps: install the environment, copy the raw MIT-BIH files from the attached Kaggle dataset (or, if none
is attached, download them from PhysioNet's AWS mirror) and check them against the committed checksums,
run each config through its package CLI, and zip runs/ plus the variant result files for kaggle/ingest_runs.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Kept in step with ecgaug.names (this script runs before the package is installed; a test checks it).
EXPERIMENTS = (
    "phase1_input_representation", "phase2_baseline_classifiers", "phase2_quality_metrics", "phase2_ptbxl_pretraining",
    "phase3_cvae", "phase3_wgan_gp", "phase4_augmentation_policy", "phase4_leakage_audit", "phase5_explainability", "phase5_clinical_notes",
)  # fmt: skip
POST_HOC = ("phase6_rr_reference", "phase6_external_validation", "phase6_extra_seeds", "phase6_audit_check", "phase6_rr_baseline", "phase6_patient_template",
            "phase6_relative_cnn", "phase6_augmentation_mechanism", "phase6_patient_transfer",
            "phase6_patient_adaptation", "phase6_more_patients", "phase6_relative_generators",
            "phase6_record_rr", "phase6_icentia")
RESULT_FOLDERS = (*EXPERIMENTS, "phase4_full_grid", *POST_HOC)  # every experiment's results, the full grid's ledger, post-hoc analyses
EXTRA_RESULT_FILES = ("results/phase2_ptbxl_data/ptbxl_preprocessing.json",)
ACCESS_LOG = "docs/TEST_ACCESS_LOG.md"  # appended by every DS2 read; its new rows are merged back by ingest
BANKS = "data/synthetic"  # synthetic banks of final generator runs: in the zip, never pushed (too big for git)


def sh(cmd: list[str], check: bool = True) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=REPO, check=check).returncode


def setup_env(mode: str, dev: bool = False) -> list[str]:
    """Install the project and return the command prefix that runs a module in it.

    'locked' installs exactly uv.lock (reference environment). 'kaggle' keeps Kaggle's preinstalled
    packages and adds only what is missing; faster, but versions differ from the lockfile and are
    recorded in each run's env.json.
    """
    if mode == "locked":
        sh([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
        groups = [] if dev else ["--no-dev"]  # dev: the test suite's tools too (--tests)
        sh(["uv", "sync", "--frozen", *groups])
        return ["uv", "run", "--frozen", *groups, "python", "-m"]
    if mode == "kaggle":
        sh([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", "."])
        sh([sys.executable, "-m", "pip", "install", "--quiet", "wfdb", "pyyaml"])
        return [sys.executable, "-m"]
    if mode == "none":
        return [sys.executable, "-m"]
    raise ValueError(f"unknown env mode {mode!r}")


RAW_CHECKSUMS = "results/phase1_data/raw_checksums.json"  # SHA-256 of all 144 files, matched against PhysioNet in phase 1
RAW_EXTENSIONS = (".hea", ".dat", ".atr")
RAW_MIRROR = "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0"  # PhysioNet's open-data mirror on AWS


def find_raw(raw_dir: Path) -> Path:
    """The folder under ``raw_dir`` that holds the MIT-BIH record files (Kaggle datasets may nest them)."""
    hits = sorted(p.parent for p in Path(raw_dir).rglob("100.dat") if (p.parent / "100.hea").is_file() and (p.parent / "100.atr").is_file())
    if not hits:
        raise FileNotFoundError(
            f"no MIT-BIH record files (100.dat, 100.hea, 100.atr) under {raw_dir}. "
            "Attach the team's mitdb-raw dataset (kaggle/make_raw_dataset.py); CSV conversions do not count."
        )
    return hits[0]


def download_raw(names: list[str], dest: Path, mirror: str = RAW_MIRROR, tries: int = 3) -> Path:
    """Download the record files from PhysioNet's AWS mirror into ``dest`` (checked by the caller)."""
    import time
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(f"{mirror}/{name}", timeout=60) as response:  # noqa: S310 (fixed https URL)
                    (dest / name).write_bytes(response.read())
                break
            except OSError as err:
                if attempt == tries - 1:
                    raise FileNotFoundError(f"could not download {name} from {mirror}: {err}") from err
                time.sleep(2 ** attempt)
    return dest


def setup_data(raw_dir: str, repo: Path = REPO, mirror: str | None = RAW_MIRROR) -> int:
    """Copy the raw files into ``data/raw/mitdb`` and check each against the committed checksums.

    The files come from the attached dataset under ``raw_dir``; if none is attached, from ``mirror``
    (None: never download). Either way every file must match ``results/phase1_data/raw_checksums.json``,
    whose hashes matched PhysioNet's published list in phase 1. Returns the number of files copied.
    """
    import hashlib

    expected = json.loads((repo / RAW_CHECKSUMS).read_text())["files"]
    try:
        source = find_raw(Path(raw_dir))
    except FileNotFoundError as err:
        if mirror is None:
            raise
        print(f"{err}\nDownloading the original files from {mirror} instead.", flush=True)
        source = download_raw(sorted(expected), repo / "data" / "raw" / "_download", mirror)
    target = repo / "data" / "raw" / "mitdb"
    target.mkdir(parents=True, exist_ok=True)
    problems = []
    for name, digest in sorted(expected.items()):
        path = source / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"checksum mismatch for {name}")
            continue
        shutil.copy2(path, target / name)
    if problems:
        raise RuntimeError(f"The MIT-BIH files in {source} are not the verified ones: {'; '.join(problems[:10])}"
                           + (f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""))  # fmt: skip
    if source.name == "_download":
        shutil.rmtree(source)
    print(f"MIT-BIH: {len(expected)} files from {source} copied and matched against {RAW_CHECKSUMS}", flush=True)
    return len(expected)


def bank_name(meta: Path) -> Path:
    """Where the bank folder holding ``meta`` (its bank.json) goes under ``data/synthetic/``: its path below the last
    ``synthetic`` folder when there is one (P12's banks sit in ``phase6_relative_generators/``), else its folder name
    (a flat dataset), under ``audit/`` for audit banks."""
    parts = meta.parent.parts
    if "synthetic" in parts:
        return Path(*parts[len(parts) - parts[::-1].index("synthetic") :])
    return Path("audit", meta.parent.name) if meta.parent.parent.name == "audit" else Path(meta.parent.name)


def setup_artifacts(artifacts_dir: str, repo: Path = REPO) -> list[str]:
    """Copy artifacts from an attached Kaggle dataset (an extracted Kaggle zip, or several):
    synthetic banks (folders holding a bank.json) into ``data/synthetic/``, and files of run folders
    (checkpoints, predictions) into ``runs/`` where they are missing. Nothing is overwritten; every run
    checks banks and checkpoints against their recorded SHA-256 before using them."""
    copied = []
    for meta in sorted(Path(artifacts_dir).rglob("bank.json")):
        name = bank_name(meta)
        target = repo / BANKS / name
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(p for p in meta.parent.iterdir() if p.is_file()):
            if not (target / path.name).exists():
                shutil.copy2(path, target / path.name)
                copied.append(f"{BANKS}/{name.as_posix()}/{path.name}")
    for runs_root in sorted(p for p in Path(artifacts_dir).rglob("runs") if p.is_dir()):
        for path in sorted(p for p in runs_root.rglob("*") if p.is_file()):
            target = repo / "runs" / path.relative_to(runs_root)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied.append(str(target.relative_to(repo)))
    return copied


GUARDED_DATA = ("test.npz",)  # DS2 arrays never travel in a dataset: they are rebuilt where the guarded loaders run


def setup_data_dir(data_dir: str, repo: Path = REPO) -> list[str]:
    """Copy the files under every ``data/`` folder of an attached dataset into the repository's ``data/``
    where they are missing (external databases, processed arrays). Refuses guarded test arrays; nothing is
    overwritten, and every loader checks its files against the recorded SHA-256 before using them."""
    copied = []
    for data_root in sorted(p for p in Path(data_dir).rglob("data") if p.is_dir() and p.parent.name != "data"):
        for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
            if path.name in GUARDED_DATA:
                raise RuntimeError(f"{path}: guarded test arrays must not come from a dataset")
            target = repo / "data" / path.relative_to(data_root)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied.append(str(target.relative_to(repo)))
    return copied


def run_tests(run: list[str], repo: Path = REPO) -> int:
    """The full test suite at this commit (``python -m pytest``); its exit code goes into the bundle manifest."""
    return sh([*run, "pytest", "-q", "-p", "no:warnings"], check=False)


def git_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() or "0" * 40


def bundle(out_dir: Path, manifest: dict, repo: Path = REPO) -> Path | None:
    """Zip runs/, each experiment's results/ folder, figures, the PTB-XL report and any synthetic banks;
    None if nothing ran."""
    staging = out_dir / "_bundle"
    shutil.rmtree(staging, ignore_errors=True)
    found = False
    for rel in ("runs", *(f"results/{name}" for name in RESULT_FOLDERS)):
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
    if found and (repo / ACCESS_LOG).is_file():  # DS2 reads made here (leakage audit) must reach the team's log
        (staging / ACCESS_LOG).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / ACCESS_LOG, staging / ACCESS_LOG)
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
    paths = [p for p in (*RESULT_PATHS, *(f"results/{e}" for e in RESULT_FOLDERS)) if (repo / p).exists()]
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
    parser.add_argument("--raw-dir", default="/kaggle/input", help="folder searched for the raw MIT-BIH files (the attached mitdb-raw dataset)")
    parser.add_argument("--no-download", action="store_true", help="refuse instead of downloading when no raw files are attached")
    parser.add_argument("--artifacts-dir", default=None, help="attached dataset holding banks and run checkpoints (phase 4 and 5 runs need them)")
    parser.add_argument("--data-dir", default=None, help="attached dataset holding a data/ folder (external databases, processed arrays)")
    parser.add_argument("--tests", action="store_true", help="run the full test suite first and stop if it fails")
    parser.add_argument("--out", default="/kaggle/working", help="where the bundle zip is written")
    parser.add_argument("--push-to", default=None, metavar="BRANCH", help="commit the outputs and push them to this branch")
    parser.add_argument("--git-name", default="", help="author name for the results commit")
    parser.add_argument("--git-email", default="", help="author email for the results commit")
    parser.add_argument("extra", nargs="*", help="arguments after -- are passed to the CLI")
    args = parser.parse_args(argv)
    os.environ["MPLBACKEND"] = "Agg"  # Kaggle sets its notebook backend, which the project environment does not have

    run = setup_env(args.env, dev=args.tests)
    if "--smoke" not in args.extra:  # smoke runs make their own fake records
        try:
            setup_data(args.raw_dir, mirror=None if args.no_download else RAW_MIRROR)
        except (FileNotFoundError, RuntimeError) as err:
            print(f"!! {err}", flush=True)
            return 2
    if args.artifacts_dir:
        copied = setup_artifacts(args.artifacts_dir)
        print(f"artifacts copied from {args.artifacts_dir}: {len(copied)} files", flush=True)
    if args.data_dir:
        copied = setup_data_dir(args.data_dir)
        print(f"data copied from {args.data_dir}: {len(copied)} files", flush=True)

    results = {}
    tests = run_tests(run) if args.tests else None
    if tests:
        print(f"!! the test suite failed (exit code {tests}); nothing else runs", flush=True)
        results["tests"] = tests
    for config in ([] if tests else args.config):
        code = sh([*run, args.cli, "--config", config, *args.extra], check=False)
        results[config] = code
        if code != 0:
            print(f"!! {config} exited with code {code}; stopping here", flush=True)
            break

    manifest = {"git_sha": git_sha(), "cli": args.cli, "env": args.env, "extra": args.extra, "exit_codes": results, "tests_exit_code": tests}
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
