"""Bring Kaggle run bundles back into the repository and update results/runs_index.csv.

Usage:
    python kaggle/ingest_runs.py <bundle.zip | directory> [...]

Runs go to runs/<run_id>/ and variant result files to results/validation/. Existing runs and result
files are never overwritten. The logic lives in ecgaug.runs.ingest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecgaug import runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="zip bundles or directories produced by the Kaggle runner")
    args = parser.parse_args(argv)
    report = runs.ingest([Path(source) for source in args.sources])
    print(json.dumps(report, indent=2))
    if report["refused"]:
        print(
            "REFUSED: the runs above carry test predictions. Nothing was ingested for them. "
            "Stop and report this to the team (CLAUDE.md rule 2)."
        )
    problems = report["refused"] or report["conflicts"] or report["incomplete"] or report["result_conflicts"]
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
