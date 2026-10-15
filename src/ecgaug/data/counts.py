"""Per-record AAMI class counts and the sanity check against the literature counts.

Usage:
    python -m ecgaug.data.counts                  # DS1 only
    ALLOW_TEST_BUILD=1 python -m ecgaug.data.counts --include-test

Counting DS2 labels is a construction access: it reads annotations only, involves no model and is
appended to docs/TEST_ACCESS_LOG.md.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ecgaug import paths
from ecgaug.names import PHASE1_DATA
from ecgaug.data.access import log_test_access, require_test_access
from ecgaug.data.beats import UNMAPPED, extract_beats
from ecgaug.data.records import CLASSES, DS1, DS2, dataset_of

# CLAUDE.md §4, "Sanity counts". Phase 1 gate asks for class totals within 1% of these.
SANITY_COUNTS: dict[str, dict[str, int]] = {
    "DS1": {"N": 45868, "S": 942, "V": 3787, "F": 415},
    "DS2": {"N": 44258, "S": 1837, "V": 3221, "F": 388},
}
SANITY_TOLERANCE = 0.01
COLUMNS = ["record", "dataset", "N", "S", "V", "F", "Q", "unmapped_qrs", "qrs_total"]


def count_record(record: int, access: str | None = None) -> dict:
    """AAMI class counts of one record, from its reference annotations."""
    beats = extract_beats(record, access)
    tally = beats["aami"].value_counts()
    row = {"record": int(record), "dataset": dataset_of(record)}
    for cls in (*CLASSES, "Q"):
        row[cls] = int(tally.get(cls, 0))
    row["unmapped_qrs"] = int(tally.get(UNMAPPED, 0))
    row["qrs_total"] = int(len(beats))
    return row


def record_counts(include_test: bool = False) -> pd.DataFrame:
    """Counts for every DS1 record, plus every DS2 record when ``include_test`` is set."""
    if include_test:
        require_test_access("construction")  # fail before reading anything
    rows = [count_record(r) for r in DS1]
    if include_test:
        rows += [count_record(r, access="construction") for r in DS2]
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["dataset", "record"]).reset_index(drop=True)


def sanity_check(counts: pd.DataFrame) -> dict:
    """Compare class totals per dataset with the literature counts."""
    report: dict = {"tolerance": SANITY_TOLERANCE, "datasets": {}, "passed": True}
    for dataset, reference in SANITY_COUNTS.items():
        rows = counts[counts["dataset"] == dataset]
        if rows.empty:
            continue
        entry = {"n_records": int(len(rows)), "Q": int(rows["Q"].sum()), "classes": {}}
        entry["unmapped_qrs"] = int(rows["unmapped_qrs"].sum())
        for cls, expected in reference.items():
            observed = int(rows[cls].sum())
            deviation = (observed - expected) / expected
            ok = abs(deviation) <= SANITY_TOLERANCE
            entry["classes"][cls] = {
                "observed": observed,
                "reference": expected,
                "difference": observed - expected,
                "relative_difference": deviation,
                "within_tolerance": ok,
            }
            report["passed"] = report["passed"] and ok
        report["datasets"][dataset] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-record AAMI class counts.")
    parser.add_argument("--include-test", action="store_true", help="also count DS2 (needs ALLOW_TEST_BUILD=1)")
    args = parser.parse_args(argv)

    counts = record_counts(include_test=args.include_test)
    if args.include_test:
        log_test_access("construction", "per-record class counts from DS2 annotations (labels only, no signals)")
    out_dir = paths.results_for(PHASE1_DATA)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / "record_counts.csv", index=False)
    report = sanity_check(counts)
    (out_dir / "sanity_check.json").write_text(json.dumps(report, indent=2) + "\n")

    for dataset, entry in report["datasets"].items():
        parts = [
            f"{cls} {v['observed']} (ref {v['reference']}, {v['relative_difference']:+.2%})"
            for cls, v in entry["classes"].items()
        ]
        print(f"{dataset}: {entry['n_records']} records | " + " | ".join(parts) + f" | Q {entry['Q']}")
    print("sanity check:", "PASS" if report["passed"] else "FAIL")
    print(f"wrote {out_dir / 'record_counts.csv'} and sanity_check.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
