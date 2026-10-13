"""Raw MIT-BIH access: download, verification, and reading signals and annotations.

This is the only module that imports ``wfdb`` or opens files under ``data/raw`` (enforced by
``tests/test_test_lock.py``). Every read of a DS2 record goes through :func:`_guard`, so the test
lock cannot be bypassed by another part of the package.

Usage: ``python -m ecgaug.data.raw`` downloads the database and verifies it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import wfdb

from ecgaug import paths
from ecgaug.data.access import require_test_access
from ecgaug.data.records import ALL_RECORDS, DS1, DS2, FS, LEAD, PACED

EXTENSIONS: tuple[str, ...] = ("hea", "dat", "atr")
REFERENCE_SHA256_URL = "https://physionet.org/files/mitdb/1.0.0/SHA256SUMS.txt"


def _guard(record: int, access: str | None) -> int:
    """Validate ``record`` and enforce the test lock for DS2 records."""
    record = int(record)
    if record in DS2:
        require_test_access(access)
    elif record in PACED:
        raise ValueError(f"Record {record} is paced and excluded by the protocol.")
    elif record not in DS1:
        raise ValueError(f"{record} is not an MIT-BIH Arrhythmia Database record.")
    return record


def _stem(record: int) -> str:
    return str(paths.raw_dir() / str(record))


def download(dest: Path | None = None) -> Path:
    """Download the MIT-BIH Arrhythmia Database from PhysioNet into ``data/raw/mitdb``."""
    dest = dest or paths.raw_dir()
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(dest))
    return dest


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_reference_checksums(timeout: float = 30.0) -> dict[str, str] | None:
    """PhysioNet's published SHA-256 list as {filename: hash}, or None if it cannot be fetched."""
    try:
        request = urllib.request.Request(REFERENCE_SHA256_URL, headers={"User-Agent": "ecgaug"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (fixed https URL)
            text = response.read().decode()
    except OSError:
        return None
    reference = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            reference[parts[1]] = parts[0]
    return reference


def read_header(record: int) -> dict:
    """Technical header fields of any of the 48 records (no signal or label data, so not locked)."""
    if int(record) not in ALL_RECORDS:
        raise ValueError(f"{record} is not an MIT-BIH Arrhythmia Database record.")
    header = wfdb.rdheader(_stem(int(record)))
    return {"fs": int(header.fs), "sig_name": list(header.sig_name), "sig_len": int(header.sig_len)}


def verify(fetch_reference: bool = True) -> dict:
    """Check that all 48 records are complete and hash every file.

    Hashing reads raw bytes without decoding them, so it is not a test-data access. Returns a report
    whose ``problems`` list is empty when the download is sound.
    """
    raw = paths.raw_dir()
    problems: list[str] = []
    files: dict[str, str] = {}
    for record in ALL_RECORDS:
        for ext in EXTENSIONS:
            path = raw / f"{record}.{ext}"
            if path.is_file():
                files[path.name] = file_sha256(path)
            else:
                problems.append(f"missing file {path.name}")

    reference = _fetch_reference_checksums() if fetch_reference else None
    mismatched = []
    matched = 0
    if reference is not None:
        for name, digest in files.items():
            if name not in reference:
                problems.append(f"{name} is not in PhysioNet's checksum list")
            elif reference[name] != digest:
                mismatched.append(name)
            else:
                matched += 1
        problems.extend(f"checksum mismatch for {name}" for name in mismatched)

    headers = {}
    for record in ALL_RECORDS:
        if not (raw / f"{record}.hea").is_file():
            continue
        info = read_header(record)
        headers[str(record)] = info
        if info["fs"] != FS:
            problems.append(f"record {record}: fs {info['fs']} != {FS}")
        if record not in PACED and LEAD not in info["sig_name"]:
            problems.append(f"record {record}: lead {LEAD} missing (has {info['sig_name']})")

    return {
        "database": "MIT-BIH Arrhythmia Database 1.0.0 (PhysioNet)",
        "n_records_expected": len(ALL_RECORDS),
        "n_files_hashed": len(files),
        "reference": {
            "url": REFERENCE_SHA256_URL,
            "available": reference is not None,
            "n_matched": matched,
            "mismatched": mismatched,
        },
        "problems": problems,
        "headers": headers,
        "files": files,
    }


def read_annotations(record: int, access: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Reference annotations of a record: sample indices and symbols.

    DS2 records need ``access='construction'`` or ``access='evaluation'`` plus the matching
    environment flag (see :mod:`ecgaug.data.access`).
    """
    record = _guard(record, access)
    ann = wfdb.rdann(_stem(record), "atr")
    return np.asarray(ann.sample, dtype=np.int64), np.asarray(ann.symbol, dtype=str)


def read_signal(record: int, access: str | None = None, lead: str = LEAD) -> np.ndarray:
    """The ``lead`` channel of a record in physical units, selected by signal name.

    Channel order differs between records (114 lists MLII second), so the index is never assumed.
    """
    record = _guard(record, access)
    rec = wfdb.rdrecord(_stem(record), channel_names=[lead])
    if rec.sig_name != [lead]:
        raise ValueError(f"Record {record}: expected lead {lead}, header has {rec.sig_name}")
    if int(rec.fs) != FS:
        raise ValueError(f"Record {record}: fs {rec.fs} != {FS}")
    return np.asarray(rec.p_signal[:, 0], dtype=np.float64)


def read_ptbxl_lead(root: Path, filename: str, lead: str = "II") -> tuple[np.ndarray, int]:
    """One lead of a PTB-XL recording in mV, selected by name, with its sampling rate.

    PTB-XL is a separate database used only for pretraining; this cannot reach MIT-BIH files.
    """
    rec = wfdb.rdrecord(str(Path(root) / filename), channel_names=[lead])
    if rec.sig_name != [lead]:
        raise ValueError(f"{filename}: expected lead {lead}, header has {rec.sig_name}")
    return np.asarray(rec.p_signal[:, 0], dtype=np.float64), int(rec.fs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify the MIT-BIH Arrhythmia Database.")
    parser.add_argument("--skip-download", action="store_true", help="only verify what is on disk")
    args = parser.parse_args(argv)
    if not args.skip_download:
        download()
    report = verify()
    out = paths.results_dir() / "data" / "raw_checksums.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ref = report["reference"]
    print(f"files hashed: {report['n_files_hashed']} (expected {len(ALL_RECORDS) * len(EXTENSIONS)})")
    print(f"PhysioNet checksums: available={ref['available']}, matched={ref['n_matched']}")
    print(f"problems: {len(report['problems'])}")
    for problem in report["problems"]:
        print("  -", problem)
    print(f"wrote {out}")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
