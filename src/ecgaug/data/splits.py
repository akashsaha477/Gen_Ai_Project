"""Patient-wise splits: definition, leakage checks, and the validation proposal report.

Usage: ``python -m ecgaug.data.splits`` writes ``results/data/val_options.md`` from DS1 counts only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from ecgaug import paths
from ecgaug.data.records import CLASSES, DS1, DS2, PACED
from ecgaug.eval.selection import MIN_F_BEATS, MIN_F_RECORDS, selection_metric_name


@dataclass(frozen=True)
class Split:
    """A patient-wise split. ``train`` and ``val`` come from DS1; ``test`` is always DS2."""

    name: str
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...] = DS2
    selection_metric: str | None = None
    meta: dict = field(default_factory=dict, compare=False)

    def split_of(self) -> dict[int, str]:
        """Map each DS1 record of this split to 'train' or 'val'."""
        return {**{r: "train" for r in self.train}, **{r: "val" for r in self.val}}


def check_split(split: Split, require_all_ds1: bool = True) -> None:
    """Raise ``ValueError`` on any leakage or protocol violation in ``split``."""
    train, val, test = set(split.train), set(split.val), set(split.test)
    if len(train) != len(split.train) or len(val) != len(split.val) or len(test) != len(split.test):
        raise ValueError(f"Split {split.name}: a record is listed twice within a split")
    for a, b, label in ((train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test")):
        if a & b:
            raise ValueError(f"Split {split.name}: records {sorted(a & b)} appear in both {label}")
    if (train | val | test) & set(PACED):
        raise ValueError(f"Split {split.name}: paced records are excluded by the protocol")
    if not (train | val) <= set(DS1):
        raise ValueError(f"Split {split.name}: train and validation must be DS1 records only")
    if test != set(DS2):
        raise ValueError(f"Split {split.name}: the test set must be exactly DS2")
    if not train or not val:
        raise ValueError(f"Split {split.name}: train and validation must both be non-empty")
    if require_all_ds1 and (train | val) != set(DS1):
        raise ValueError(f"Split {split.name}: DS1 records {sorted(set(DS1) - train - val)} are unassigned")


def make_split(name: str, val: list[int] | tuple[int, ...], **kwargs) -> Split:
    """Build a split of all DS1 records from its validation record list and check it."""
    val_t = tuple(sorted(int(r) for r in val))
    train_t = tuple(r for r in DS1 if r not in val_t)
    split = Split(name=name, train=train_t, val=val_t, **kwargs)
    check_split(split)
    return split


def approved_split_path() -> Path:
    return paths.configs_dir() / "validation_split" / "approved.yaml"


def load_approved_split(counts: pd.DataFrame | None = None) -> Split:
    """The validation split approved at Gate 1 (``configs/validation_split/approved.yaml``).

    If ``counts`` (per-record class counts) is given, the file's selection metric is checked
    against the minimum-support rule and a contradiction raises ``ValueError``.
    """
    path = approved_split_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. The validation split must be approved by the team at the "
            "Phase 1 decision point before any Module 1 variant runs."
        )
    data = yaml.safe_load(path.read_text())
    if not data.get("approved_by"):
        raise ValueError(f"{path}: 'approved_by' is empty; the split has not been approved")
    split = make_split(
        data["name"],
        data["val"],
        selection_metric=data["selection_metric"],
        meta={k: data.get(k) for k in ("approved_by", "approved_on", "decision")},
    )
    if counts is not None:
        expected = selection_metric_for(counts, split)
        if expected != split.selection_metric:
            raise ValueError(
                f"{path}: selection_metric is {split.selection_metric!r} but the minimum-support "
                f"rule gives {expected!r} for this split"
            )
    return split


def selection_metric_for(counts: pd.DataFrame, split: Split) -> str:
    """Apply the minimum-F-support rule to the validation records of ``split``."""
    val = counts[counts["record"].isin(split.val)]
    return selection_metric_name(int(val["F"].sum()), int((val["F"] > 0).sum()))


def summarize(counts: pd.DataFrame, split: Split) -> dict:
    """Class counts, shares and concentration for the train and validation parts of ``split``."""
    ds1 = counts[counts["dataset"] == "DS1"]
    summary: dict = {"name": split.name, "val_records": list(split.val), "train_records": list(split.train)}
    for part, records in (("train", split.train), ("val", split.val)):
        rows = ds1[ds1["record"].isin(records)]
        entry = {"n_records": int(len(rows)), "classes": {}}
        for cls in CLASSES:
            total, ds1_total = int(rows[cls].sum()), int(ds1[cls].sum())
            carriers = rows[rows[cls] > 0]
            top = carriers.sort_values(cls, ascending=False).head(1)
            entry["classes"][cls] = {
                "beats": total,
                "share_of_ds1": total / ds1_total if ds1_total else 0.0,
                "n_records_with_class": int(len(carriers)),
                "largest_record": int(top["record"].iloc[0]) if len(top) else None,
                "largest_record_share": float(top[cls].iloc[0] / total) if total else 0.0,
            }
        entry["beats_total"] = int(rows[list(CLASSES)].to_numpy().sum())
        summary[part] = entry
    summary["selection_metric"] = selection_metric_for(counts, split)
    return summary


def _row(part: dict) -> str:
    cells = []
    for cls in CLASSES:
        c = part["classes"][cls]
        cells.append(f"{c['beats']:,} ({c['share_of_ds1']:.0%})")
    return " | ".join(cells)


def options_report(counts: pd.DataFrame, options: list[dict]) -> str:
    """Markdown report of the validation options. Uses DS1 rows of ``counts`` only."""
    ds1 = counts[counts["dataset"] == "DS1"].sort_values("record")
    lines = [
        "# Validation-split options (Phase 1 decision point a)",
        "",
        "Generated by `python -m ecgaug.data.splits` from `results/data/record_counts.csv`.",
        "Only DS1 rows are read: the validation split is justified from DS1 counts alone.",
        "",
        "## DS1 per-record class counts",
        "",
        "| record | N | S | V | F | Q | share of DS1 S | share of DS1 F |",
        "|---|---|---|---|---|---|---|---|",
    ]
    s_total, f_total = int(ds1["S"].sum()), int(ds1["F"].sum())
    for _, r in ds1.iterrows():
        lines.append(
            f"| {r['record']} | {r['N']:,} | {r['S']} | {r['V']} | {r['F']} | {r['Q']} | "
            f"{r['S'] / s_total:.1%} | {r['F'] / f_total:.1%} |"
        )
    totals = {c: int(ds1[c].sum()) for c in (*CLASSES, "Q")}
    lines.append(
        f"| **DS1** | **{totals['N']:,}** | **{totals['S']}** | **{totals['V']:,}** | **{totals['F']}** | "
        f"**{totals['Q']}** | 100% | 100% |"
    )
    lines += [
        "",
        "## Options",
        "",
        f"Minimum-support rule (PREREGISTRATION §3, draft): F enters the selection metric only if "
        f"validation holds at least {MIN_F_BEATS} F beats from at least {MIN_F_RECORDS} records.",
        "",
    ]
    for option in options:
        split = make_split(option["name"], option["val"])
        s = summarize(counts, split)
        f_val = s["val"]["classes"]["F"]
        s_val = s["val"]["classes"]["S"]
        s_train = s["train"]["classes"]["S"]
        f_train = s["train"]["classes"]["F"]
        lines += [
            f"### {option['name']}: validation = {list(split.val)}",
            "",
            option.get("rationale", "").strip(),
            "",
            "| part | records | N | S | V | F |",
            "|---|---|---|---|---|---|",
            f"| train | {s['train']['n_records']} | {_row(s['train'])} |",
            f"| validation | {s['val']['n_records']} | {_row(s['val'])} |",
            "",
            f"- Validation S: {s_val['beats']} beats in {s_val['n_records_with_class']} records; "
            f"largest is record {s_val['largest_record']} with {s_val['largest_record_share']:.0%}.",
            f"- Validation F: {f_val['beats']} beats in {f_val['n_records_with_class']} records; "
            f"largest is record {f_val['largest_record']} with {f_val['largest_record_share']:.0%}.",
            f"- Training S: {s_train['beats']} beats in {s_train['n_records_with_class']} records; "
            f"largest is record {s_train['largest_record']} with {s_train['largest_record_share']:.0%}.",
            f"- Training F: {f_train['beats']} beats in {f_train['n_records_with_class']} records; "
            f"largest is record {f_train['largest_record']} with {f_train['largest_record_share']:.0%}.",
            f"- Selection metric under the minimum-support rule: `{s['selection_metric']}`.",
            f"- Record 201 (same subject as test record 202) is in "
            f"{'validation' if 201 in split.val else 'training'}.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-split options report (DS1 counts only).")
    parser.add_argument("--options", default=str(paths.configs_dir() / "validation_split" / "options.yaml"))
    args = parser.parse_args(argv)
    counts = pd.read_csv(paths.results_dir() / "data" / "record_counts.csv")
    counts = counts[counts["dataset"] == "DS1"]  # DS2 rows are never loaded into the proposal
    options = yaml.safe_load(Path(args.options).read_text())["options"]
    out = paths.results_dir() / "data" / "val_options.md"
    out.write_text(options_report(counts, options))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
