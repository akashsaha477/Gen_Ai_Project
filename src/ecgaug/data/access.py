"""The test lock (CLAUDE.md rule 2).

DS2 is the locked test set. Reading it needs a declared access kind and the matching environment
flag, and every use is appended to ``docs/TEST_ACCESS_LOG.md``:

- ``construction``: counting labels and building arrays; no model sees the data (``ALLOW_TEST_BUILD=1``).
- ``evaluation``: the single final evaluation, set only by ``/final-eval`` (``ALLOW_TEST=1``).
"""

from __future__ import annotations

import datetime as _dt
import os
import shlex
import sys

from ecgaug import paths

ACCESS_ENV: dict[str, str] = {"construction": "ALLOW_TEST_BUILD", "evaluation": "ALLOW_TEST"}

_LOG_HEADER = (
    "# TEST ACCESS LOG\n\n"
    "Every read of DS2 (the locked test set) is appended here by `ecgaug.data.access`.\n"
    "Kinds: `construction` (labels counted or arrays built, no model involved), `evaluation`\n"
    "(only `/final-eval`), `audit` (leakage-audit arm, only after `/final-eval`).\n\n"
    "| date | kind | git SHA | command | purpose | run IDs | approved by |\n"
    "|---|---|---|---|---|---|---|\n"
)


class LockedTestDataError(RuntimeError):
    """DS2 (test) data was requested without the matching permission."""


def require_test_access(kind: str | None) -> None:
    """Raise :class:`LockedTestDataError` unless ``kind`` is declared and its flag is set to '1'."""
    if kind not in ACCESS_ENV:
        raise LockedTestDataError(
            "DS2 is the locked test set. Reading it needs access='construction' (labels and array "
            "building, with ALLOW_TEST_BUILD=1) or access='evaluation' (only /final-eval, with "
            f"ALLOW_TEST=1). Got access={kind!r}."
        )
    flag = ACCESS_ENV[kind]
    if os.environ.get(flag) != "1":
        raise LockedTestDataError(
            f"DS2 access of kind '{kind}' requires the environment variable {flag}=1, which is not set."
        )


def log_test_access(kind: str, purpose: str, run_ids: str = "", approved_by: str = "") -> None:
    """Append one row to ``docs/TEST_ACCESS_LOG.md`` (created with its header if missing)."""
    from ecgaug.runs import git_sha  # local import: runs imports nothing from data

    log = paths.docs_dir() / "TEST_ACCESS_LOG.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text(_LOG_HEADER)
    root = str(paths.repo_root()) + "/"
    command = " ".join(shlex.quote(a.replace(root, "")) for a in sys.argv) or "(interactive)"
    cells = [
        _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z"),
        kind,
        git_sha(short=True),
        f"`{command}`",
        purpose,
        run_ids,
        approved_by,
    ]
    row = "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |\n"
    with log.open("a") as fh:
        fh.write(row)
