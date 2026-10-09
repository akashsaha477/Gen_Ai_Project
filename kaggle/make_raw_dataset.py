"""Package the verified raw MIT-BIH files for upload as a private Kaggle dataset.

    python kaggle/make_raw_dataset.py            # writes data/kaggle/mitdb-raw.zip

Upload the zip once at kaggle.com/datasets (New Dataset, keep it private, share it with the team),
then attach it to the runner notebook. This avoids the slow PhysioNet download in every session.
The runner still checks every file against PhysioNet's SHA-256 list.
"""

from __future__ import annotations

import sys
import zipfile

from ecgaug import paths
from ecgaug.data import raw


def main() -> int:
    report = raw.verify()
    if report["problems"] or not report["reference"]["available"]:
        print("raw data is not verified; run `python -m ecgaug.data.raw` first:", report["problems"][:5])
        return 1
    out = paths.repo_root() / "data" / "kaggle" / "mitdb-raw.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(report["files"]):
            zf.write(paths.raw_dir() / name, name)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(report['files'])} files, all checksums verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
