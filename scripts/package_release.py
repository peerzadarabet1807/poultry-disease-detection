"""Package the trained model for a GitHub Release and write models/checksums.sha256.

The SavedModel directory is zipped deterministically (sorted entries, fixed timestamps, fixed permissions),
so re-running this script on the same files produces a byte-identical archive and the same checksum.

Usage:
    python scripts/package_release.py --run downloads/extracted_191001/poultry-output/run_20261029_191001
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERAS_NAME = "poultry_resnet50.keras"
SAVEDMODEL_NAME = "poultry_resnet50_savedmodel"
FIXED_TIME = (2026, 10, 29, 19, 10, 2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_dir(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        entries = sorted(p for p in src.rglob("*"))
        for p in entries:
            arc = f"{SAVEDMODEL_NAME}/{p.relative_to(src).as_posix()}"
            if p.is_dir():
                arc += "/"
            info = zipfile.ZipInfo(arc, date_time=FIXED_TIME)
            info.external_attr = (0o40755 << 16 | 0x10) if p.is_dir() else (0o100644 << 16)
            info.compress_type = zipfile.ZIP_STORED if p.is_dir() else zipfile.ZIP_DEFLATED
            zf.writestr(info, b"" if p.is_dir() else p.read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="extracted run folder (contains models/)")
    ap.add_argument("--out", default=str(ROOT / ".work" / "release"))
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keras_out = out / KERAS_NAME
    shutil.copyfile(run / "models" / KERAS_NAME, keras_out)
    zip_out = out / f"{SAVEDMODEL_NAME}.zip"
    zip_dir(run / "models" / SAVEDMODEL_NAME, zip_out)

    lines = [f"{sha256(p)}  {p.name}" for p in (keras_out, zip_out)]
    (ROOT / "models" / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    for p, line in zip((keras_out, zip_out), lines, strict=False):
        print(f"{line}  ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
