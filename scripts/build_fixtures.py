"""Copy the selected dataset images into tests/fixtures/images/ and write tests/fixtures/fixtures.json.

Run after scripts/select_samples.py and scripts/fetch_zenodo_images.py --selection tests/fixtures/selection.json.
The images are the original, unmodified files (MD5-checked against results/dataset_manifest.csv), so the
parity tests exercise the full decode and resize path on real phone photos. They are CC BY 4.0, see
tests/fixtures/README.md for the attribution.

Usage:
    python scripts/build_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SOURCE = ROOT / "data" / "zenodo"
DOIS = {"version-2": "10.5281/zenodo.4628934", "version-3": "10.5281/zenodo.5801834"}


def main() -> int:
    selection = json.loads((FIXTURES / "selection.json").read_text("utf-8"))["images"]
    images = FIXTURES / "images"
    shutil.rmtree(images, ignore_errors=True)
    images.mkdir(parents=True)
    rows = []
    for im in selection:
        src = SOURCE / im["relpath"]
        data = src.read_bytes()
        if hashlib.md5(data).hexdigest() != im["md5"]:  # noqa: S324 (manifest checksum)
            raise ValueError(f"{src}: MD5 does not match the manifest; re-run scripts/fetch_zenodo_images.py")
        name = f"{im['id']}_{src.name}"
        (images / name).write_bytes(data)
        rows.append({"file": name, **im, "source_doi": DOIS[im["relpath"].split("/", 1)[0]], "license": "CC-BY-4.0"})
    out = {
        "description": (
            "Parity fixtures: original dataset images with the probabilities the training notebook recorded "
            "for them in Colab (results/test_predictions.csv, results/pcr_predictions.csv)."
        ),
        "run": "run_20261029_191001",
        "images": rows,
    }
    (FIXTURES / "fixtures.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    total = sum(p.stat().st_size for p in images.iterdir())
    print(f"{len(rows)} images -> {images} ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
