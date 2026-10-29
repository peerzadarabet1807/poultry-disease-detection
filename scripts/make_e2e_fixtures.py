"""Small derived files for the Playwright tests (web/tests/fixtures/), made from the CC BY sample photos.

* sample.heic          a sample re-encoded as HEIC (the iPhone default)
* exif-rotated.jpg     a sample with an EXIF orientation tag of 6 (90 degrees); its pixels are unchanged
* not-an-image.txt     plain text, for the validation error

Usage:
    python scripts/make_e2e_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "web" / "public" / "samples"
OUT = ROOT / "web" / "tests" / "fixtures"


def main() -> int:
    import pillow_heif

    pillow_heif.register_heif_opener()
    with Image.open(SAMPLES / "f10.jpg") as im:
        small = im.convert("RGB")
        small.thumbnail((512, 512))
        small.save(OUT / "sample.heic", format="HEIF", quality=90)
    with Image.open(SAMPLES / "f05.jpg") as im:
        exif = Image.Exif()
        exif[0x0112] = 6
        im.save(OUT / "exif-rotated.jpg", format="JPEG", quality=95, exif=exif.tobytes())
    (OUT / "not-an-image.txt").write_text("This is not a photo.\n", encoding="utf-8", newline="\n")
    for f in ("sample.heic", "exif-rotated.jpg", "not-an-image.txt"):
        print(f, (OUT / f).stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
