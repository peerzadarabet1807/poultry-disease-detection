"""Fetch individual dataset images from the Zenodo zips with HTTP range requests, and verify them.

The dataset is about 13 GB of zips (one per class folder). Zenodo serves them with ``Accept-Ranges: bytes``,
so this script reads each zip's central directory remotely and downloads only the entries it needs. Every
image is checked against the MD5 recorded in results/dataset_manifest.csv by the training notebook, so the
bytes are exactly the ones the model was trained and evaluated on.

Usage:
    python scripts/fetch_zenodo_images.py --selection tests/fixtures/selection.json
    python scripts/fetch_zenodo_images.py --set internal --set external --workers 6   # all 2,175 test images

Images land in data/zenodo/<relpath> (gitignored). Files already present with the right MD5 are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import time
import urllib.request
import zipfile
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUT = ROOT / "data" / "zenodo"

# dataset folder -> (Zenodo record, zip file). Version 2: DOI 10.5281/zenodo.4628934, version 3 (PCR
# annotated): DOI 10.5281/zenodo.5801834. Both CC BY 4.0, Machuve et al.
RECORDS = {"version-2": "4628934", "version-3": "5801834"}
URL = "https://zenodo.org/api/records/{record}/files/{name}/content"
SETS = {"internal": "test_predictions.csv", "external": "pcr_predictions.csv"}


class RangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, fetched in cached blocks with HTTP range requests."""

    def __init__(self, url: str, block: int = 1 << 20, cache_blocks: int = 64):
        self.url, self.block, self.cache_blocks = url, block, cache_blocks
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed https URL)
            if r.status != 206:
                raise OSError(f"{url}: server ignored the range request (HTTP {r.status})")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _fetch(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block
        end = min(start + self.block, self.size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                    data = r.read()
                if len(data) != end - start + 1:
                    raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
                break
            except (OSError, http.client.HTTPException):
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        self.cache[index] = data
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            index, offset = divmod(self.pos, self.block)
            chunk = self._fetch(index)[offset : offset + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324 (matches the notebook's manifest, not a security use)


def fetch_from_zip(version: str, folder: str, wanted: list[tuple[str, str]], out: Path) -> list[str]:
    """Download ``wanted`` [(relpath, md5)] from one class zip. Returns the relpaths written."""
    url = URL.format(record=RECORDS[version], name=f"{folder}.zip")
    zf = zipfile.ZipFile(RangeFile(url))
    by_name = defaultdict(list)
    for info in zf.infolist():
        if not info.is_dir():
            by_name[info.filename.rsplit("/", 1)[-1]].append(info)
    written = []
    for relpath, expected in wanted:
        candidates = by_name.get(relpath.rsplit("/", 1)[-1], [])
        if len(candidates) != 1:
            raise LookupError(f"{relpath}: {len(candidates)} entries with that file name in {folder}.zip")
        data = zf.read(candidates[0])
        if md5(data) != expected:
            raise ValueError(f"{relpath}: MD5 {md5(data)} does not match the manifest ({expected})")
        dest = out / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written.append(relpath)
        print(f"  {relpath} ({len(data) / 1e6:.1f} MB) ok", flush=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, help="JSON with an 'images' list of {relpath}")
    ap.add_argument("--set", action="append", choices=sorted(SETS), default=[], help="all images of a test set")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=1, help="parallel downloads (each opens its own zip)")
    args = ap.parse_args()

    relpaths: list[str] = []
    if args.selection:
        relpaths += [im["relpath"] for im in json.loads(args.selection.read_text("utf-8"))["images"]]
    for s in args.set:
        relpaths += pd.read_csv(RESULTS / SETS[s], usecols=["relpath"])["relpath"].tolist()
    if not relpaths:
        ap.error("nothing to fetch: pass --selection and/or --set")

    manifest = pd.read_csv(RESULTS / "dataset_manifest.csv", usecols=["relpath", "md5"]).set_index("relpath")["md5"]
    todo = defaultdict(list)
    skipped = 0
    for rel in dict.fromkeys(relpaths):
        dest = args.out / rel
        if dest.exists() and md5(dest.read_bytes()) == manifest[rel]:
            skipped += 1
            continue
        version, folder, _ = rel.split("/", 2)
        todo[(version, folder)].append((rel, manifest[rel]))
    print(f"{skipped} already present, {sum(map(len, todo.values()))} to fetch from {len(todo)} zip(s)")

    # split each zip's list into chunks so several workers can read the same zip in parallel
    jobs = []
    for (version, folder), items in sorted(todo.items()):
        n = max(1, min(args.workers, len(items) // 20 or 1))
        jobs += [(version, folder, items[i::n]) for i in range(n)]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        done = sum(len(w) for w in pool.map(lambda j: fetch_from_zip(*j, args.out), jobs))
    print(f"fetched {done}, all MD5s match results/dataset_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
