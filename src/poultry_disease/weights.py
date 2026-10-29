"""Fetch the released Keras weights, verify their SHA-256 and cache them.

Lookup order:
1. ``$POULTRY_DISEASE_WEIGHTS``: a local ``.keras`` file (still checksum-verified).
2. The cache: ``$POULTRY_DISEASE_CACHE`` or ``~/.cache/poultry-disease``.
3. Download from the GitHub Release into the cache.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from poultry_disease.constants import RELEASE_BASE_URL, WEIGHTS_FILE, WEIGHTS_SHA256


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir() -> Path:
    return Path(os.environ.get("POULTRY_DISEASE_CACHE", Path.home() / ".cache" / "poultry-disease"))


def verify(path: str | Path, expected: str = WEIGHTS_SHA256) -> Path:
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumError(f"{path}: SHA-256 {actual} does not match the release checksum {expected}")
    return Path(path)


def download(url: str, dest: Path, expected: str = WEIGHTS_SHA256, timeout: float = 60.0) -> Path:
    """Download to a temporary file next to ``dest``, verify, then move into place atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            shutil.copyfileobj(resp, out, length=1 << 20)
        verify(tmp, expected)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def get_weights(allow_download: bool = True) -> Path:
    """Path to a verified ``poultry_resnet50.keras``, downloading it once if necessary."""
    explicit = os.environ.get("POULTRY_DISEASE_WEIGHTS")
    if explicit:
        return verify(explicit)
    cached = cache_dir() / WEIGHTS_FILE
    if cached.exists():
        return verify(cached)
    if not allow_download:
        raise FileNotFoundError(f"{cached} not found and downloads are disabled")
    return download(f"{RELEASE_BASE_URL}/{WEIGHTS_FILE}", cached)
