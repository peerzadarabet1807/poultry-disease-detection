"""Configuration from environment variables (no secrets are needed)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "https://peerzadarabet1807.github.io",
)


def _origins() -> tuple[str, ...]:
    raw = os.environ.get("POULTRY_API_CORS_ORIGINS")
    if raw is None:
        return DEFAULT_ORIGINS
    return tuple(o.strip().rstrip("/") for o in raw.split(",") if o.strip())


@dataclass(frozen=True)
class Settings:
    #: Exact origins allowed by CORS (comma-separated in POULTRY_API_CORS_ORIGINS). No wildcard.
    cors_origins: tuple[str, ...] = field(default_factory=_origins)
    #: Largest accepted upload, in bytes (the web app uses the same 25 MB limit).
    max_upload_bytes: int = int(float(os.environ.get("POULTRY_API_MAX_UPLOAD_MB", "25")) * 1024 * 1024)
    #: Largest accepted image, in pixels, checked from the header before decoding (decompression bombs).
    max_pixels: int = int(os.environ.get("POULTRY_API_MAX_PIXELS", str(40_000_000)))
    #: Folder that holds results/ and models/ (the repository root, or /srv in the Docker image).
    repo_root: Path = Path(os.environ.get("POULTRY_API_REPO_ROOT", Path(__file__).resolve().parents[2]))
    #: Download the release weights on first start if they are not cached.
    allow_download: bool = os.environ.get("POULTRY_API_ALLOW_DOWNLOAD", "1") not in ("0", "false", "no")
