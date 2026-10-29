from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def weights_path() -> Path | None:
    """The .keras file from $POULTRY_DISEASE_WEIGHTS or the release cache, if present (never downloads)."""
    from poultry_disease.weights import cache_dir

    explicit = os.environ.get("POULTRY_DISEASE_WEIGHTS")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(cache_dir() / "poultry_resnet50.keras")
    return next((p for p in candidates if p.exists()), None)


@pytest.fixture(scope="session")
def predictor():
    path = weights_path()
    if path is None:
        pytest.skip("weights not available (set POULTRY_DISEASE_WEIGHTS or run scripts/fetch_weights.py)")
    from poultry_disease import Predictor
    from poultry_disease.weights import verify

    return Predictor(verify(path))


@pytest.fixture(scope="session")
def fixture_manifest() -> list[dict]:
    path = FIXTURES / "fixtures.json"
    if not path.exists():
        pytest.skip("tests/fixtures/fixtures.json not found")
    rows = json.loads(path.read_text("utf-8"))["images"]
    missing = [r["file"] for r in rows if not (FIXTURES / "images" / r["file"]).exists()]
    if missing:
        pytest.skip(f"fixture images missing: {missing[:3]}")
    return rows
