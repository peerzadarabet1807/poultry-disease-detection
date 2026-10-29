"""Parity through the HTTP path: the API returns the probabilities the notebook recorded in Colab."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poultry_api.main import create_app
from poultry_api.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
PROB_TOLERANCE = 1e-4  # as tests/test_parity.py

pytestmark = [pytest.mark.weights, pytest.mark.fixtures]


@pytest.fixture(scope="module")
def client():
    from tests.conftest import weights_path

    if weights_path() is None:
        pytest.skip("weights not available")
    if not (FIXTURES / "fixtures.json").exists():
        pytest.skip("fixtures not available")
    with TestClient(create_app(Settings(allow_download=False))) as c:
        yield c


def test_fixtures_through_http(client):
    images = json.loads((FIXTURES / "fixtures.json").read_text("utf-8"))["images"]
    worst = 0.0
    for im in images:
        data = (FIXTURES / "images" / im["file"]).read_bytes()
        r = client.post("/api/v1/predict", files={"file": (im["file"], data, "image/jpeg")}, params={"cam": "true"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["predicted_class"] == im["colab_predicted_class"], im["file"]
        diff = max(abs(body["probabilities"][c] - p) for c, p in im["colab_probabilities"].items())
        assert diff < PROB_TOLERANCE, f"{im['file']}: {diff:.2e}"
        worst = max(worst, diff)
        top = np.asarray(body["cam"]["maps"][body["predicted_class"]])
        assert top.max() == pytest.approx(1.0)
        assert body["image"]["width"] == im["width"] and body["image"]["height"] == im["height"]
    print(f"API vs Colab max abs probability difference: {worst:.2e}")
