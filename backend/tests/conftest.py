from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from poultry_api.main import create_app
from poultry_api.settings import Settings

ALLOWED_ORIGIN = "https://peerzadarabet1807.github.io"


class StubEngine:
    """Fixed outputs, so request handling can be tested without the weights."""

    def __init__(self, probs=(0.02, 0.01, 0.95, 0.02)):
        self.probs = np.asarray(probs, dtype=np.float32)
        self.calls = 0

    def predict(self, seen: np.ndarray):
        assert seen.shape == (224, 224, 3) and seen.dtype == np.uint8
        self.calls += 1
        cam = np.linspace(-1, 1, 7 * 7 * 4, dtype=np.float32).reshape(7, 7, 4)
        return self.probs, cam


def make_client(engine=None, **overrides) -> TestClient:
    settings = replace(Settings(), cors_origins=(ALLOWED_ORIGIN,), **overrides)
    engine = engine or StubEngine()
    return TestClient(create_app(settings, engine_factory=lambda _: engine))


@pytest.fixture
def stub_client():
    with make_client() as client:
        yield client


def encode(fmt: str, size=(64, 48), mode="RGB", **kw) -> bytes:
    rng = np.random.default_rng(0)
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    arr = (rng.random((size[1], size[0], channels)) * 255).astype(np.uint8)
    im = Image.fromarray(arr[..., 0] if channels == 1 else arr, mode)
    buf = io.BytesIO()
    im.save(buf, format=fmt, **kw)
    return buf.getvalue()
