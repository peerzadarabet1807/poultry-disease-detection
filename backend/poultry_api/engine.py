"""The model behind the API, loaded once at startup. Tests substitute a stub with the same interface."""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np


class Engine(Protocol):
    def predict(self, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(224, 224, 3) uint8 -> (probabilities (4,), pre-ReLU class maps (7, 7, 4))."""
        ...


class KerasEngine:
    """The released Keras model with its in-graph CAM head (poultry_disease.Predictor)."""

    def __init__(self, allow_download: bool = True):
        from poultry_disease import Predictor

        self.predictor = Predictor.from_release(allow_download=allow_download)
        self.predictor.warm_up()
        # One inference at a time: TensorFlow already uses every core for a single image, and this keeps
        # peak memory flat under concurrent requests.
        self._lock = threading.Lock()

    def predict(self, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            pred = self.predictor.predict_seen(seen)
        return pred.probs, pred.cam
