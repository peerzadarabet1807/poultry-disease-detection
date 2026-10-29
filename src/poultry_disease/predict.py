"""Inference: one forward pass returns the probabilities and the four class activation maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from poultry_disease.cam import normalize_cam
from poultry_disease.constants import CLASS_NAMES
from poultry_disease.model import CAM_OUTPUT, PROBS_OUTPUT, build_inference_model, load_keras_model
from poultry_disease.preprocess import preprocess_bytes, preprocess_pil, to_model_input


@dataclass
class Prediction:
    #: softmax over CLASS_NAMES, in model order
    probs: np.ndarray
    #: pre-ReLU class activation maps, (7, 7, 4)
    cam: np.ndarray
    #: the 224 x 224 uint8 RGB image the model saw
    seen: np.ndarray
    class_names: tuple[str, ...] = field(default=CLASS_NAMES)

    @property
    def top_index(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def predicted_class(self) -> str:
        return self.class_names[self.top_index]

    @property
    def confidence(self) -> float:
        return float(self.probs[self.top_index])

    @property
    def probabilities(self) -> dict[str, float]:
        return {c: float(p) for c, p in zip(self.class_names, self.probs, strict=False)}

    def cam_for(self, class_index: int | None = None) -> np.ndarray:
        """ReLU + max-normalised 7 x 7 map for a class (default: the predicted class)."""
        idx = self.top_index if class_index is None else class_index
        return normalize_cam(self.cam[..., idx])

    def to_dict(self) -> dict:
        """The notebook's ``predict_image`` result: predicted_class, confidence, probabilities."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class Predictor:
    """Wraps the trained model with its CAM head. Load once, call many times."""

    def __init__(self, keras_path: str | Path):
        self.keras_path = Path(keras_path)
        self.trained = load_keras_model(self.keras_path)
        self.model = build_inference_model(self.trained)

    @classmethod
    def from_release(cls, allow_download: bool = True) -> Predictor:
        from poultry_disease.weights import get_weights

        return cls(get_weights(allow_download=allow_download))

    def predict_seen(self, seen: np.ndarray) -> Prediction:
        """From the already resized 224 x 224 uint8 image."""
        out = self.model(to_model_input(seen), training=False)
        return Prediction(
            probs=np.asarray(out[PROBS_OUTPUT])[0].astype(np.float32),
            cam=np.asarray(out[CAM_OUTPUT])[0].astype(np.float32),
            seen=seen,
        )

    def predict_bytes(self, data: bytes) -> Prediction:
        seen, _ = preprocess_bytes(data)
        return self.predict_seen(seen)

    def predict_image(self, path: str | Path) -> Prediction:
        return self.predict_bytes(Path(path).read_bytes())

    def predict_pil_image(self, image: Image.Image) -> Prediction:
        seen, _ = preprocess_pil(image)
        return self.predict_seen(seen)

    def warm_up(self) -> None:
        self.predict_seen(np.zeros((224, 224, 3), np.uint8))
