"""The input check's decision rule, mirrored in TypeScript by the web app (web/src/lib/inputCheck.ts).

The check answers one question: does this photo look like the poultry droppings photos the classifier was
trained on? It never changes the classifier's answer. The image embedding comes from the TinyCLIP ViT-8M/16
image encoder (run in the browser with TensorFlow.js; the Python reference is scripts/gate_study.py). This
module holds only the rule, so the package gains no dependency:

    score  = max over the four classes of cosine(embedding, class prototype)
    passed = score >= threshold

Prototypes, threshold and model details live in ``data/input_check.json`` (scripts/build_input_check.py).
A safeguard, not a guarantee: the measured rates are in results/input_check/evaluation.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

import numpy as np

CONFIG: dict = json.loads(resources.files("poultry_disease.data").joinpath("input_check.json").read_text("utf-8"))
THRESHOLD: float = CONFIG["threshold"]
VERSION: str = CONFIG["version"]
PROTOTYPES: np.ndarray = np.asarray(CONFIG["prototypes"]["vectors"], dtype=np.float64)


@dataclass(frozen=True)
class InputCheck:
    score: float
    threshold: float
    passed: bool
    version: str


def input_check_score(embedding: np.ndarray) -> float:
    """Cosine similarity of one image embedding with the nearest class prototype."""
    e = np.asarray(embedding, dtype=np.float64).ravel()
    if e.shape != (PROTOTYPES.shape[1],):
        raise ValueError(f"expected a {PROTOTYPES.shape[1]}-d embedding, got shape {e.shape}")
    norm = float(np.linalg.norm(e))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("embedding has no direction (zero or non-finite norm)")
    return float((PROTOTYPES @ (e / norm)).max())


def check_embedding(embedding: np.ndarray) -> InputCheck:
    s = input_check_score(embedding)
    return InputCheck(score=s, threshold=THRESHOLD, passed=s >= THRESHOLD, version=VERSION)
