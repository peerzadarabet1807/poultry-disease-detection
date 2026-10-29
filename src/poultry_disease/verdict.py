"""Turn probabilities into a cautious verdict category, shared by the API and (in TypeScript) the web app.

Categories:
* ``uncertain``: the top-2 margin or the top probability is below the thresholds in
  ``data/decision_thresholds.json`` (a usability heuristic chosen by scripts/choose_thresholds.py).
* ``disease_pattern``: the most likely class is Coccidiosis, Salmonella or NCD ("pattern consistent with").
* ``no_disease_pattern``: the most likely class is Healthy. Not a clean bill of health.

``notifiable`` is set when NCD (Newcastle disease) is the most likely class, whatever the category.
None of this is a veterinary diagnosis.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Literal

from poultry_disease.constants import CLASS_NAMES

Category = Literal["disease_pattern", "no_disease_pattern", "uncertain"]
HEALTHY = "Healthy"
NOTIFIABLE = "NCD"

THRESHOLDS: dict = json.loads(
    resources.files("poultry_disease.data").joinpath("decision_thresholds.json").read_text("utf-8")
)
MIN_TOP_PROBABILITY: float = THRESHOLDS["min_top_probability"]
MIN_TOP2_MARGIN: float = THRESHOLDS["min_top2_margin"]


@dataclass(frozen=True)
class Verdict:
    category: Category
    top_class: str
    runner_up: str
    top_probability: float
    margin: float
    low_confidence: bool
    reasons: tuple[str, ...]
    notifiable: bool


def interpret(probabilities: Sequence[float], class_names: Sequence[str] = CLASS_NAMES) -> Verdict:
    if len(probabilities) != len(class_names):
        raise ValueError(f"expected {len(class_names)} probabilities, got {len(probabilities)}")
    # stable: ties keep model order, as in the web app
    order = sorted(range(len(class_names)), key=lambda i: (-probabilities[i], i))
    top, second = order[0], order[1]
    p_top = float(probabilities[top])
    margin = p_top - float(probabilities[second])
    reasons = []
    if p_top < MIN_TOP_PROBABILITY:
        reasons.append("low_top_probability")
    if margin < MIN_TOP2_MARGIN:
        reasons.append("small_margin")
    top_class = class_names[top]
    if reasons:
        category: Category = "uncertain"
    elif top_class == HEALTHY:
        category = "no_disease_pattern"
    else:
        category = "disease_pattern"
    return Verdict(
        category=category,
        top_class=top_class,
        runner_up=class_names[second],
        top_probability=p_top,
        margin=margin,
        low_confidence=bool(reasons),
        reasons=tuple(reasons),
        notifiable=top_class == NOTIFIABLE,
    )
