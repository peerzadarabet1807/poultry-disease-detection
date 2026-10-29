"""The ``GET /model`` payload, read from the committed artifacts (nothing here is typed in by hand)."""

from __future__ import annotations

import json
from pathlib import Path

from poultry_api.imaging import accepted_formats
from poultry_disease import __version__
from poultry_disease.constants import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    IMAGE_SIZE,
    RELEASE_BASE_URL,
    RELEASE_TAG,
    WEIGHTS_FILE,
    WEIGHTS_SHA256,
)
from poultry_disease.verdict import THRESHOLDS

SOURCES = {
    "config": "models/model_config.json",
    "checksums": "models/checksums.sha256",
    "experiment": "results/experiment_config.json",
    "bootstrap": "results/posthoc/bootstrap_ci.json",
    "calibration": "results/posthoc/calibration.json",
}


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_text("utf-8")


def load_model_info(root: Path) -> dict:
    config = json.loads(_read(root, SOURCES["config"]))
    experiment = json.loads(_read(root, SOURCES["experiment"]))
    bootstrap = json.loads(_read(root, SOURCES["bootstrap"]))["sets"]
    calibration = json.loads(_read(root, SOURCES["calibration"]))["sets"]
    checksums = dict(reversed(line.split()) for line in _read(root, SOURCES["checksums"]).splitlines() if line)
    if checksums.get(WEIGHTS_FILE) != WEIGHTS_SHA256:
        raise RuntimeError(f"{SOURCES['checksums']} does not list the packaged weights checksum")
    if tuple(config["class_names"]) != CLASS_NAMES:
        raise RuntimeError("models/model_config.json class order differs from the package")

    def evaluation(key: str) -> dict:
        b, c = bootstrap[key], calibration[key]
        return {
            "label": b["label"],
            "n": b["n"],
            "accuracy": b["metrics"]["accuracy"]["value"],
            "accuracy_ci95": b["metrics"]["accuracy"]["ci95"],
            "macro_f1": b["metrics"]["macro_f1"]["value"],
            "macro_f1_ci95": b["metrics"]["macro_f1"]["ci95"],
            "expected_calibration_error": c["ece"],
            "per_class_precision": {k: v["precision"]["value"] for k, v in b["per_class"].items()},
            "per_class_recall": {k: v["recall"]["value"] for k, v in b["per_class"].items()},
        }

    return {
        "name": "poultry_resnet50",
        "version": __version__,
        "release": RELEASE_TAG,
        "run_id": experiment["run_id"],
        "task": config["task"],
        "intended_use": (
            "Research and decision-support prototype. Not a veterinary diagnosis; it does not replace a "
            "veterinarian or a laboratory test."
        ),
        "classes": [{"index": i, "name": c, "display_name": CLASS_DISPLAY_NAMES[c]} for i, c in enumerate(CLASS_NAMES)],
        "architecture": {"backbone": config["architecture"]["backbone"], "head": config["architecture"]["head"]},
        "input": {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1], "channels": 3, "formats": accepted_formats()},
        "preprocessing": {
            "steps": config["preprocessing"]["steps"],
            "exif_orientation": "ignored, as in training",
        },
        "weights": {"file": WEIGHTS_FILE, "sha256": WEIGHTS_SHA256, "url": f"{RELEASE_BASE_URL}/{WEIGHTS_FILE}"},
        "decision_thresholds": {k: THRESHOLDS[k] for k in ("min_top_probability", "min_top2_margin", "status")},
        "evaluation": {"internal": evaluation("internal"), "external": evaluation("external")},
        "sources": sorted(SOURCES.values()),
    }
