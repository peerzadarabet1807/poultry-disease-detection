"""Shared helpers for the post-hoc analyses of saved predictions (no images or weights needed).

Every script reads results/test_predictions.csv and results/pcr_predictions.csv (written by the notebook)
and writes to results/posthoc/. Plot style matches the notebook (seaborn whitegrid, Set2 class palette,
blue/orange source palette, savefig dpi 150, tight bbox).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POSTHOC = RESULTS / "posthoc"
CLASS_NAMES: list[str] = json.loads((ROOT / "models" / "class_names.json").read_text("utf-8"))
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42

SETS = {
    "internal": ("Internal test (version 2 held-out)", RESULTS / "test_predictions.csv"),
    "external": ("External PCR-confirmed (version 3)", RESULTS / "pcr_predictions.csv"),
}
# Notebook SOURCE_PALETTE: version-2 blue, version-3-pcr orange.
SET_COLORS = {"internal": "#4C72B0", "external": "#DD8452"}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def class_palette() -> dict[str, tuple]:
    return dict(zip(CLASS_NAMES, sns.color_palette("Set2", NUM_CLASSES), strict=False))


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (y_true class indices, probabilities [n, 4] in model order)."""
    df = pd.read_csv(path)
    y = df["true_class"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()
    probs = df[[f"prob_{c}" for c in CLASS_NAMES]].to_numpy(dtype=np.float64)
    if np.isnan(y.astype(float)).any():
        raise ValueError(f"{path}: unknown class label")
    return y.astype(int), probs


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


def save_figure(fig, name: str) -> Path:
    POSTHOC.mkdir(parents=True, exist_ok=True)
    path = POSTHOC / name
    fig.savefig(path)
    plt.close(fig)
    return path
