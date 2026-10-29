"""Class activation maps on the internal test set and on the PCR-confirmed set, side by side.

The notebook's Grad-CAM figures (results/gradcam/) cover the internal test set only. This figure adds the
PCR-confirmed set, where the model is much weaker. Images are chosen deterministically from the saved
predictions: per class the most confident correct prediction on each set, then the four most confident PCR-set
errors of four different kinds (true -> predicted). Maps come from the package's in-graph CAM, which equals the
notebook's Grad-CAM (models/PARITY.md), overlaid as in the notebook (jet, alpha 0.4).

Needs the images: python scripts/fetch_zenodo_images.py --set internal --set external
Output: results/posthoc/cam_internal_vs_pcr.png

Usage:
    python scripts/posthoc_cam_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from posthoc_common import CLASS_NAMES, RESULTS, save_figure, setup_style

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "zenodo"


def most_confident(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return df.sort_values(["confidence", "relpath"], ascending=[False, True]).head(n)


def pick() -> list[tuple[str, pd.Series]]:
    internal = pd.read_csv(RESULTS / "test_predictions.csv")
    external = pd.read_csv(RESULTS / "pcr_predictions.csv")
    rows = []
    for name, df in (("Internal test", internal), ("PCR-confirmed", external)):
        for c in CLASS_NAMES:
            ok = df[(df["true_class"] == c) & df["correct"]]
            rows.append((name, most_confident(ok).iloc[0]))
    errors = most_confident(external[~external["correct"]], len(external))
    seen = set()
    for _, r in errors.iterrows():
        kind = (r["true_class"], r["predicted_class"])
        if kind not in seen:
            seen.add(kind)
            rows.append(("PCR-confirmed, error", r))
        if len(seen) == 4:
            break
    return rows


def main() -> int:
    import matplotlib.pyplot as plt

    from poultry_disease import Predictor
    from poultry_disease.cam import overlay
    from poultry_disease.weights import get_weights

    setup_style()
    predictor = Predictor(get_weights(allow_download=False))
    picks = pick()
    fig, axes = plt.subplots(3, 4, figsize=(13, 11.6))
    for ax, (label, r) in zip(axes.ravel(), picks, strict=True):
        pred = predictor.predict_image(IMAGES / r["relpath"])
        assert pred.predicted_class == r["predicted_class"], r["relpath"]
        ax.imshow(overlay(pred.seen, pred.cam_for()))
        ax.set_axis_off()
        wrong = r["true_class"] != r["predicted_class"]
        ax.set_title(
            f"{label}\nTrue: {r['true_class']} | Pred: {r['predicted_class']} ({pred.confidence:.2f})",
            fontsize=10,
            color="#b2182b" if wrong else "#222222",
        )
    fig.suptitle(
        "Class activation maps for the predicted class (in-graph CAM = notebook Grad-CAM), post-hoc",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(h_pad=2.2)
    path = save_figure(fig, "cam_internal_vs_pcr.png")
    print(path)
    for label, r in picks:
        print(f"{label:<22} {r['relpath']:<45} {r['true_class']:>12} -> {r['predicted_class']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
