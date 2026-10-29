"""Parity check: does the Python package reproduce the probabilities the notebook recorded in Colab?

Runs the package (local CPU, TensorFlow 2.21) on original dataset images and compares with the prob_*
columns the notebook saved on a Colab T4 GPU. Also compares the in-graph CAM with the notebook's
gradient-based Grad-CAM for the predicted class (Pearson r of the normalised 7 x 7 maps).

Usage:
    python scripts/parity_python.py --fixtures                # the 18 committed fixtures
    python scripts/parity_python.py --set internal --set external   # all 2,175 evaluation images
                                                              # (fetch them first with fetch_zenodo_images.py)

With --decoder-effect, every JPEG is also decoded with libjpeg's accurate integer IDCT, the decoder that
browsers (and Pillow, sharp) use, instead of TensorFlow's default fast IDCT that training used. The browser
cannot reproduce the training decode, so this measures how much that changes the predictions.

Writes results/parity/python_vs_colab_<scope>.json (summary) and, for full sets, a per-image CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "parity"
SETS = {"internal": "test_predictions.csv", "external": "pcr_predictions.csv"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", action="store_true", help="the committed tests/fixtures images")
    ap.add_argument("--set", action="append", choices=sorted(SETS), default=[])
    ap.add_argument("--images", type=Path, default=ROOT / "data" / "zenodo", help="root of fetched images")
    ap.add_argument("--no-gradcam", action="store_true", help="skip the (slower) gradient Grad-CAM comparison")
    ap.add_argument("--decoder-effect", action="store_true", help="also predict from the accurate-IDCT decode")
    args = ap.parse_args()

    import tensorflow as tf

    from poultry_disease import CLASS_NAMES, Predictor
    from poultry_disease.cam import gradient_grad_cam, normalize_cam, pearson
    from poultry_disease.preprocess import resize_uint8, to_model_input
    from poultry_disease.weights import get_weights

    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    if args.fixtures:
        scope = "fixtures"
        fx = json.loads((ROOT / "tests" / "fixtures" / "fixtures.json").read_text("utf-8"))["images"]
        items = [
            (
                im["relpath"],
                im["set"],
                ROOT / "tests" / "fixtures" / "images" / im["file"],
                np.array([im["colab_probabilities"][c] for c in CLASS_NAMES]),
                im["true_class"],
            )
            for im in fx
        ]
    elif args.set:
        scope = "_".join(sorted(args.set))
        items = []
        for s in sorted(args.set):
            df = pd.read_csv(RESULTS / SETS[s])
            items += [
                (r.relpath, s, args.images / r.relpath, np.array([getattr(r, c) for c in prob_cols]), r.true_class)
                for r in df.itertuples()
            ]
    else:
        ap.error("pass --fixtures or --set")

    predictor = Predictor(get_weights(allow_download=False))
    rows, start = [], time.time()
    for i, (relpath, set_name, path, colab, true_class) in enumerate(items):
        data = path.read_bytes()
        pred = predictor.predict_bytes(data)
        row = {
            "relpath": relpath,
            "set": set_name,
            "true_class": true_class,
            "colab_predicted": CLASS_NAMES[int(np.argmax(colab))],
            "python_predicted": pred.predicted_class,
            "max_abs_diff": float(np.abs(pred.probs.astype(np.float64) - colab).max()),
            **{f"python_{c}": float(p) for c, p in zip(prob_cols, pred.probs, strict=True)},
        }
        if args.decoder_effect and data[:3] == b"\xff\xd8\xff":
            rgb = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_ACCURATE").numpy()
            accurate = predictor.predict_seen(resize_uint8(rgb))
            row["accurate_idct_predicted"] = accurate.predicted_class
            row["accurate_idct_max_abs_diff"] = float(np.abs(accurate.probs.astype(np.float64) - colab).max())
        if not args.no_gradcam:
            grad = gradient_grad_cam(predictor.trained, to_model_input(pred.seen), pred.top_index)
            row["cam_vs_gradcam_r"] = float(pearson(normalize_cam(grad), pred.cam_for()))
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)} ({time.time() - start:.0f} s)", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    df["argmax_match"] = df["colab_predicted"] == df["python_predicted"]
    summary = {
        "description": (
            "Python package (CPU) vs the probabilities recorded by the training notebook (Colab T4 GPU), on "
            "original dataset images. Engineering check, not a model evaluation."
        ),
        "run": "run_20261029_191001",
        "environment": {"tensorflow": tf.__version__, "device": "CPU", "platform": sys.platform},
        "n_images": int(len(df)),
        "argmax_identical": int(df["argmax_match"].sum()),
        "max_abs_prob_diff": float(df["max_abs_diff"].max()),
        "median_max_abs_prob_diff": float(df["max_abs_diff"].median()),
        "by_set": {
            s: {
                "n": int(len(g)),
                "argmax_identical": int(g["argmax_match"].sum()),
                "max_abs_prob_diff": float(g["max_abs_diff"].max()),
            }
            for s, g in df.groupby("set")
        },
    }
    if "cam_vs_gradcam_r" in df:
        summary["cam_vs_gradcam_min_r"] = float(df["cam_vs_gradcam_r"].min())
    if "accurate_idct_predicted" in df:
        from sklearn.metrics import accuracy_score, f1_score

        summary["decoder_effect"] = {
            "description": (
                "Predictions from the accurate-IDCT JPEG decode (browsers) vs the training decode (fast IDCT, "
                "Colab). Accuracy and macro-F1 of both on the same images."
            ),
            "by_set": {
                s: {
                    "n": int(len(g)),
                    "argmax_identical_to_colab": int((g["accurate_idct_predicted"] == g["colab_predicted"]).sum()),
                    "max_abs_prob_diff": float(g["accurate_idct_max_abs_diff"].max()),
                    "median_max_abs_prob_diff": float(g["accurate_idct_max_abs_diff"].median()),
                    "accuracy_training_decode": float(accuracy_score(g["true_class"], g["colab_predicted"])),
                    "accuracy_accurate_idct": float(accuracy_score(g["true_class"], g["accurate_idct_predicted"])),
                    "macro_f1_training_decode": float(f1_score(g["true_class"], g["colab_predicted"], average="macro")),
                    "macro_f1_accurate_idct": float(
                        f1_score(g["true_class"], g["accurate_idct_predicted"], average="macro")
                    ),
                }
                for s, g in df.groupby("set")
            },
        }
    mismatches = df.loc[~df["argmax_match"], ["relpath", "colab_predicted", "python_predicted", "max_abs_diff"]]
    summary["argmax_mismatches"] = mismatches.to_dict("records")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"python_vs_colab_{scope}.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8", newline="\n")
    if scope != "fixtures":
        df.to_csv(OUT / f"python_vs_colab_{scope}.csv", index=False, float_format="%.8g", lineterminator="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "description"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
