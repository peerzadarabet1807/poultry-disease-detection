"""Step 1 of the input-check study: the images the model sees, its pooled features and its probabilities.

For every image of the study sets this script runs the package's preprocessing (decode without EXIF rotation,
antialiased bilinear resize of the full frame to 224 x 224, uint8) and the published model, and stores

* ``seen``: the 224 x 224 uint8 RGB image. The input check (zero-shot gate) reads exactly this image, so it
  needs no preprocessing of its own beyond its own normalisation, in Python and in the browser;
* ``feat``: the 2048-d global-average-pooled ResNet50 features (for the feature-distance baseline);
* ``probs``: the softmax output (to see what the gate does on the model's own errors).

Sets (in-domain images must be in data/zenodo/, see scripts/fetch_zenodo_images.py):

* ``val``: the 973 validation images (tuning the threshold)
* ``train_sample``: 150 training images per class, seeded (feature bank and image prototypes)
* ``test_internal`` / ``test_external``: the 974 internal and 1,201 PCR test images (reporting only)
* ``samples``: the photos shipped with the web app
* ``ood``: every photo listed in docs/ood/credits.json

Output: .work/gate/<set>.npz and .work/gate/<set>.json (ids and labels). Run with the TensorFlow venv:

    .work/venv-tf/Scripts/python scripts/gate_extract.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "zenodo"
OUT = ROOT / ".work" / "gate"
SETS = ("val", "train_sample", "test_internal", "test_external", "samples", "ood")
TRAIN_PER_CLASS, TRAIN_SEED = 150, 0


def in_domain(set_name: str) -> list[dict]:
    if set_name in ("test_internal", "test_external"):
        csv = "test_predictions.csv" if set_name == "test_internal" else "pcr_predictions.csv"
        df = pd.read_csv(RESULTS / csv, usecols=["relpath", "true_class"]).rename(columns={"true_class": "class"})
    else:
        df = pd.read_csv(RESULTS / "dataset_manifest.csv", usecols=["relpath", "class", "split"])
        df = df[df["split"] == ("val" if set_name == "val" else "train")].sort_values("relpath")
        if set_name == "train_sample":  # the same sample as fetch_zenodo_images.py --split train --per-class 150
            df = pd.concat(
                [g.sample(n=min(TRAIN_PER_CLASS, len(g)), random_state=TRAIN_SEED) for _, g in df.groupby("class")]
            )
    return [{"id": r.relpath, "path": str(DATA / r.relpath), "label": r["class"]} for _, r in df.iterrows()]


def items(set_name: str) -> list[dict]:
    if set_name == "samples":
        manifest = json.loads((ROOT / "web" / "public" / "samples" / "manifest.json").read_text("utf-8"))
        entries = manifest["samples"] if isinstance(manifest, dict) else manifest
        out = []
        for e in entries:
            f = e.get("file") or e.get("src") or e.get("path")
            out.append(
                {
                    "id": Path(f).name,
                    "path": str(ROOT / "web" / "public" / "samples" / Path(f).name),
                    "label": e["trueClass"],
                }
            )
        return out
    if set_name == "ood":
        credits = json.loads((ROOT / "docs" / "ood" / "credits.json").read_text("utf-8"))
        return [
            {
                "id": p["file"],
                "path": str(ROOT / "docs" / "ood" / p["file"]),
                "label": p.get("category") or "original_" + p["group"].replace(" ", "_"),
            }
            for p in credits["photos"]
        ]
    return in_domain(set_name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", choices=SETS, help="default: all sets")
    args = ap.parse_args()

    import keras

    from poultry_disease.constants import GAP_LAYER
    from poultry_disease.model import load_keras_model
    from poultry_disease.preprocess import preprocess_bytes, to_model_input
    from poultry_disease.weights import get_weights

    trained = load_keras_model(get_weights(allow_download=False))
    model = keras.Model(trained.inputs, [trained.get_layer(GAP_LAYER).output, trained.outputs[0]])
    OUT.mkdir(parents=True, exist_ok=True)
    for set_name in args.set or SETS:
        rows = items(set_name)
        missing = [r["path"] for r in rows if not Path(r["path"]).exists()]
        if missing:
            raise FileNotFoundError(f"{set_name}: {len(missing)} images missing, e.g. {missing[0]}")
        seen = np.stack([preprocess_bytes(Path(r["path"]).read_bytes())[0] for r in rows])
        feats, probs = [], []
        for i in range(0, len(seen), 32):
            f, p = model(to_model_input(seen[i : i + 32]), training=False)
            feats.append(np.asarray(f, np.float32))
            probs.append(np.asarray(p, np.float32))
        np.savez(OUT / f"{set_name}.npz", seen=seen, feat=np.concatenate(feats), probs=np.concatenate(probs))
        meta = [{"id": r["id"], "label": r["label"]} for r in rows]
        (OUT / f"{set_name}.json").write_text(json.dumps(meta, indent=1), "utf-8")
        print(f"{set_name:<14} {len(rows)} images", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
