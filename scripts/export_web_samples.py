"""Export the sample photos shown in the web app, and their Keras references for the e2e tests.

Samples are a subset of the committed fixtures (CC BY 4.0), downscaled to 1024 px on the long side so the
page stays light. Downscaling changes what the model sees, so each sample's reference is recomputed on the
exported file, decoded the way browsers decode JPEGs (accurate integer IDCT, see models/PARITY.md).

Outputs:
    web/public/samples/<id>.jpg, web/public/samples/thumbs/<id>.jpg, web/public/samples/manifest.json
    web/tests/fixtures/sample-references.json

Usage:
    python scripts/export_web_samples.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "web" / "public" / "samples"
MAX_SIDE = 1024
THUMB_SIDE = 320

#: fixture id -> an optional remark shown with the dataset label after classifying (class and set come from data)
SAMPLES = {
    "f01": "",
    "f03": "",
    "f05": "",
    "f07": "",
    "f09": "",
    "f10": "",
    "f11": "",
    "f12": "",
    "f15": "Picked as the PCR-set Newcastle disease photo the model most confidently calls Healthy.",
    "f16": "Picked as the PCR-set Healthy photo the model most confidently calls Salmonella.",
    "f18": "Picked as the PCR-set photo on which the training run's prediction was most split.",
}


def save_jpeg(img: Image.Image, side: int, path: Path) -> None:
    scale = side / max(img.size)
    small = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS) if scale < 1 else img
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=90, optimize=True)  # no EXIF, no ICC profile
    path.write_bytes(buf.getvalue())


def main() -> int:
    import tensorflow as tf

    from poultry_disease import CLASS_NAMES, Predictor
    from poultry_disease.preprocess import resize_uint8
    from poultry_disease.verdict import interpret
    from poultry_disease.weights import get_weights

    fixtures = {im["id"]: im for im in json.loads((FIXTURES / "fixtures.json").read_text("utf-8"))["images"]}
    predictor = Predictor(get_weights(allow_download=False))
    (OUT / "thumbs").mkdir(parents=True, exist_ok=True)
    for old in [*OUT.glob("*.jpg"), *(OUT / "thumbs").glob("*.jpg")]:
        old.unlink()

    manifest, refs = [], []
    for sid, note in SAMPLES.items():
        im = fixtures[sid]
        with Image.open(FIXTURES / "images" / im["file"]) as src:
            img = src.convert("RGB")  # stored pixels: training did not apply EXIF rotation either
        file = f"{sid}.jpg"
        save_jpeg(img, MAX_SIDE, OUT / file)
        save_jpeg(img, THUMB_SIDE, OUT / "thumbs" / file)

        data = (OUT / file).read_bytes()
        rgb = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_ACCURATE").numpy()
        pred = predictor.predict_seen(resize_uint8(rgb))
        v = interpret(pred.probs.tolist())
        manifest.append(
            {
                "id": sid,
                "file": file,
                "thumb": f"thumbs/{file}",
                "trueClass": im["true_class"],
                "set": im["set"],
                "note": note,
                "source": im["relpath"],
                "doi": im["source_doi"],
            }
        )
        refs.append(
            {
                "id": sid,
                "file": file,
                "kerasPredicted": pred.predicted_class,
                "kerasProbabilities": [float(p) for p in pred.probs],
                "category": v.category,
            }
        )
        conf = float(np.max(pred.probs))
        print(f"{sid} {im['true_class']:<12} -> {pred.predicted_class:<12} {conf:.4f} {v.category:<20} {file}")

    credit = (
        "Machuve et al., Machine Learning Dataset for Poultry Diseases Diagnostics, Zenodo, CC BY 4.0. "
        "Downscaled to 1024 px; no other changes."
    )
    doc = {"credit": credit, "license": "CC-BY-4.0", "samples": manifest}
    (OUT / "manifest.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    ref_doc = {"classNames": list(CLASS_NAMES), "decode": "accurate integer IDCT", "samples": refs}
    out = ROOT / "web" / "tests" / "fixtures" / "sample-references.json"
    out.write_text(json.dumps(ref_doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
