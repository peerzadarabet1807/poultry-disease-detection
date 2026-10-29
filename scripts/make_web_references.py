"""Reference data for the TypeScript preprocessing and TF.js parity tests (web/tests/fixtures/).

golden/      TensorFlow's resize output for deterministic synthetic images. The pixel formula is simple
             enough to regenerate in TypeScript (web/tests/unit/resize.test.ts), so only outputs are stored.
reference/   For every committed fixture photo, two decodes:
             * "accurate": libjpeg's accurate integer IDCT (tf.io.decode_jpeg(dct_method="INTEGER_ACCURATE")),
               which is what browsers, sharp and Pillow use. SHA-256 of its RGB pixels, the 224 x 224 image
               the model sees and the Keras probabilities and ReLU + max-normalised CAMs for it. The Node test
               must reproduce these pixels exactly; the model comparison then isolates the TF.js conversion.
             * "training": tf.io.decode_image's default (the fast integer IDCT), the training pipeline. Keras
               probabilities on this machine and the Colab ones, for the end-to-end comparison.

Usage:
    python scripts/make_web_references.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "web" / "tests" / "fixtures"
#: (height, width): identity, upsampling, odd downsampling, portrait and landscape phone-like sizes.
GOLDEN_SHAPES = [(224, 224), (100, 130), (517, 333), (1000, 750), (1152, 2048)]


def synthetic(height: int, width: int) -> np.ndarray:
    """Same formula as web/tests/unit/resize.test.ts: busy, deterministic, uses the full 0-255 range."""
    y, x, c = np.meshgrid(np.arange(height), np.arange(width), np.arange(3), indexing="ij")
    return ((x * 7 + y * 13 + c * 101 + (x * y) % 251) % 256).astype(np.uint8)


def main() -> int:
    import tensorflow as tf

    from poultry_disease import CLASS_NAMES, Predictor
    from poultry_disease.cam import normalize_cam
    from poultry_disease.preprocess import decode_image, resize_uint8
    from poultry_disease.weights import get_weights

    golden = OUT / "golden"
    golden.mkdir(parents=True, exist_ok=True)
    rows = []
    for h, w in GOLDEN_SHAPES:
        name = f"{h}x{w}.npy"
        np.save(golden / name, resize_uint8(synthetic(h, w)))
        rows.append({"height": h, "width": w, "file": name})
    golden_doc = {
        "description": "tf.image.resize(bilinear, antialias=True) -> round -> clip -> uint8 of synthetic images",
        "formula": "(x * 7 + y * 13 + c * 101 + (x * y) % 251) % 256",
        "images": rows,
    }
    (golden / "golden.json").write_text(json.dumps(golden_doc, indent=2) + "\n", encoding="utf-8", newline="\n")

    predictor = Predictor(get_weights(allow_download=False))
    ref_dir = OUT / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for im in json.loads((FIXTURES / "fixtures.json").read_text("utf-8"))["images"]:
        data = (FIXTURES / "images" / im["file"]).read_bytes()
        rgb = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_ACCURATE").numpy()
        seen = resize_uint8(rgb)
        pred = predictor.predict_seen(seen)
        training = predictor.predict_seen(resize_uint8(decode_image(data)))
        np.save(ref_dir / f"{im['id']}.seen.npy", seen)
        refs.append(
            {
                "id": im["id"],
                "file": im["file"],
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "decodedSha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                "seen": f"{im['id']}.seen.npy",
                "kerasPredicted": pred.predicted_class,
                "kerasProbabilities": [float(p) for p in pred.probs],
                "trainingPathProbabilities": [float(p) for p in training.probs],
                "colabProbabilities": [im["colab_probabilities"][c] for c in CLASS_NAMES],
                "cams": {
                    c: np.round(normalize_cam(pred.cam[..., i]).ravel(), 7).tolist() for i, c in enumerate(CLASS_NAMES)
                },
            }
        )
        print(f"{im['id']} {rgb.shape[1]}x{rgb.shape[0]} {pred.predicted_class} {pred.confidence:.4f}")
    doc = {
        "classNames": list(CLASS_NAMES),
        "decode": "accurate integer IDCT (browsers, sharp, Pillow); trainingPath = TF default fast IDCT",
        "images": refs,
    }
    (ref_dir / "reference.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
