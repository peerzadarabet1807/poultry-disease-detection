"""Export the browser model: one graph that returns class probabilities *and* class activation maps.

TF.js graph models cannot compute gradients, so the notebook's Grad-CAM is folded into the forward pass
(see src/poultry_disease/model.py: for a GAP -> Dense head, Grad-CAM equals ReLU(sum_k w_k A^k) / (H W),
a 1 x 1 convolution of the feature map with the Dense kernel).

Signature ``serving_default`` (float32):
    input   image  [N, 224, 224, 3]   ResNet50 "caffe" input (BGR, ImageNet means subtracted), as in training
    output  probs  [N, 4]             softmax, the trained model's output
    output  cam    [N, 7, 7, 4]       pre-ReLU class maps (the client applies ReLU and divides by the max)

Checks before export (on the committed fixtures): probabilities identical to the trained model, CAM equal
to the notebook's gradient Grad-CAM (Pearson r), and the exported SavedModel equal to the Keras graph.

Usage:
    python scripts/build_web_model.py [--out build/web_saved_model]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PROB_TOL = 1e-6
CAM_MIN_R = 0.99999


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "build" / "web_saved_model"))
    args = ap.parse_args()

    import tensorflow as tf

    from poultry_disease.cam import gradient_grad_cam, normalize_cam, pearson
    from poultry_disease.model import CAM_OUTPUT, INPUT_NAME, PROBS_OUTPUT, build_inference_model, load_keras_model
    from poultry_disease.preprocess import preprocess_bytes
    from poultry_disease.weights import get_weights

    trained = load_keras_model(get_weights(allow_download=False))
    web = build_inference_model(trained)

    images = json.loads((FIXTURES / "fixtures.json").read_text("utf-8"))["images"]
    x = np.concatenate([preprocess_bytes((FIXTURES / "images" / im["file"]).read_bytes())[1] for im in images])
    p_ref = trained.predict(x, batch_size=6, verbose=0)
    out = web.predict(x, batch_size=6, verbose=0)
    p_web, cam_web = out[PROBS_OUTPUT], out[CAM_OUTPUT]
    diff = float(np.abs(p_ref - p_web).max())
    print(f"[check] probs vs trained model: max |diff| {diff:.2e} (tolerance {PROB_TOL:.0e})")
    assert diff < PROB_TOL

    rs = []
    for i in range(len(x)):
        c = int(p_web[i].argmax())
        grad = gradient_grad_cam(trained, x[i : i + 1], c)
        rs.append(pearson(normalize_cam(cam_web[i, ..., c]), normalize_cam(grad)))
    print(f"[check] CAM vs gradient Grad-CAM: Pearson r min {min(rs):.12f}")
    assert min(rs) > CAM_MIN_R

    web.export(args.out, verbose=False)
    sig = tf.saved_model.load(args.out).signatures["serving_default"]
    print("[export] inputs :", {k: (v.shape, v.dtype.name) for k, v in sig.structured_input_signature[1].items()})
    print("[export] outputs:", {k: (v.shape, v.dtype.name) for k, v in sig.structured_outputs.items()})
    served = sig(**{INPUT_NAME: tf.constant(x[:6])})
    served_diff = float(np.abs(served[PROBS_OUTPUT].numpy() - p_web[:6]).max())
    cam_diff = float(np.abs(served[CAM_OUTPUT].numpy() - cam_web[:6]).max())
    print(f"[check] SavedModel vs Keras: max |prob diff| {served_diff:.2e}, max |cam diff| {cam_diff:.2e}")
    assert served_diff < PROB_TOL
    print(f"Exported -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
