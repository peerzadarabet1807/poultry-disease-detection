"""Class activation maps: the notebook's gradient-based Grad-CAM, and post-processing shared with the web app.

``Predictor`` gets CAMs from the in-graph 1 x 1 convolution (see :mod:`poultry_disease.model`).
:func:`gradient_grad_cam` is the notebook's implementation, kept to prove the two agree.
"""

from __future__ import annotations

import io

import numpy as np
import tensorflow as tf
from PIL import Image

from poultry_disease.constants import BACKBONE_LAYER, CLASSIFIER_LAYER, GAP_LAYER


def gradient_grad_cam(trained, x: np.ndarray, class_index: int) -> np.ndarray:
    """Notebook Section 18: Grad-CAM of the class logit w.r.t. conv5_block3_out, pre-ReLU, 7 x 7.

    ``trained`` is the model loaded from the ``.keras`` file, ``x`` one preprocessed input (1, 224, 224, 3).
    """
    backbone = trained.get_layer(BACKBONE_LAYER)
    gap = trained.get_layer(GAP_LAYER)
    clf = trained.get_layer(CLASSIFIER_LAYER)
    with tf.GradientTape() as tape:
        conv = backbone(tf.convert_to_tensor(x), training=False)
        tape.watch(conv)
        logits = tf.matmul(gap(conv), clf.kernel) + clf.bias
        score = logits[:, class_index]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    return tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)[0].numpy()


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """ReLU, then divide by the maximum (the notebook's normalisation). All-zero if nothing is positive."""
    cam = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    peak = float(cam.max())
    return cam / peak if peak > 0 else np.zeros_like(cam)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def jet(v: np.ndarray) -> np.ndarray:
    """The classic "jet" colormap, [0, 1] -> uint8 RGB. Same formula as web/src/lib/cam.ts."""
    t = np.clip(v, 0.0, 1.0)[..., None]
    rgb = np.clip(1.5 - np.abs(4.0 * t - np.array([3.0, 2.0, 1.0])), 0.0, 1.0)
    return np.round(255.0 * rgb).astype(np.uint8)


def upsample(cam01: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear upsampling with half-pixel centres (as the notebook's ``tf.image.resize`` of the CAM)."""
    out = tf.image.resize(np.asarray(cam01, np.float32)[..., None], size, method="bilinear")
    return out.numpy()[..., 0]


def overlay(seen_uint8: np.ndarray, cam01: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend the jet-coloured, upsampled map over the image the model saw (notebook: alpha 0.4)."""
    heat = jet(upsample(cam01, seen_uint8.shape[:2])).astype(np.float32)
    blended = (1.0 - alpha) * seen_uint8.astype(np.float32) + alpha * heat
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def to_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
