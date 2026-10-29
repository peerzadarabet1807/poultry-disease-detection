"""A NumPy port of TensorFlow's antialiased bilinear resize (ScaleAndTranslate, triangle kernel).

It documents the algorithm that web/src/lib/resize.ts implements, and tests prove it is bit-identical to
``tf.image.resize(..., method="bilinear", antialias=True)``. All arithmetic is float32, as in the C++ kernel:

* per output axis, a span of source pixels and normalised triangle weights (kernel radius scaled by
  ``max(1/scale, 1)`` when downsampling, sample position ``(x + 0.5) / scale``);
* rows are resampled first (into an intermediate of output_height x input_width), then columns;
* each output value accumulates ``weight * value`` one source pixel at a time, in float32.
"""

from __future__ import annotations

import numpy as np

f32 = np.float32


def compute_spans(out_size: int, in_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    scale = f32(f32(out_size) / f32(in_size))
    inv_scale = f32(f32(1.0) / scale)
    kernel_scale = max(inv_scale, f32(1.0))
    radius = f32(kernel_scale)  # triangle radius 1, scaled
    span_size = min(2 * int(np.ceil(radius)) + 1, in_size)
    starts = np.zeros(out_size, np.int64)
    weights = np.zeros((out_size, span_size), f32)
    one_over_kernel_scale = f32(f32(1.0) / kernel_scale)
    for x in range(out_size):
        sample = f32(f32(f32(x) + f32(0.5)) * inv_scale)
        if sample < 0 or sample > in_size:
            continue
        start = int(np.ceil(f32(sample - radius) - f32(0.5)))
        end = int(np.floor(f32(sample + radius) - f32(0.5)))
        start = min(max(start, 0), in_size - 1)
        end = min(max(end, 0), in_size - 1) + 1
        ws, total = [], f32(0.0)
        for src in range(start, end):
            pos = f32(f32(f32(src) + f32(0.5)) - sample)
            w = max(f32(0.0), f32(f32(1.0) - abs(f32(pos * one_over_kernel_scale))))
            ws.append(w)
            total = f32(total + w)
        if abs(total) >= 1000 * np.finfo(f32).eps:
            inv_total = f32(f32(1.0) / total)
            ws = [f32(w * inv_total) for w in ws]
        starts[x] = start
        weights[x, : len(ws)] = ws
    return starts, weights, span_size


def _gather(img: np.ndarray, starts, weights, span_size: int, axis: int) -> np.ndarray:
    src = np.moveaxis(img, axis, 0)
    out = np.zeros((len(starts), *src.shape[1:]), f32)
    for o, s in enumerate(starts):
        acc = np.zeros(src.shape[1:], f32)
        for i in range(min(s + span_size, src.shape[0]) - s):
            acc = (acc + (weights[o, i] * src[s + i]).astype(f32)).astype(f32)
        out[o] = acc
    return np.moveaxis(out, 0, axis)


def resize_float(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    x = np.asarray(img, f32)
    rows = compute_spans(out_h, x.shape[0])
    cols = compute_spans(out_w, x.shape[1])
    return _gather(_gather(x, *rows, axis=0), *cols, axis=1)


def resize_uint8(img: np.ndarray, out_h: int = 224, out_w: int = 224) -> np.ndarray:
    """resize -> round half to even (np.round == tf.round) -> clip -> uint8."""
    return np.clip(np.round(resize_float(img, out_h, out_w)), 0, 255).astype(np.uint8)
