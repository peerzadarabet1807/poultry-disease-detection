"""The preprocessing contract, identical to the training notebook (Section 10 and Section 21).

1. Decode JPEG/PNG to 3-channel RGB with ``tf.io.decode_image(channels=3, expand_animations=False)``.
   EXIF orientation is **not** applied: training saw the stored pixels, so inference does too.
2. ``tf.image.resize(float32, (224, 224), method="bilinear", antialias=True)``. Aspect ratio is not kept.
3. Round (half to even, as ``tf.round``), clip to [0, 255], cast to uint8.
4. Cast to float32 and apply ``keras.applications.resnet50.preprocess_input`` (RGB -> BGR, subtract the
   ImageNet channel means, no scaling).

The model graph starts after step 4. The browser (web/src/lib/preprocess.ts) and the API use the same steps.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from poultry_disease.constants import CAFFE_MEANS_BGR, IMAGE_SIZE  # noqa: E402


def decode_image(data: bytes) -> np.ndarray:
    """Encoded JPEG/PNG bytes -> (H, W, 3) uint8 RGB, exactly as in training (no EXIF rotation)."""
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    return img.numpy()


def resize_uint8(img: np.ndarray | tf.Tensor) -> np.ndarray:
    """(H, W, 3) image -> (224, 224, 3) uint8 with TF's antialiased bilinear resize, rounded and clipped."""
    img = tf.ensure_shape(tf.convert_to_tensor(img), [None, None, 3])
    img = tf.image.resize(tf.cast(img, tf.float32), IMAGE_SIZE, method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8).numpy()


def to_model_input(img_uint8: np.ndarray) -> np.ndarray:
    """uint8 (224, 224, 3) or (N, 224, 224, 3) -> float32 ResNet50 (caffe) input with a batch axis."""
    x = np.asarray(img_uint8, dtype=np.float32)
    if x.ndim == 3:
        x = x[None]
    x = x[..., ::-1] - np.asarray(CAFFE_MEANS_BGR, dtype=np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def preprocess_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Encoded bytes -> (the 224 x 224 uint8 image the model sees, the float32 model input)."""
    seen = resize_uint8(decode_image(data))
    return seen, to_model_input(seen)


def preprocess_pil(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """PIL image -> (seen, model input), as the notebook's ``predict_pil_image`` (``image.convert("RGB")``).

    PIL's JPEG decoder is not bit-identical to TensorFlow's, so prefer :func:`preprocess_bytes` for files.
    """
    seen = resize_uint8(np.asarray(image.convert("RGB")))
    return seen, to_model_input(seen)
