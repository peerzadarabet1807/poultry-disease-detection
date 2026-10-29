import io
import json

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from poultry_disease import preprocess
from poultry_disease.constants import CAFFE_MEANS_BGR, CLASS_NAMES, IMAGE_SIZE
from tests.conftest import ROOT
from tests.resize_reference import resize_float
from tests.resize_reference import resize_uint8 as reference_resize

SHAPES = [(1000, 750), (1152, 2048), (3200, 1440), (224, 224), (100, 130), (517, 333)]


def test_class_names_match_training_run():
    assert list(CLASS_NAMES) == json.loads((ROOT / "models" / "class_names.json").read_text("utf-8"))
    cfg = json.loads((ROOT / "models" / "model_config.json").read_text("utf-8"))
    assert list(CLASS_NAMES) == cfg["class_names"]
    assert list(IMAGE_SIZE) == cfg["image_size"]
    assert cfg["preprocessing"]["function"] == "tf.keras.applications.resnet50.preprocess_input"
    assert cfg["preprocessing"]["applied_inside_model"] is False


def notebook_resize(img):
    """Notebook Section 10, verbatim: resize_uint8."""
    img = tf.ensure_shape(img, [None, None, 3])
    img = tf.image.resize(tf.cast(img, tf.float32), IMAGE_SIZE, method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)


@pytest.mark.parametrize("shape", SHAPES)
def test_resize_matches_notebook(shape):
    img = np.random.default_rng(sum(shape)).integers(0, 256, (*shape, 3), dtype=np.uint8)
    np.testing.assert_array_equal(preprocess.resize_uint8(img), notebook_resize(tf.constant(img)).numpy())


@pytest.mark.parametrize("shape", SHAPES)
def test_numpy_reference_port_is_bit_exact(shape):
    """The algorithm ported to TypeScript reproduces TF's float32 output exactly, not just after rounding."""
    img = np.random.default_rng(7 + sum(shape)).integers(0, 256, (*shape, 3), dtype=np.uint8)
    ref = tf.image.resize(tf.cast(img, tf.float32), IMAGE_SIZE, method="bilinear", antialias=True).numpy()
    np.testing.assert_array_equal(resize_float(img, *IMAGE_SIZE), ref)
    np.testing.assert_array_equal(reference_resize(img), preprocess.resize_uint8(img))


def test_round_half_to_even():
    # Averaging 100 and 101 (2 x 1 -> 1 x 1) gives exactly 100.5; tf.round rounds half to even: 100, not 101.
    img = np.array([[[100, 100, 100], [101, 101, 101]]], dtype=np.uint8)
    out = tf.image.resize(tf.cast(img, tf.float32), (1, 1), method="bilinear", antialias=True).numpy()
    assert out[0, 0, 0] == 100.5
    assert int(tf.round(out)[0, 0, 0]) == 100


def test_caffe_preprocessing_matches_keras():
    seen = np.random.default_rng(0).integers(0, 256, (2, 224, 224, 3), dtype=np.uint8)
    ours = preprocess.to_model_input(seen)
    keras_np = keras.applications.resnet50.preprocess_input(seen.astype(np.float32))
    keras_tf = keras.applications.resnet50.preprocess_input(tf.cast(seen, tf.float32)).numpy()
    np.testing.assert_array_equal(ours, keras_np)
    np.testing.assert_array_equal(ours, keras_tf)
    assert ours.dtype == np.float32 and ours.shape == (2, 224, 224, 3)
    expected = seen[0, 0, 0, ::-1] - np.array(CAFFE_MEANS_BGR)
    np.testing.assert_allclose(ours[0, 0, 0], expected, rtol=0, atol=1e-4)


def jpeg_with_orientation(orientation: int, size=(64, 32)) -> bytes:
    img = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_exif_orientation_is_ignored_like_training():
    data = jpeg_with_orientation(6)  # "rotate 90 degrees clockwise to display"
    assert preprocess.decode_image(data).shape == (32, 64, 3)  # the stored pixels, not rotated


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_png_with_alpha_and_grayscale_become_rgb(mode):
    buf = io.BytesIO()
    Image.new(mode, (40, 30)).save(buf, format="PNG")
    seen, x = preprocess.preprocess_bytes(buf.getvalue())
    assert seen.shape == (224, 224, 3) and seen.dtype == np.uint8
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
