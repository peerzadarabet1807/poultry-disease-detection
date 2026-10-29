"""The in-graph CAM equals the notebook's gradient Grad-CAM. Uses a randomly initialised model with the same
architecture and layer names, so it needs no weights (tests/test_parity.py repeats it with the trained model)."""

import keras
import numpy as np
import pytest

from poultry_disease.cam import gradient_grad_cam, jet, normalize_cam, overlay, pearson, to_png
from poultry_disease.model import build_inference_model


@pytest.fixture(scope="module")
def random_models():
    keras.utils.set_random_seed(0)
    base = keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
    backbone = keras.Model(base.input, base.output, name="resnet50_backbone")
    inputs = keras.Input((224, 224, 3), name="input_image")
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    out = keras.layers.Dense(4, activation="softmax", name="classifier")(x)
    trained = keras.Model(inputs, out)
    return trained, build_inference_model(trained)


def test_probs_identical_and_cam_matches_gradcam(random_models):
    trained, model = random_models
    x = np.random.default_rng(1).normal(0, 60, (2, 224, 224, 3)).astype(np.float32)
    out = model(x, training=False)
    np.testing.assert_allclose(np.asarray(out["probs"]), trained(x, training=False).numpy(), rtol=0, atol=1e-6)
    assert tuple(out["cam"].shape) == (2, 7, 7, 4)
    for i in range(2):
        for c in range(4):
            grad = gradient_grad_cam(trained, x[i : i + 1], c)
            cam = np.asarray(out["cam"])[i, :, :, c]
            # Grad-CAM = CAM / (H * W) exactly, before any normalisation
            np.testing.assert_allclose(grad, cam / 49.0, rtol=1e-3, atol=1e-5)
            assert pearson(normalize_cam(grad), normalize_cam(cam)) > 0.9999


def test_normalize_and_colormap():
    assert normalize_cam(np.array([[-1.0, 0.0], [2.0, 4.0]])).tolist() == [[0, 0], [0.5, 1]]
    assert normalize_cam(-np.ones((7, 7))).max() == 0
    assert jet(np.array(0.0)).tolist() == [0, 0, 128]
    assert jet(np.array(1.0)).tolist() == [128, 0, 0]
    rgb = overlay(np.zeros((224, 224, 3), np.uint8), np.eye(7, dtype=np.float32))
    assert rgb.shape == (224, 224, 3)
    assert to_png(rgb)[:8] == b"\x89PNG\r\n\x1a\n"
