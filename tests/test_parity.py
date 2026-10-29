"""Parity test 1: the package reproduces the probabilities the notebook recorded in Colab.

Colab ran on a T4 GPU, these tests on CPU, so the probabilities are not bit-identical. On the reference
machine the largest difference over the 18 fixtures is about 2e-6 (results/parity/); the tolerance leaves
room for other CPUs and BLAS builds while still catching any preprocessing change, which moves
probabilities by far more.
"""

import numpy as np
import pytest

from poultry_disease import CLASS_NAMES
from poultry_disease.cam import gradient_grad_cam, normalize_cam, pearson
from poultry_disease.preprocess import to_model_input
from tests.conftest import FIXTURES

pytestmark = [pytest.mark.weights, pytest.mark.fixtures]
PROB_TOLERANCE = 1e-4


def test_probabilities_match_colab(predictor, fixture_manifest):
    worst = 0.0
    for im in fixture_manifest:
        pred = predictor.predict_bytes((FIXTURES / "images" / im["file"]).read_bytes())
        colab = np.array([im["colab_probabilities"][c] for c in CLASS_NAMES])
        assert pred.predicted_class == im["colab_predicted_class"], im["file"]
        diff = float(np.abs(pred.probs - colab).max())
        assert diff < PROB_TOLERANCE, f"{im['file']}: max abs diff {diff:.2e}"
        worst = max(worst, diff)
    print(f"max abs probability difference vs Colab: {worst:.2e}")


def test_cam_matches_gradient_gradcam(predictor, fixture_manifest):
    for im in fixture_manifest:
        pred = predictor.predict_bytes((FIXTURES / "images" / im["file"]).read_bytes())
        grad = gradient_grad_cam(predictor.trained, to_model_input(pred.seen), pred.top_index)
        r = pearson(normalize_cam(grad), pred.cam_for())
        assert r > 0.99999, f"{im['file']}: r = {r}"
