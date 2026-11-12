"""The input check's rule and its committed configuration (no model needed)."""

import json
from pathlib import Path

import numpy as np
import pytest

from poultry_disease import CLASS_NAMES
from poultry_disease.input_check import (
    CONFIG,
    PROTOTYPES,
    THRESHOLD,
    VERSION,
    check_embedding,
    input_check_score,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_matches_classes_and_shape():
    assert CONFIG["prototypes"]["classes"] == list(CLASS_NAMES)
    assert PROTOTYPES.shape == (len(CLASS_NAMES), CONFIG["model"]["embedding_dim"])
    np.testing.assert_allclose(np.linalg.norm(PROTOTYPES, axis=1), 1.0, atol=1e-6)
    assert 0 < THRESHOLD < 1


def test_threshold_matches_committed_evaluation():
    ev = json.loads((ROOT / "results" / "input_check" / "evaluation.json").read_text("utf-8"))
    assert ev["threshold"] == THRESHOLD
    assert ev["version"] == VERSION
    tuning = CONFIG["tuning"]["false_rejection"]
    assert tuning["rate"] <= 0.01


def test_prototype_itself_scores_one_and_passes():
    for p in PROTOTYPES:
        assert input_check_score(p) == pytest.approx(1.0, abs=1e-6)
        assert check_embedding(p * 3.7).passed  # scale does not matter


def test_orthogonal_embedding_fails():
    rng = np.random.default_rng(0)
    v = rng.normal(size=PROTOTYPES.shape[1])
    # remove every prototype direction: cosine 0 with all of them
    q, _ = np.linalg.qr(PROTOTYPES.T)
    v -= q @ (q.T @ v)
    res = check_embedding(v)
    assert res.score == pytest.approx(0.0, abs=1e-9)
    assert not res.passed
    assert res.threshold == THRESHOLD


def test_score_is_cosine_with_nearest_prototype():
    # an embedding at cosine = threshold from prototype 0 and orthogonal to the rest of the prototype span
    p = PROTOTYPES[0]
    q, _ = np.linalg.qr(PROTOTYPES.T)
    rng = np.random.default_rng(1)
    o = rng.normal(size=p.shape)
    o -= q @ (q.T @ o)
    o /= np.linalg.norm(o)
    e = THRESHOLD * p + np.sqrt(1 - THRESHOLD**2) * o
    s = input_check_score(e)
    assert s == pytest.approx(THRESHOLD, abs=1e-6)


@pytest.mark.parametrize("bad", [np.zeros(512), np.full(512, np.nan), np.ones(10)])
def test_rejects_malformed_embeddings(bad):
    with pytest.raises(ValueError):
        input_check_score(bad)
