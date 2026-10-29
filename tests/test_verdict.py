import json

import pytest

from poultry_disease import verdict
from tests.conftest import ROOT


def test_thresholds_come_from_the_committed_analysis():
    published = json.loads((ROOT / "results" / "posthoc" / "thresholds.json").read_text("utf-8"))
    assert verdict.MIN_TOP_PROBABILITY == published["min_top_probability"]
    assert verdict.MIN_TOP2_MARGIN == published["min_top2_margin"]


def test_confident_disease_and_notifiable():
    v = verdict.interpret([0.01, 0.01, 0.01, 0.97])
    assert v.category == "disease_pattern" and v.top_class == "NCD" and v.notifiable
    assert not v.low_confidence


def test_confident_healthy_is_no_disease_pattern():
    v = verdict.interpret([0.0, 0.99, 0.01, 0.0])
    assert v.category == "no_disease_pattern" and not v.notifiable


def test_small_margin_is_uncertain_even_for_ncd():
    v = verdict.interpret([0.0, 0.3, 0.0, 0.7])
    assert v.category == "uncertain" and "small_margin" in v.reasons and v.notifiable
    assert v.margin == pytest.approx(0.4)


def test_ties_keep_model_order():
    v = verdict.interpret([0.25, 0.25, 0.25, 0.25])
    assert (v.top_class, v.runner_up) == ("Coccidiosis", "Healthy") and v.category == "uncertain"
    assert set(v.reasons) == {"low_top_probability", "small_margin"}


def test_wrong_length():
    with pytest.raises(ValueError):
        verdict.interpret([0.5, 0.5])
