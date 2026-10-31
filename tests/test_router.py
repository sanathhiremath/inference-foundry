import numpy as np
import pytest
from fastapi.testclient import TestClient
from inference_foundry.api import create_app
from inference_foundry.experiment import calibrate, experiment, predict_one
from inference_foundry.models import pooled, train_bundle
from inference_foundry.policy import RoutingConstraints, select_threshold


def test_selects_least_strong_usage_that_meets_quality():
    selected, _ = select_threshold([.2, .6, .9, .99], [0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], RoutingConstraints(.75, .5))
    assert selected["strong_fraction"] == .25
    assert selected["accuracy"] == .75


def test_infeasible_policy_is_explicit_instead_of_relaxed():
    selected, _ = select_threshold([.2, .5], [0, 0], [0, 0], [1, 1], RoutingConstraints(.9, 1))
    assert selected is None


def test_equal_confidence_cannot_be_arbitrarily_split():
    selected, frontier = select_threshold([.5, .5], [0, 0], [1, 1], [1, 1], RoutingConstraints(.5, .5))
    assert selected is None
    assert all(point["strong_fraction"] in (0, 1) for point in frontier)


def test_full_budget_can_override_an_overconfident_cheap_model():
    selected, _ = select_threshold([1.0], [0], [1], [1], RoutingConstraints(1, 1))
    assert selected is not None
    assert selected["strong_fraction"] == 1
    assert selected["accuracy"] == 1


@pytest.mark.parametrize("confidence", [[], [float('nan')], [-.1], [1.1]])
def test_invalid_calibration_input_rejected(confidence):
    with pytest.raises(ValueError):
        select_threshold(confidence, [1], [1], [1], RoutingConstraints())


def test_all_splits_are_disjoint_and_cover_the_dataset():
    bundle = train_bundle()
    sets = [set(bundle[name]) for name in ("train", "calibration", "test")]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert len(set.union(*sets)) == 1797


def test_policy_selection_cannot_read_test_labels():
    bundle = train_bundle()
    original, _ = calibrate(bundle, RoutingConstraints())
    altered = {**bundle, "labels": bundle["labels"].copy()}
    altered["labels"][bundle["test"]] = (altered["labels"][bundle["test"]] + 1) % 10
    after, _ = calibrate(altered, RoutingConstraints())
    assert original == after


def test_pooling_preserves_constant_signal():
    result = pooled(np.ones((2, 64)) * 3)
    assert result.shape == (2, 16)
    assert np.all(result == 3)


@pytest.mark.parametrize("features", [[0] * 63, [float('nan')] * 64, [17] * 64])
def test_invalid_prediction_inputs_rejected(features):
    with pytest.raises(ValueError):
        predict_one(train_bundle(), features, .7)


def test_real_benchmark_keeps_baselines_and_observation_evidence():
    report = experiment(RoutingConstraints())
    assert report["status"] == "CALIBRATED"
    assert report["test"]["routed"]["count"] == 360
    assert len(report["observations"]) == 1080
    assert report["selection"]["strong_fraction"] <= .4
    assert report["selection"]["accuracy"] >= .94
    assert all(point["latency_ms_p95"] >= point["latency_ms_p50"] >= 0 for point in report["test"].values())


def test_api_validation_routing_and_infeasible_readiness():
    bundle = train_bundle()
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post("/predict", json={"pixels": bundle["data"][bundle["test"][0]].tolist()})
        assert response.status_code == 200
        assert response.json()["path"] in ("cheap", "strong")
        assert client.post("/predict", json={"pixels": [0] * 63}).status_code == 422
        assert client.post("/predict", json={"pixels": [17] * 64}).status_code == 422
    with TestClient(create_app(RoutingConstraints(1, 0))) as client:
        assert client.get("/health").json()["status"] == "infeasible"
        assert client.post("/predict", json={"pixels": [0] * 64}).status_code == 503
