"""Smoke tests for criteria F1, F4, F5, F6, F7 and the served SPA (F8)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from flameguard import core, train


@pytest.fixture(scope="session")
def artifacts():
    if not core.paths()["bundle"].exists():
        assert train.main(["--synthetic-rows", "150"]) == 0
    return core.load_bundle()


@pytest.fixture(scope="session")
def client(artifacts):
    from flameguard.api import app

    with TestClient(app) as test_client:
        yield test_client


def valid_payload(bundle) -> dict:
    row = bundle["rows"][0]
    return {
        name: float(row[name]) if spec["kind"] == "numeric" else row[name]
        for name, spec in bundle["meta"]["features"].items()
    }


def test_f1_schema_rejects_unknown_category_missing_and_extra_fields(artifacts):
    schema = core.scenario_model(artifacts["meta"])
    payload = valid_payload(artifacts)
    schema(**payload)
    categorical = next(
        (n for n, s in artifacts["meta"]["features"].items() if s["kind"] == "categorical"), None
    )
    if categorical:
        with pytest.raises(ValidationError):
            schema(**{**payload, categorical: "definitely-not-a-category"})
    with pytest.raises(ValidationError):
        schema(**{k: v for k, v in payload.items() if k != next(iter(payload))})
    with pytest.raises(ValidationError):
        schema(**{**payload, "unexpected_field": 1})


def test_f4_abstention_on_flat_low_margin_and_confident_vectors():
    assert core.abstain(np.array([0.34, 0.33, 0.33]))[0] is True
    assert core.abstain(np.array([0.55, 0.45, 0.00]))[0] is True
    assert core.abstain(np.array([0.62, 0.55, 0.00]))[0] is True
    assert core.abstain(np.array([0.90, 0.05, 0.05]))[0] is False
    assert core.abstain(np.array([0.90, 0.05, 0.05]))[1] == ""


def test_f5_suppression_is_total_over_every_band():
    for verdict in ("sustained_spread", "marginal", "extinction", core.VERDICT_REVIEW):
        for o2 in (0.10, 0.18, 0.25, 0.40, 1.00):
            for pressure in (30.0, 60.0, 95.0, 150.0):
                for confined in (True, False, None):
                    text = core.suppression_for(verdict, o2, pressure, confined)
                    assert isinstance(text, str) and text.strip()
    assert core.suppression_for("sustained_spread").strip()
    assert "Abstained" in core.suppression_for(core.VERDICT_REVIEW)


def test_f5_physical_units_are_normalised():
    assert core.o2_fraction(21.0) == pytest.approx(0.21)
    assert core.o2_fraction(0.21) == pytest.approx(0.21)
    assert core.o2_fraction(None) is None
    assert core.pressure_kpa(1.0) == pytest.approx(101.325)
    assert core.pressure_kpa(101.3) == pytest.approx(101.3)
    assert core.pressure_kpa(None) is None
    assert "below_limiting" in core.suppression_for("extinction", 0.14, 101.0)


def test_f6_nearest_runs_ordered_labelled_and_exact_match(artifacts):
    similar = core.nearest_runs(
        artifacts["x_knn"][0],
        artifacts["x_knn"],
        artifacts["rows"],
        artifacts["labels"],
        artifacts["meta"],
        k=4,
    )
    assert len(similar) == 4
    distances = [s["distance"] for s in similar]
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0.0, abs=1e-9)
    assert all(s["actual_label"] in set(map(str, artifacts["classes"])) for s in similar)


def test_f7_health_and_meta_expose_the_trained_schema(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    body = client.get("/api/meta").json()
    assert body["features"] and body["classes"] and body["training_rows"] > 0
    assert body["suppression_basis"].startswith("rule-based")


def test_f7_predict_contract_and_rejection(client, artifacts):
    response = client.post("/api/predict", json=valid_payload(artifacts))
    assert response.status_code == 200, response.text
    body = response.json()
    for key in ("verdict", "confidence", "abstain", "suppression", "similar_runs", "model_version", "latency_ms"):
        assert key in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["similar_runs"]) == core.K_NEIGHBOURS
    assert body["latency_ms"] < 1000
    if body["abstain"]:
        assert body["verdict"] == core.VERDICT_REVIEW
    assert client.post("/api/predict", json={"nonsense": 1}).status_code == 422


def test_f8_spa_is_served_at_root(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "FlameGuard" in page.text


def test_label_normalisation_is_negation_aware():
    assert core.normalize_class("did not spread") == "extinction"
    assert core.normalize_class("extinguished") == "extinction"
    assert core.normalize_class("no ignition") == "extinction"
    assert core.normalize_class("self-extinguished") == "extinction"
    assert core.normalize_class("sustained spread") == "sustained_spread"
    assert core.normalize_class("marginal spread") == "marginal"
    assert core.normalize_class("") is None


def test_f4_input_outside_observed_range_forces_abstention(client, artifacts):
    numeric = [n for n, s in artifacts["meta"]["features"].items() if s["kind"] == "numeric"]
    if not numeric:
        pytest.skip("no numeric features to push out of range")
    payload = valid_payload(artifacts)
    payload[numeric[0]] = artifacts["meta"]["features"][numeric[0]]["max"] * 10 + 1000
    body = client.post("/api/predict", json=payload).json()
    assert body["abstain"] is True
    assert body["abstain_reason"] == "input_outside_observed_range"
    assert body["verdict"] == core.VERDICT_REVIEW
    assert numeric[0] in body["outside_features"]


def test_f6_similar_runs_expose_only_feature_columns(artifacts):
    similar = core.nearest_runs(
        artifacts["x_knn"][0],
        artifacts["x_knn"],
        artifacts["rows"],
        artifacts["labels"],
        artifacts["meta"],
        k=1,
    )
    assert set(similar[0]["record"]).issubset(set(artifacts["meta"]["features"]))
