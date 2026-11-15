"""Contract tests for the data gate, Jev parsing, combiner, API and offline path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from flameguard import combiner, core, dataset, jev, llm, precompute


def fake_answers(label: str = "sustained_spread") -> dict:
    probabilities = {"extinction": 0.1, "marginal": 0.2, "sustained_spread": 0.7}
    probabilities[label] = 0.7
    others = [name for name in probabilities if name != label]
    probabilities[others[0]] = 0.2
    probabilities[others[1]] = 0.1
    return {
        "o2_above_limiting": {"type": "noul", "noul": 0.8},
        "flow_above_blowoff": {"type": "noul", "noul": 0.2},
        "self_sustaining_60s": {"type": "noul", "noul": 0.7},
        "spread_beyond_ignition": {"type": "noul", "noul": 0.6},
        "flame_class": {
            "type": "choice",
            "choice": label,
            "confidence": 0.7,
            "probabilities": probabilities,
        },
        "severity": {
            "type": "score",
            "score": 2.0,
            "confidence": 0.6,
            "legend": {"0": "none", "1": "low", "2": "steady", "3": "severe"},
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.2},
        },
    }


@pytest.fixture(scope="module")
def runs():
    return dataset.load_runs()


@pytest.fixture(scope="module")
def schema(runs):
    return dataset.build_schema(runs, dataset.audit(runs))


def test_data_gate_and_provenance(runs):
    report = dataset.audit(runs)
    assert report["n_runs"] >= report["gate"] >= 150
    assert report["reduced_claims"] is False
    assert set(report["label_counts"]) == set(core.CLASSES)
    assert report["tier_a"] >= 20
    for run in runs:
        assert run["source"]["doc_id"] and run["source"]["table"] and run["source"]["url"]
        assert run["outcome_quote"]
        assert run["label_basis"] in {"published", "derived"}


def test_canonical_cases_exist_and_match_labels(runs):
    cases = dataset.load_cases()
    canonical = dataset.canonical_cases(runs, cases)
    assert len(canonical) == 6
    assert len(dataset.novel_cases(cases)) == 8
    by_id = {run["run_id"]: run for run in runs}
    for item in canonical:
        assert item["actual_label"] == by_id[item["run_id"]]["label"]


def test_scenario_model_rejects_unknown_category_and_extra(schema):
    model = core.scenario_model(schema)
    run = dataset.load_runs()[0]
    values = {name: run[name] for name in dataset.CONDITION_FIELDS}
    model(**values)
    with pytest.raises(ValidationError):
        model(**{**values, "material": "unobtainium"})
    with pytest.raises(ValidationError):
        model(**{**values, "unexpected": 1})


def test_all_canonical_and_novel_scenarios_are_in_schema(schema):
    model = core.scenario_model(schema)
    runs = dataset.load_runs()
    for item in dataset.canonical_cases(runs, dataset.load_cases()):
        model(**item["scenario"])
    for item in dataset.novel_cases(dataset.load_cases()):
        model(**item["scenario"])


def test_cache_roundtrip_and_corruption(tmp_path):
    cache = tmp_path / "engine_cache.json"
    core.cache_put("abc", {"answers": {}}, cache)
    assert core.cache_get("abc", cache) == {"answers": {}}
    cache.write_text("{not json", encoding="utf-8")
    assert core.cache_read(cache)["entries"] == {}
    assert list(tmp_path.glob("engine_cache.json.corrupt.*"))


def test_parse_answers_valid_and_invalid():
    parsed = jev.parse_answers(fake_answers())
    assert parsed["flame_class"]["choice"] == "sustained_spread"
    with pytest.raises(jev.JevError):
        jev.parse_answers({k: v for k, v in fake_answers().items() if k != "flame_class"})
    bad = fake_answers()
    bad["flame_class"]["probabilities"] = {"extinction": 0.1, "marginal": 0.1, "sustained_spread": 0.1}
    with pytest.raises(jev.JevError):
        jev.parse_answers(bad)
    bad = fake_answers()
    bad["o2_above_limiting"]["noul"] = 1.2
    with pytest.raises(jev.JevError):
        jev.parse_answers(bad)


def test_parse_answers_tolerates_null_confidence():
    answers = fake_answers()
    answers["flame_class"]["confidence"] = None
    answers["severity"]["confidence"] = None
    parsed = jev.parse_answers(answers)
    chosen = answers["flame_class"]["choice"]
    assert parsed["flame_class"]["confidence"] == answers["flame_class"]["probabilities"][chosen]
    assert parsed["severity"]["confidence"] == 0.0


def test_parse_answers_tolerates_choice_without_probabilities():
    answers = fake_answers()
    del answers["flame_class"]["probabilities"]
    parsed = jev.parse_answers(answers)
    assert parsed["flame_class"]["choice"] == "sustained_spread"
    assert parsed["flame_class"]["probabilities"]["sustained_spread"] == 1.0
    assert parsed["flame_class"]["probabilities"]["extinction"] == 0.0


def test_provider_defaults_and_key_aliases(monkeypatch):
    for name in (
        "JEV_API_KEY",
        "TYPESAFE_API_KEY",
        "BEATAPI_API_KEY",
        "AI_GATEWAY_API_KEY",
        "VERCEL_AI_GATEWAY_KEY",
        "CODIV_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JEV_MODEL", "")
    monkeypatch.setattr(jev, "BASE_URL", "https://api.beatapi.io")
    assert jev.model_name() == "jev-1.14"
    monkeypatch.setattr(jev, "BASE_URL", "https://ai-gateway.vercel.sh/typesafe")
    assert jev.model_name() == "typesafe-ai/jev"
    monkeypatch.setattr(jev, "BASE_URL", "https://api.typesafe.ai")
    assert jev.model_name() == "jev-latest"

    monkeypatch.setattr(jev, "BASE_URL", "https://api.beatapi.io")
    monkeypatch.setenv("BEATAPI_API_KEY", "beat-test")
    assert jev.api_key() == "beat-test" and jev.key_source() == "BEATAPI_API_KEY"

    monkeypatch.setattr(jev, "BASE_URL", "https://ai-gateway.vercel.sh/typesafe")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-test")
    assert jev.api_key() == "gw-test" and jev.key_source() == "AI_GATEWAY_API_KEY"

    monkeypatch.setattr(jev, "BASE_URL", "https://api.codiv.ai")
    monkeypatch.setenv("CODIV_API_KEY", "codiv-test")
    assert jev.api_key() == "codiv-test" and jev.key_source() == "CODIV_API_KEY"

    monkeypatch.setattr(jev, "BASE_URL", "https://example.test")
    monkeypatch.setenv("JEV_API_KEY", "generic-test")
    assert jev.api_key() == "generic-test" and jev.key_source() == "JEV_API_KEY"


def test_llm_engine_parses_caches_and_rejects(tmp_path):
    content = json.dumps(
        {
            "o2_above_limiting": 0.8,
            "flow_above_blowoff": 0.2,
            "self_sustaining_60s": 0.7,
            "spread_beyond_ignition": 0.6,
            "flame_class": {"extinction": 0.1, "marginal": 0.2, "sustained_spread": 0.7},
            "severity": 2.2,
        }
    )

    def transport(url, payload, headers, timeout):
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 700, "completion_tokens": 60},
        }

    cache = tmp_path / "llm_cache.json"
    engine = llm.LlmEngine(
        base="https://example.test/v1",
        key="test-key",
        model="free-model",
        transport=transport,
        cache_path=str(cache),
    )
    state = dataset.canonical_cases(dataset.load_runs(), dataset.load_cases())[0]["scenario"]
    first = engine.answer(state, allow_live=True)
    assert first.engine == "llm_live"
    assert set(first.signals()) == {
        "o2_above_limiting",
        "flow_above_blowoff",
        "self_sustaining_60s",
        "spread_beyond_ignition",
        "severity",
    }
    assert sum(first.answers["flame_class"]["probabilities"].values()) == pytest.approx(1.0)
    second = engine.answer(state, allow_live=True)
    assert second.engine == "llm_cache"

    def bad_transport(url, payload, headers, timeout):
        return {"choices": [{"message": {"content": "not json"}}], "usage": {}}

    bad = llm.LlmEngine(
        base="https://example.test/v1",
        key="test-key",
        model="free-model",
        transport=bad_transport,
        cache_path=str(tmp_path / "bad.json"),
    )
    with pytest.raises(jev.JevError):
        bad.answer(state, allow_live=True)


def test_build_engine_selects_by_env(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("ENGINE_PROVIDER", raising=False)
    assert isinstance(llm.build_engine(), jev.JevEngine)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    assert isinstance(llm.build_engine(), llm.LlmEngine)
    monkeypatch.setenv("ENGINE_PROVIDER", "openjev")
    assert isinstance(llm.build_engine(), jev.JevEngine)
    monkeypatch.setenv("ENGINE_PROVIDER", "llm")
    assert isinstance(llm.build_engine(), llm.LlmEngine)


def test_combiner_fit_predict_and_probability_sum():
    rng = np.random.default_rng(0)
    vectors, labels = [], []
    for label in core.CLASSES:
        base = combiner.answers_to_vector(jev.parse_answers(fake_answers(label)))
        for _ in range(12):
            vectors.append([min(1.0, max(0.0, value + 0.02 * rng.normal())) for value in base])
            labels.append(label)
    bundle = combiner.fit(vectors, labels)
    result = combiner.predict(bundle, vectors[0])
    assert result["verdict"] in core.CLASSES
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= result["confidence"] <= 1.0


def test_abstention_and_suppression_are_total():
    assert core.abstain({"a": 0.34, "b": 0.33, "c": 0.33})[0] is True
    assert core.abstain({"a": 0.55, "b": 0.44, "c": 0.01})[0] is True
    assert core.abstain({"a": 0.90, "b": 0.05, "c": 0.05})[0] is False
    assert core.abstain(None)[1] == "jev_unavailable_no_cache"
    for verdict in (*core.CLASSES, core.VERDICT_REVIEW):
        for o2 in (10.0, 21.0, 30.0, 40.0):
            for pressure in (50.0, 101.3, 200.0):
                for flow in (0.0, 5.0, 20.0):
                    for severity in (0.0, 3.0):
                        for abstained in (False, True):
                            strategy, detail = core.suppression_for(
                                verdict, o2, pressure, flow, severity, abstained
                            )
                            assert strategy in core.STRATEGIES
                            assert detail
    assert core.suppression_for("extinction", 21.0, 101.3, 0.0, 0.0, False)[0] == "none_needed"
    assert core.suppression_for(core.VERDICT_REVIEW, 21.0, 101.3, 0.0, 0.0, True)[0] == "abort_configuration"


def test_demo_states_are_14():
    states = precompute.demo_states()
    assert len(states) == 14
    assert len({name for name, _ in states}) == 14


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    runs = dataset.load_runs()
    report = dataset.audit(runs)
    schema = dataset.build_schema(runs, report)
    (tmp_path / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    scenario = dataset.canonical_cases(runs, dataset.load_cases())[0]["scenario"]

    rng = np.random.default_rng(1)
    vectors, labels = [], []
    for label in core.CLASSES:
        base = combiner.answers_to_vector(jev.parse_answers(fake_answers(label)))
        for _ in range(12):
            vectors.append([min(1.0, max(0.0, value + 0.02 * rng.normal())) for value in base])
            labels.append(label)
    bundle = combiner.fit(vectors, labels)
    bundle.update({"tau_global": 0.0, "tau_margin": 0.0, "jev_model": "jev-test"})
    combiner.save(bundle, tmp_path / "combiner.joblib")

    cache_path = tmp_path / "engine_cache.json"
    engine = jev.JevEngine(cache_path=str(cache_path))
    core.cache_put(
        engine.key_for(scenario),
        {
            "requested_model": engine.model,
            "model": "jev-test",
            "questions_version": jev.QUESTIONS_VERSION,
            "state": jev.render_state(scenario),
            "answers": fake_answers(),
            "usage": {},
            "fetched_at": core.iso_now(),
            "latency_ms": 1.0,
        },
        cache_path,
    )
    monkeypatch.setattr(core, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(core, "CACHE_PATH", cache_path)
    from flameguard.api import app

    with TestClient(app) as client:
        yield client, scenario


def test_api_health_meta_cases_and_predict(api_client):
    client, scenario = api_client
    health = client.get("/api/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert meta["n_runs"] >= 150
    assert meta["reduced_claims"] is False
    cases = client.get("/api/cases").json()
    assert len(cases["canonical"]) == 6
    assert len(cases["novel"]) == 8
    response = client.post("/api/predict", json=scenario)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine"] == "jev_cache"
    assert body["verdict"] in core.CLASSES
    assert 0.0 <= body["calibrated_probability"] <= 1.0
    assert body["suppression_strategy"] in core.STRATEGIES
    assert body["reason_codes"]
    assert body["latency_ms"] < 1000


def test_api_422_and_ood(api_client):
    client, scenario = api_client
    bad = client.post("/api/predict", json={**scenario, "material": "unobtainium"})
    assert bad.status_code == 422
    assert bad.json()["detail"]
    ood = client.post("/api/predict", json={**scenario, "o2_percent": 99.0})
    assert ood.status_code == 200
    body = ood.json()
    assert body["abstain"] is True
    assert body["abstain_reason"] == "input_outside_observed_range"
    assert body["engine"] == "jev_skipped_ood"


def test_api_offline_cache_miss_returns_review(api_client):
    client, scenario = api_client
    response = client.post("/api/predict", json={**scenario, "thickness_mm": 2.0})
    assert response.status_code == 200
    body = response.json()
    assert body["abstain"] is True
    assert body["abstain_reason"] == "jev_unavailable_no_cache"
    assert body["verdict"] == core.VERDICT_REVIEW
    assert body["calibrated_probability"] is None


def test_static_pages(api_client):
    client, _ = api_client
    assert client.get("/").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/deck").status_code == 200
    assert client.get("/api/metrics").status_code == 200


def test_cache_contains_no_secret():
    path = Path(core.CACHE_PATH)
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("no committed cache yet")
    text = path.read_text(encoding="utf-8")
    assert "Bearer" not in text
    assert "JEV_API_KEY" not in text