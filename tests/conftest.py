"""Pin the engine environment so tests never use the developer's real keys."""

from __future__ import annotations

import pytest

from flameguard import jev

ENGINE_ENV = (
    "JEV_API_KEY",
    "TYPESAFE_API_KEY",
    "BEATAPI_API_KEY",
    "AI_GATEWAY_API_KEY",
    "VERCEL_AI_GATEWAY_KEY",
    "CODIV_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "ENGINE_PROVIDER",
    "JEV_MODEL",
    "JEV_API_BASE",
    "JEV_MODE",
)

TEST_BASE = "http://127.0.0.1:1"


@pytest.fixture(autouse=True)
def hermetic_engine_env(monkeypatch):
    for name in ENGINE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JEV_API_BASE", TEST_BASE)
    monkeypatch.setenv("JEV_MODEL", "test-model")
    monkeypatch.setenv("JEV_API_KEY", "test")
    monkeypatch.setenv("ENGINE_PROVIDER", "jev")
    monkeypatch.setenv("JEV_MODE", "cache")
    monkeypatch.setattr(jev, "BASE_URL", TEST_BASE)
    monkeypatch.setattr(jev, "ENDPOINT", f"{TEST_BASE}/v1/systemone")
    monkeypatch.setattr(jev, "MODELS_ENDPOINT", f"{TEST_BASE}/v1/models")
    monkeypatch.setattr(jev, "PROBE_URL", f"{TEST_BASE}/openapi.json")
    yield