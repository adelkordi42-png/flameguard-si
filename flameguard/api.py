"""FlameGuard SI — inference service.

    uvicorn flameguard.api:app --port 7860

Cache-first: a cached Jev response plus the combiner answers in milliseconds. A
cache miss calls Jev only when JEV_MODE allows it and the network probe passes;
otherwise the verdict is `needs_human_review` with a typed reason. The API never
turns a Jev/network failure into a 500.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import combiner, core, dataset, features, jev, llm

STATIC_DIR = core.ROOT / "static"
STATE: dict[str, Any] = {}
log = core.get_logger()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        schema = dataset.load_schema()
        bundle = combiner.load()
        runs = dataset.load_runs()
        cases = dataset.load_cases()
    except (FileNotFoundError, dataset.DataError, ValueError) as exc:
        log.error("model unavailable: %s", exc)
        yield
        return
    provider = (os.environ.get("ENGINE_PROVIDER") or "").strip().lower()
    feature_mode = bundle.get("feature_mode", "engine+margins")
    engine = None if (provider == "physics" or feature_mode == "margins_only") else llm.build_engine()
    STATE.update(
        schema=schema,
        bundle=bundle,
        runs=runs,
        model=core.scenario_model(schema),
        canonical=dataset.canonical_cases(runs, cases),
        novel=dataset.novel_cases(cases),
        engine=engine,
        metrics=core.read_json(core.paths()["metrics"], {}) or {},
    )
    core.log_event(
        log,
        "service_ready",
        n_runs=len(runs),
        reduced_claims=schema["reduced_claims"],
        provenance_warning=schema["provenance_warning"],
        cache_entries=core.cache_size(),
        tau=bundle.get("tau_global"),
        engine_mode=jev.mode(),
    )
    yield
    STATE.clear()


app = FastAPI(title="FlameGuard SI", version="2.0.0", lifespan=lifespan)


def _gate() -> dict[str, Any]:
    schema = STATE["schema"]
    return {
        "n_runs": schema["n_runs"],
        "gate": schema["gate"],
        "reduced_claims": schema["reduced_claims"],
        "provenance_warning": schema["provenance_warning"],
        "tier_a": schema["tier_a"],
        "derived_share": schema["derived_share"],
    }


def _unavailable(reason: str, detail: str, values: dict[str, Any]) -> dict[str, Any]:
    strategy, detail_text = core.suppression_for(
        core.VERDICT_REVIEW,
        values.get("o2_percent"),
        values.get("pressure_kpa"),
        values.get("opposed_flow_cm_s"),
        None,
        True,
    )
    engine = STATE.get("engine")
    provider = getattr(engine, "provider", "unavailable")
    return {
        "engine": "unavailable",
        "engine_provider": provider,
        "engine_label": jev.engine_label(provider) if provider != "unavailable" else "unavailable",
        "model_name": getattr(engine, "model", None),
        "verdict": core.VERDICT_REVIEW,
        "raw_verdict": None,
        "calibrated_probability": None,
        "probabilities": None,
        "abstain": True,
        "abstain_reason": reason,
        "reason_codes": [reason],
        "suppression_strategy": strategy,
        "suppression_detail": detail_text,
        "suppression_basis": "rule-based policy; not learned from data",
        "jev": None,
        "note": detail,
        "data_gate": _gate(),
    }


def _decide(values: dict[str, Any], allow_live: bool | None) -> dict[str, Any]:
    outside = core.outside_observed_range(values, STATE["schema"])
    if outside:
        result = _unavailable("input_outside_observed_range", f"outside observed range: {outside}", values)
        result["engine"] = "jev_skipped_ood"
        return result

    margin_vector = features.vector(values)
    feature_mode = STATE["bundle"].get("feature_mode", "engine+margins")
    response = None
    engine_note = ""
    if feature_mode == "margins_only":
        engine_note = "margins-only model: the engine was not called"
    else:
        try:
            response = STATE["engine"].answer(values, allow_live=allow_live)
        except (jev.JevError, ValueError, KeyError, TypeError) as exc:
            core.log_event(log, "engine_unavailable", error=str(exc), level=logging.WARNING)
            return _unavailable("jev_unavailable_no_cache", str(exc), values)

    answers = response.answers if response is not None else None
    vector = combiner.answers_to_vector(answers, margin_vector)
    try:
        decision = combiner.predict(STATE["bundle"], vector)
    except (ValueError, KeyError) as exc:
        core.log_event(log, "combiner_failed", error=str(exc), level=logging.ERROR)
        return _unavailable("combiner_response_invalid", str(exc), values)

    abstained, reason = core.abstain(
        decision["probabilities"],
        STATE["bundle"].get("tau_global", core.TAU_GLOBAL),
        STATE["bundle"].get("tau_margin", core.TAU_MARGIN),
    )
    signals = response.signals() if response is not None else features.signals(values)
    strategy, detail = core.suppression_for(
        decision["verdict"],
        values.get("o2_percent"),
        values.get("pressure_kpa"),
        values.get("opposed_flow_cm_s"),
        signals.get("severity"),
        abstained,
    )
    codes = core.reason_codes(signals, decision["verdict"], reason)
    if response is None:
        codes = list(dict.fromkeys(codes + ["physics_margins_engine"]))
    provider = response.provider if response is not None else "physics"
    return {
        "engine": response.engine if response is not None else "margin_fallback",
        "engine_provider": provider,
        "engine_label": jev.engine_label(provider) if provider != "physics" else "published-threshold margins",
        "model_name": response.model if response is not None else "published-threshold margins",
        "verdict": core.VERDICT_REVIEW if abstained else decision["verdict"],
        "raw_verdict": decision["verdict"],
        "calibrated_probability": round(decision["confidence"], 4),
        "probabilities": {name: round(value, 4) for name, value in decision["probabilities"].items()},
        "abstain": abstained,
        "abstain_reason": reason,
        "reason_codes": codes,
        "suppression_strategy": strategy,
        "suppression_detail": detail,
        "suppression_basis": "rule-based policy from published NASA fire-safety guidance; not learned from data",
        "jev": response.as_dict() if response is not None else None,
        "note": engine_note,
        "data_gate": _gate(),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    if "schema" not in STATE or "bundle" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    engine_obj = STATE["engine"]
    provider = getattr(engine_obj, "provider", "physics") if engine_obj is not None else "physics"
    return {
        "status": "ok",
        "engine_mode": jev.mode(),
        "engine_provider": provider,
        "engine_label": jev.engine_label(provider) if engine_obj is not None else "published-threshold margins",
        "feature_mode": STATE["bundle"].get("feature_mode", "engine+margins"),
        "cache_entries": core.cache_size(),
        "model_name": getattr(engine_obj, "model", "published-threshold margins"),
        "n_runs": STATE["schema"]["n_runs"],
        "reduced_claims": STATE["schema"]["reduced_claims"],
        "provenance_warning": STATE["schema"]["provenance_warning"],
        "schema_version": core.SCHEMA_VERSION,
    }


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    if "schema" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    schema = STATE["schema"]
    engine_obj = STATE["engine"]
    provider = getattr(engine_obj, "provider", "physics") if engine_obj is not None else "physics"
    return {
        "engine_provider": provider,
        "engine_label": jev.engine_label(provider) if engine_obj is not None else "published-threshold margins",
        "feature_mode": STATE["bundle"].get("feature_mode", "engine+margins"),
        "model_name": getattr(engine_obj, "model", "published-threshold margins"),
        "classes": list(core.CLASSES),
        "strategies": list(core.STRATEGIES),
        "strategy_labels": core.STRATEGY_LABELS,
        "fields": schema["fields"],
        "defaults": STATE["canonical"][0]["scenario"],
        "tau_global": STATE["bundle"].get("tau_global", core.TAU_GLOBAL),
        "tau_margin": STATE["bundle"].get("tau_margin", core.TAU_MARGIN),
        "n_runs": schema["n_runs"],
        "gate": schema["gate"],
        "reduced_claims": schema["reduced_claims"],
        "provenance_warning": schema["provenance_warning"],
        "tier_a": schema["tier_a"],
        "derived_share": schema["derived_share"],
        "label_counts": schema["label_counts"],
        "basis_counts": schema["basis_counts"],
        "data_version": schema["data_version"],
        "schema_version": schema["schema_version"],
        "suppression_basis": "rule-based policy; not learned from data",
    }


@app.get("/api/cases")
def cases() -> dict[str, Any]:
    if "schema" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    canonical = []
    for item in STATE["canonical"]:
        result = _decide(item["scenario"], allow_live=False)
        predicted = result["raw_verdict"]
        canonical.append(
            {
                "run_id": item["run_id"],
                "family": item["family"],
                "scenario": item["scenario"],
                "actual_label": item["actual_label"],
                "label_basis": item["label_basis"],
                "source": item["source"],
                "outcome_raw": item["outcome_raw"],
                "predicted": predicted,
                "probability": result["calibrated_probability"],
                "abstained": result["abstain"],
                "match": (predicted == item["actual_label"]) if predicted else None,
                "engine": result["engine"],
            }
        )
    novel = []
    for item in STATE["novel"]:
        result = _decide(item["scenario"], allow_live=False)
        novel.append(
            {
                "id": item["id"],
                "title": item["title"],
                "scenario": item["scenario"],
                "predicted": result["raw_verdict"],
                "probability": result["calibrated_probability"],
                "abstained": result["abstain"],
                "engine": result["engine"],
            }
        )
    return {"canonical": canonical, "novel": novel}


@app.post("/api/predict")
def predict(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if "schema" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        values = STATE["model"].model_validate(payload).model_dump()
    except ValidationError as exc:
        raise RequestValidationError(json.loads(exc.json())) from exc
    started = time.perf_counter()
    result = _decide(values, allow_live=None)
    result["model_version"] = STATE["schema"]["data_version"]
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    core.log_event(
        log,
        "predict",
        engine=result["engine"],
        verdict=result["verdict"],
        abstain=result["abstain"],
        reason=result["abstain_reason"],
        latency_ms=result["latency_ms"],
    )
    return result


@app.get("/api/metrics")
def metrics() -> FileResponse:
    path = core.paths()["metrics"]
    if not path.exists():
        raise HTTPException(status_code=503, detail="no validation metrics; run python -m flameguard.evaluate")
    return FileResponse(path, media_type="application/json")


@app.get("/VALIDATION.md")
def validation() -> FileResponse:
    path = core.paths()["validation"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="VALIDATION.md not found")
    return FileResponse(path, media_type="text/markdown; charset=utf-8")


@app.get("/metrics")
def metrics_page() -> FileResponse:
    path = STATIC_DIR / "metrics.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics page not found")
    return FileResponse(path, media_type="text/html; charset=utf-8")


@app.get("/deck")
def deck_page() -> FileResponse:
    path = core.ROOT / "docs" / "deck.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="deck not found")
    return FileResponse(path, media_type="text/html; charset=utf-8")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")