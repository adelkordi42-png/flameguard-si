"""FlameGuard SI — inference service.

    uvicorn flameguard.api:app --port 7860

Serves the single-page demo and the typed verdict API. The request schema is
generated at startup from ``artifacts/features.json``, so input validation
always tracks the schema the model was actually trained on.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import core

STATIC_DIR = core.ROOT / "static"
STATE: dict[str, Any] = {}

O2_HINTS = ("o2", "oxygen", "oxidizer")
O2_EXCLUDE = ("co2", "carbon", "co_2")
PRESSURE_HINTS = ("pressure", "kpa", "atm", "psia", "torr", "bar")
CONFINEMENT_HINTS = ("geo", "confin", "enclos", "config")
FLOW_HINTS = ("velocity", "veloc", "flow", "speed")
CONFINED_MARKERS = ("confin", "enclos", "closed", "seal", "cavity", "annular")
OPEN_MARKERS = ("open", "unconfin", "ventilat", "free", "unbounded")

logger = core.get_logger()


def _locate(meta: dict[str, Any], hints: tuple[str, ...], kinds: tuple[str, ...] = ("numeric",)) -> str | None:
    """Find a physical role by token match first, substring second."""
    exact: str | None = None
    partial: str | None = None
    for name, spec in meta["features"].items():
        if spec["kind"] not in kinds:
            continue
        lowered = name.lower()
        tokens = {token for token in re.split(r"[^a-z0-9]+", lowered) if token}
        if hints is O2_HINTS and any(marker in lowered for marker in O2_EXCLUDE):
            continue
        if tokens & set(hints):
            exact = exact or name
        elif any(hint in lowered for hint in hints):
            partial = partial or name
    return exact or partial


def _confined(value: Any) -> bool | None:
    text = str(value).lower()
    if any(marker in text for marker in OPEN_MARKERS):
        return False
    if any(marker in text for marker in CONFINED_MARKERS):
        return True
    return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        bundle = core.load_bundle()
    except FileNotFoundError as exc:
        # Serve anyway: /api/health reports 503 and the SPA shows "model unavailable".
        logger.error("model unavailable: %s", exc)
        yield
        return
    STATE.update(
        bundle=bundle,
        meta=bundle["meta"],
        model=bundle["model"],
        schema=core.scenario_model(bundle["meta"]),
        o2_column=_locate(bundle["meta"], O2_HINTS),
        pressure_column=_locate(bundle["meta"], PRESSURE_HINTS),
        confinement_column=_locate(bundle["meta"], CONFINEMENT_HINTS, ("categorical",)),
        flow_column=_locate(bundle["meta"], FLOW_HINTS),
    )
    core.log_event(
        logger,
        "service_ready",
        model_version=bundle["model_version"],
        data_source=bundle["meta"]["data_source"],
        classes=[str(c) for c in bundle["classes"]],
        rows=len(bundle["rows"]),
        o2_column=STATE["o2_column"],
        pressure_column=STATE["pressure_column"],
        confinement_column=STATE["confinement_column"],
    )
    if bundle["meta"]["data_source"] != "provided":
        logger.warning("model trained on SYNTHETIC placeholder data; verdicts are not NASA-validated")
    yield
    STATE.clear()


app = FastAPI(title="FlameGuard SI", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict[str, Any]:
    if not STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_version": STATE["bundle"]["model_version"],
        "data_source": STATE["meta"]["data_source"],
    }


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    """Drives the SPA form: exact feature list, kinds, ranges and categories."""
    if not STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    bundle = STATE["bundle"]
    metrics: dict[str, Any] = {}
    metrics_path = core.paths()["metrics"]
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metrics = {}
    if metrics.get("model_version") != bundle["model_version"]:
        metrics = {}  # never present another run's numbers as this model's
    return {
        "model_version": bundle["model_version"],
        "data_source": bundle["meta"]["data_source"],
        "classes": [str(c) for c in bundle["classes"]],
        "features": bundle["meta"]["features"],
        "tau_global": bundle["tau_global"],
        "tau_margin": bundle["tau_margin"],
        "k_neighbours": core.K_NEIGHBOURS,
        "training_rows": len(bundle["rows"]),
        "suppression_basis": "rule-based lookup (published NASA fire-safety guidance); not learned from data",
        "roles": {
            "o2": STATE["o2_column"],
            "pressure": STATE["pressure_column"],
            "confinement": STATE["confinement_column"],
            "flow": STATE["flow_column"],
        },
        "metrics": {
            key: metrics.get(key)
            for key in ("accuracy", "macro_f1", "baseline_macro_f1", "calibrated", "split", "abstention")
            if metrics
        },
    }


@app.post("/api/predict")
def predict(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    started = time.perf_counter()
    try:
        scenario = STATE["schema"](**payload)
    except Exception as exc:  # pydantic ValidationError -> 422 with field paths
        errors = json.loads(exc.json()) if hasattr(exc, "json") else [{"msg": str(exc)}]
        core.log_event(logger, "predict_rejected", errors=errors, level=logging.WARNING)
        raise HTTPException(status_code=422, detail=errors) from exc

    values = scenario.model_dump()
    bundle, meta = STATE["bundle"], STATE["meta"]
    x_model = core.encode_model([values], meta)
    probabilities = bundle["model"].predict_proba(x_model)[0]
    model_classes = [str(c) for c in bundle["model"].classes_]
    best = int(np.argmax(probabilities))
    raw_verdict = model_classes[best]
    confidence = float(probabilities[best])
    abstained, reason = core.abstain(probabilities, bundle["tau_global"], bundle["tau_margin"])
    outside = core.outside_observed_range(values, meta)
    if not abstained and outside:
        abstained, reason = True, "input_outside_observed_range"

    def numeric(column: str | None) -> float | None:
        if column is None:
            return None
        try:
            return float(values[column])
        except (TypeError, ValueError):
            return None

    confined = _confined(values[STATE["confinement_column"]]) if STATE["confinement_column"] else None
    suppression = core.suppression_for(
        core.VERDICT_REVIEW if abstained else raw_verdict,
        core.o2_fraction(numeric(STATE["o2_column"])),
        core.pressure_kpa(numeric(STATE["pressure_column"])),
        confined,
    )
    x_knn = core.encode_knn(x_model, meta)
    similar = core.nearest_runs(x_knn[0], bundle["x_knn"], bundle["rows"], bundle["labels"], meta)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    core.log_event(
        logger,
        "predict",
        latency_ms=latency_ms,
        verdict=core.VERDICT_REVIEW if abstained else raw_verdict,
        raw_verdict=raw_verdict,
        max_prob=round(confidence, 4),
        abstained=abstained,
        abstain_reason=reason,
        outside_features=outside,
        nearest_distance=similar[0]["distance"] if similar else None,
    )
    return {
        "model_version": bundle["model_version"],
        "verdict": core.VERDICT_REVIEW if abstained else raw_verdict,
        "raw_verdict": raw_verdict,
        "confidence": round(confidence, 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(model_classes, probabilities)},
        "abstain": abstained,
        "abstain_reason": reason,
        "outside_features": outside,
        "suppression": suppression,
        "suppression_basis": "rule-based lookup (published NASA fire-safety guidance); not learned from data",
        "similar_runs": similar,
        "data_source": meta["data_source"],
        "latency_ms": latency_ms,
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")