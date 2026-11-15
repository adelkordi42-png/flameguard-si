"""FlameGuard SI — shared domain core.

Constants, non-blocking logging, JSON/cache I/O, the physical bands used by the
suppression policy, abstention, and the runtime request schema. Imported by the
dataset builder, evaluator, precompute tool and API so those four cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import os
import queue
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

try:  # Linux is the deployment target; the fallback keeps local tests portable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
CACHE_PATH = ROOT / "engine_cache.json"
SCHEMA_VERSION = "2026-11-15.2"

CLASSES = ("extinction", "marginal", "sustained_spread")
VERDICT_REVIEW = "needs_human_review"
STRATEGIES = ("none_needed", "flow_reduction", "diluent_gas", "water_mist", "abort_configuration")
STRATEGY_LABELS = {
    "none_needed": "No active suppression required",
    "flow_reduction": "Reduce forced convection / ventilation",
    "diluent_gas": "Diluent gas (CO2 / N2) inerting",
    "water_mist": "Water mist and boundary cooling",
    "abort_configuration": "Isolate / abort the configuration until review",
}
REASON_CODES = (
    "o2_below_limiting",
    "o2_near_limiting",
    "o2_above_limiting",
    "flow_above_blowoff",
    "flow_near_blowoff",
    "near_limit_behavior",
    "low_confidence",
    "ambiguous_top_classes",
    "input_outside_observed_range",
    "jev_unavailable_no_cache",
    "physics_margins_engine",
    "combiner_response_invalid",
)
TAU_GLOBAL = 0.60
TAU_MARGIN = 0.10

_LOG_QUEUE: queue.Queue = queue.Queue()
_LOG_LISTENER: logging.handlers.QueueListener | None = None


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def paths() -> dict[str, Path]:
    return {
        "dir": Path(ARTIFACT_DIR),
        "schema": Path(ARTIFACT_DIR) / "schema.json",
        "combiner": Path(ARTIFACT_DIR) / "combiner.joblib",
        "metrics": Path(ARTIFACT_DIR) / "metrics.json",
        "split": Path(ARTIFACT_DIR) / "split.json",
        "runs": ROOT / "data" / "runs.jsonl",
        "cases": ROOT / "data" / "cases.json",
        "cache": Path(CACHE_PATH),
        "validation": ROOT / "VALIDATION.md",
    }


def get_logger(name: str = "flameguard", level: int = logging.INFO) -> logging.Logger:
    """Records are queued and drained by a daemon thread; no call blocks on I/O."""
    global _LOG_LISTENER
    logger = logging.getLogger(name)
    if getattr(logger, "_fg_ready", False):
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if _LOG_LISTENER is None:
        _LOG_LISTENER = logging.handlers.QueueListener(_LOG_QUEUE, handler, respect_handler_level=True)
        _LOG_LISTENER.daemon = True
        _LOG_LISTENER.start()
    logger.addHandler(logging.handlers.QueueHandler(_LOG_QUEUE))
    logger.setLevel(level)
    logger.propagate = False
    logger._fg_ready = True
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, default=str, sort_keys=True))


# --------------------------------------------------------------------- JSON I/O


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    _atomic_write(target, json.dumps(data, indent=2, sort_keys=True) + "\n")
    return target


def canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------- cache
_CACHE_THREAD_LOCK = threading.Lock()


@contextmanager
def _file_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_THREAD_LOCK:
        with open(lock_path, "a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def cache_read(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or CACHE_PATH)
    empty = {"version": 1, "entries": {}}
    if not target.exists():
        return empty
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("entries"), dict):
        quarantine = target.with_name(f"{target.name}.corrupt.{int(time.time())}")
        try:
            target.replace(quarantine)
        except OSError:
            quarantine = None
        log_event(
            get_logger(),
            "cache_corrupt",
            path=str(target),
            quarantine=str(quarantine) if quarantine else None,
            level=logging.ERROR,
        )
        return empty
    return data


def cache_get(key: str, path: str | Path | None = None) -> dict[str, Any] | None:
    return cache_read(path)["entries"].get(key)


def cache_put(key: str, entry: dict[str, Any], path: str | Path | None = None) -> None:
    target = Path(path or CACHE_PATH)
    with _file_lock(target):
        data = cache_read(target)
        data["entries"][key] = entry
        _atomic_write(target, json.dumps(data, indent=2, sort_keys=True) + "\n")


def cache_size(path: str | Path | None = None) -> int:
    return len(cache_read(path)["entries"])


# --------------------------------------------------------------------- schema


def scenario_model(schema: dict[str, Any]) -> type[BaseModel]:
    """Build the request model from the observed data, so 422s carry field paths."""
    fields: dict[str, Any] = {}
    limits = {
        "thickness_mm": (0.0, 100.0),
        "o2_percent": (0.0, 100.0),
        "pressure_kpa": (0.0, 1000.0),
        "opposed_flow_cm_s": (0.0, 200.0),
    }
    for spec in schema["fields"]:
        name = spec["name"]
        if spec["kind"] == "numeric":
            low, high = limits.get(name, (0.0, 1e9))
            fields[name] = (
                float,
                Field(..., ge=low, le=high, allow_inf_nan=False, description=f"{name} ({spec.get('unit', '')})"),
            )
        else:
            categories = tuple(spec["categories"])
            fields[name] = (Literal[categories], Field(..., description="one of: " + ", ".join(categories)))
    return create_model("Scenario", __config__=ConfigDict(extra="forbid"), **fields)


def outside_observed_range(values: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    outside: list[str] = []
    for spec in schema["fields"]:
        if spec["kind"] != "numeric":
            continue
        value = values.get(spec["name"])
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number < spec["min"] or number > spec["max"]:
            outside.append(spec["name"])
    return outside


# ------------------------------------------------------------------ decisions


def abstain(
    probabilities: dict[str, float] | None,
    tau_global: float = TAU_GLOBAL,
    tau_margin: float = TAU_MARGIN,
) -> tuple[bool, str]:
    if not probabilities:
        return True, "jev_unavailable_no_cache"
    ordered = sorted((float(v) for v in probabilities.values()), reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    if top < tau_global:
        return True, "low_confidence"
    if (top - second) < tau_margin:
        return True, "ambiguous_top_classes"
    return False, ""


def suppression_for(
    verdict: str,
    o2_percent: float | None,
    pressure_kpa: float | None,
    flow_cm_s: float | None,
    severity: float | None,
    abstained: bool,
) -> tuple[str, str]:
    """Deterministic policy, not a learned classifier. Documented in README."""
    if abstained:
        return (
            "abort_configuration",
            "Verdict withheld: isolate or abort the configuration and escalate to a human fire-safety reviewer.",
        )
    if verdict == "extinction":
        return (
            "none_needed",
            "No active suppression required: maintain the inerting margin, monitor, and verify no re-ignition.",
        )
    o2 = float(o2_percent) if o2_percent is not None else 21.0
    pressure = float(pressure_kpa) if pressure_kpa is not None else 101.3
    flow = float(flow_cm_s) if flow_cm_s is not None else 0.0
    sev = float(severity) if severity is not None else 0.0
    if verdict == "marginal":
        if flow >= 15.0:
            return "flow_reduction", "Reduce forced convection / ventilation and re-assess the near-limit flame."
        if o2 >= 25.0:
            return "diluent_gas", "Dilute cabin oxygen with CO2/N2 below the limiting concentration."
        return "water_mist", "Apply water mist and boundary cooling; verify the flame has not anchored."
    if sev >= 3.0 or (o2 >= 30.0 and pressure >= 100.0):
        return "abort_configuration", "High-severity spread: isolate or abort the configuration and suppress remotely."
    if o2 >= 25.0:
        return "diluent_gas", "Inert the volume with CO2/N2 to below the limiting oxygen concentration."
    if flow >= 15.0:
        return "flow_reduction", "Reduce forced convection / ventilation to limit the opposed-flow supply."
    return "water_mist", "Apply water mist and boundary cooling, then verify spread has stopped."


def reason_codes(answers: dict[str, Any], verdict: str | None, abstain_reason: str = "") -> list[str]:
    codes: list[str] = []
    o2 = float(answers.get("o2_above_limiting", 0.5))
    flow = float(answers.get("flow_above_blowoff", 0.5))
    sustained = float(answers.get("self_sustaining_60s", 0.5))
    if o2 >= 0.60:
        codes.append("o2_above_limiting")
    elif o2 <= 0.40:
        codes.append("o2_below_limiting")
    else:
        codes.append("o2_near_limiting")
    if flow >= 0.60:
        codes.append("flow_above_blowoff")
    elif flow >= 0.40:
        codes.append("flow_near_blowoff")
    if verdict == "marginal" or 0.40 <= sustained <= 0.60:
        codes.append("near_limit_behavior")
    if abstain_reason:
        codes.append(abstain_reason)
    return list(dict.fromkeys(code for code in codes if code in REASON_CODES))