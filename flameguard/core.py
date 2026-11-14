"""FlameGuard SI — shared core.

Schema discovery, CSV I/O, feature encoding, abstention rule, suppression
lookup and nearest-neighbour retrieval. Imported by both the training
pipeline (``flameguard.train``) and the inference service (``flameguard.api``)
so that the two can never disagree about the schema.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import logging.handlers
import queue
import re
import sys
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"

CLASSES = ("sustained_spread", "marginal", "extinction")
VERDICT_REVIEW = "needs_human_review"
EXTINCTION_MARKERS = (
    "extinct",
    "extinguish",
    "quench",
    "no spread",
    "did not spread",
    "not spread",
    "no flame",
    "did not ignite",
    "no ignition",
    "failed to ignite",
    "stopped",
    "nonflammable",
    "non flammable",
    "self extinguish",
)
TARGET_NAMES = ("class", "outcome", "label", "verdict", "flame_behavior", "behavior", "result")
ID_TOKENS = {"id", "run", "test", "source", "ref", "cite", "note", "notes", "comment", "comments"}
O2_BANDS = ((0.16, "below_limiting"), (0.21, "normoxic"), (0.30, "elevated"), (2.0, "high"))
PRESSURE_BANDS = ((50.0, "sub_atmospheric"), (80.0, "reduced"), (110.0, "standard"), (1e9, "high"))
TAU_GLOBAL = 0.60
TAU_MARGIN = 0.10
K_NEIGHBOURS = 5

_LOG_QUEUE: queue.Queue = queue.Queue()
_LOG_LISTENER: logging.handlers.QueueListener | None = None


def get_logger(name: str = "flameguard", level: int = logging.INFO) -> logging.Logger:
    """Non-blocking logger: records go to a queue drained by a daemon thread."""
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


# --------------------------------------------------------------------- ingest


def is_identifier(column: str) -> bool:
    """Token match, so 'fluid'/'solid'/'oxidizer' are not mistaken for 'id'."""
    return any(token in ID_TOKENS for token in re.split(r"[^a-z0-9]+", column.lower()) if token)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"cannot decode {path} as utf-8, cp1252 or latin-1")
    rows = [
        {k.strip(): (v.strip() if isinstance(v, str) else "") for k, v in record.items() if k}
        for record in csv.DictReader(io.StringIO(text))
    ]
    if not rows:
        raise ValueError(f"no data rows parsed from {path}")
    return rows


def detect_target(columns: list[str], explicit: str | None, logger: logging.Logger) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"target column {explicit!r} not in {columns}")
        return explicit
    lowered = {c.lower(): c for c in columns}
    for candidate in TARGET_NAMES:
        if candidate in lowered:
            return lowered[candidate]
    log_event(logger, "target_autodetected", column=columns[-1], level=logging.WARNING)
    return columns[-1]


def normalize_class(raw: str) -> str | None:
    """Negation-aware mapping, so 'did not spread' is not read as spread."""
    key = " ".join(raw.strip().lower().replace("_", " ").replace("-", " ").split())
    if not key:
        return None
    if "margin" in key:
        return "marginal"
    if any(marker in key for marker in EXTINCTION_MARKERS):
        return "extinction"
    if "sustain" in key or "spread" in key:
        return "sustained_spread"
    return None


def infer_meta(rows: list[dict[str, str]], target: str, logger: logging.Logger) -> dict[str, Any]:
    """Discover the schema from the data itself; never assume column names."""
    columns = list(rows[0].keys())
    features: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for name in columns:
        if name == target:
            continue
        if is_identifier(name):
            skipped.append(name)
            continue
        values = [r.get(name, "") for r in rows]
        present = [v for v in values if v != ""]
        if not present:
            skipped.append(name)
            continue
        numbers: list[float] = []
        failed = 0
        for value in present:
            try:
                number = float(value)
            except ValueError:
                failed += 1
                continue
            if np.isfinite(number):
                numbers.append(number)
            else:
                failed += 1
        if numbers and failed <= 0.2 * len(present):
            mean = float(np.mean(numbers))
            std = float(np.std(numbers))
            features[name] = {
                "kind": "numeric",
                "min": float(min(numbers)),
                "max": float(max(numbers)),
                "mean": mean if np.isfinite(mean) else 0.0,
                "std": std if np.isfinite(std) and std > 0.0 else 1.0,
            }
        else:
            features[name] = {"kind": "categorical", "categories": sorted(set(present))}
    if not features:
        raise ValueError("no feature columns discovered after excluding the target column")
    labels = [normalize_class(r.get(target, "")) for r in rows]
    unknown = sorted({r.get(target, "") for r, lab in zip(rows, labels) if lab is None and r.get(target, "")})
    if unknown:
        log_event(logger, "unmapped_labels", labels=unknown[:10], count=len(unknown), level=logging.WARNING)
    classes = sorted({lab for lab in labels if lab})
    log_event(
        logger,
        "schema_inferred",
        target=target,
        features={n: s["kind"] for n, s in features.items()},
        skipped_columns=skipped,
        classes=classes,
        rows=len(rows),
    )
    return {"target": target, "features": features, "classes": classes, "skipped_columns": skipped}


def labelled_rows(rows: list[dict[str, str]], target: str) -> list[dict[str, str]]:
    return [r for r in rows if normalize_class(r.get(target, "")) is not None]


# -------------------------------------------------------------------- encoding


def label_vector(rows: list[dict[str, str]], meta: dict[str, Any]) -> np.ndarray:
    target = meta["target"]
    return np.array([normalize_class(r.get(target, "")) for r in rows], dtype=object)


def encode_model(rows: list[dict[str, str]], meta: dict[str, Any]) -> np.ndarray:
    """Numeric matrix for the tree model: numeric stays raw, categorical becomes a code."""
    features = list(meta["features"].items())
    matrix = np.full((len(rows), len(features)), np.nan, dtype=float)
    for j, (name, spec) in enumerate(features):
        for i, row in enumerate(rows):
            value = row.get(name, "")
            if spec["kind"] == "numeric":
                try:
                    matrix[i, j] = float(value)
                except (TypeError, ValueError):
                    matrix[i, j] = np.nan
            elif value in spec["categories"]:
                matrix[i, j] = spec["categories"].index(value)
    return matrix


def categorical_mask(meta: dict[str, Any]) -> np.ndarray:
    return np.array([spec["kind"] == "categorical" for spec in meta["features"].values()], dtype=bool)


def encode_knn(matrix: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Standardised numeric block + one-hot categorical block, for distances."""
    features = list(meta["features"].items())
    blocks: list[np.ndarray] = []
    numeric_idx = [j for j, (_, s) in enumerate(features) if s["kind"] == "numeric"]
    if numeric_idx:
        means = np.array([features[j][1]["mean"] for j in numeric_idx])
        stds = np.array([features[j][1]["std"] or 1.0 for j in numeric_idx])
        values = matrix[:, numeric_idx]
        values = np.where(np.isnan(values), means, values)
        blocks.append((values - means) / stds)
    for j, (_, spec) in enumerate(features):
        if spec["kind"] == "categorical":
            onehot = np.zeros((matrix.shape[0], len(spec["categories"])))
            column = matrix[:, j]
            for code in range(len(spec["categories"])):
                onehot[:, code] = (column == code).astype(float)
            blocks.append(onehot)
    if not blocks:
        raise ValueError("cannot build a distance matrix with no numeric or categorical features")
    return np.hstack(blocks)


# ------------------------------------------------------------------ decisions


def abstain(probabilities: np.ndarray, tau_global: float = TAU_GLOBAL, tau_margin: float = TAU_MARGIN) -> tuple[bool, str]:
    ordered = np.sort(np.asarray(probabilities, dtype=float))[::-1]
    top = float(ordered[0]) if ordered.size else 0.0
    second = float(ordered[1]) if ordered.size > 1 else 0.0
    if top < tau_global:
        return True, "low_confidence"
    if (top - second) < tau_margin:
        return True, "ambiguous_top_classes"
    return False, ""


def outside_observed_range(values: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    """Numeric inputs beyond the training range are out of distribution."""
    outside: list[str] = []
    for name, spec in meta["features"].items():
        if spec["kind"] != "numeric":
            continue
        try:
            value = float(values.get(name))
        except (TypeError, ValueError):
            continue
        if value < spec["min"] or value > spec["max"]:
            outside.append(name)
    return outside


def o2_fraction(value: float | None) -> float | None:
    """Normalise O2 to a fraction: datasets report either 0.21 or 21.0."""
    if value is None:
        return None
    return value / 100.0 if value > 1.5 else value


def pressure_kpa(value: float | None) -> float | None:
    """Normalise pressure to kPa: datasets report either kPa (~101) or atm (~1.0)."""
    if value is None:
        return None
    return value * 101.325 if value < 5.0 else value


def band(value: float, bands: tuple[tuple[float, str], ...]) -> str:
    for upper, name in bands:
        if value <= upper:
            return name
    return bands[-1][1]


def suppression_for(
    verdict: str,
    o2_fraction_value: float | None = None,
    pressure_kpa_value: float | None = None,
    confined: bool | None = None,
) -> str:
    """Deterministic mapping: (verdict, O2 band, pressure band) -> one strategy.

    Rule-based from published NASA microgravity fire-safety guidance. This is
    deliberately NOT learned from data, and is labelled as such in the UI.
    """
    if verdict == VERDICT_REVIEW:
        return "Abstained: escalate to a human fire-safety reviewer before acting."
    o2_band = band(o2_fraction_value, O2_BANDS) if o2_fraction_value is not None else "unknown"
    pressure_band = band(pressure_kpa_value, PRESSURE_BANDS) if pressure_kpa_value is not None else "unknown"
    place = "confined volume" if confined else "open volume" if confined is not None else "volume"
    if verdict == "extinction":
        return (
            f"No active suppression required in the {place}: maintain inerting margin and monitor "
            f"(O2 {o2_band}, pressure {pressure_band})."
        )
    if verdict == "marginal":
        return (
            f"Pre-emptive action: reduce O2 toward the limiting concentration and lower opposed-flow "
            f"velocity below the blow-off threshold, then monitor continuously ({place}, O2 {o2_band})."
        )
    if o2_band == "high":
        return (
            f"Immediate inerting with CO2/N2 to below the limiting O2 concentration and depressurise "
            f"the {place} ({pressure_band}); continuous agent until temperature falls."
        )
    if o2_band == "elevated":
        return f"Fine water mist plus reduced O2 concentration in the {place}; boundary cooling."
    return (
        f"Water mist and boundary cooling in the {place}, with O2 reduction and verification that "
        f"spread has stopped."
    )


def nearest_runs(
    query: np.ndarray,
    train_matrix: np.ndarray,
    train_rows: list[dict[str, str]],
    train_labels: np.ndarray,
    meta: dict[str, Any],
    k: int = K_NEIGHBOURS,
) -> list[dict[str, Any]]:
    distances = np.linalg.norm(train_matrix - query, axis=1)
    order = np.argsort(distances, kind="stable")[:k]
    target = meta["target"]
    feature_names = set(meta["features"])
    return [
        {
            "distance": round(float(distances[i]), 4),
            "actual_label": str(train_labels[i]),
            "record": {key: value for key, value in train_rows[i].items() if key in feature_names},
            "actual_raw": train_rows[i].get(target, ""),
        }
        for i in order
    ]


# ------------------------------------------------------------------ artifacts


def paths(directory: Path | str = ARTIFACT_DIR) -> dict[str, Path]:
    directory = Path(directory)
    return {
        "dir": directory,
        "bundle": directory / "model.joblib",
        "meta": directory / "features.json",
        "metrics": directory / "metrics.json",
        "validation": directory / "validation.md",
    }


def save_bundle(bundle: dict[str, Any], directory: Path | str = ARTIFACT_DIR) -> Path:
    target = paths(directory)
    target["dir"].mkdir(parents=True, exist_ok=True)
    staging = target["bundle"].with_suffix(".joblib.tmp")
    joblib.dump(bundle, staging)
    staging.replace(target["bundle"])
    return target["bundle"]


def load_bundle(directory: Path | str = ARTIFACT_DIR) -> dict[str, Any]:
    target = paths(directory)
    if not target["bundle"].exists():
        raise FileNotFoundError(f"no trained model at {target['bundle']}; run: python -m flameguard.train")
    return joblib.load(target["bundle"])


def scenario_model(meta: dict[str, Any]) -> type[BaseModel]:
    """Build the request schema from features.json so validation tracks the data."""
    fields: dict[str, Any] = {}
    for name, spec in meta["features"].items():
        if spec["kind"] == "numeric":
            fields[name] = (float, Field(..., allow_inf_nan=False, description=f"numeric {name}"))
        else:
            fields[name] = (
                Literal[tuple(spec["categories"])],
                Field(..., description="one of: " + ", ".join(spec["categories"])),
            )
    return create_model("Scenario", __config__=ConfigDict(extra="forbid"), **fields)