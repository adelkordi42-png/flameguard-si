"""FlameGuard SI — calibrated logistic combiner over Jev sub-answers.

Jev returns six typed signals; this module turns them into the fixed 8-vector
and fits a small multinomial logistic regression with Platt calibration. The
combiner is the only object fitted on the labelled runs. Jev itself is frozen.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import core

FEATURE_ORDER = (
    "o2_margin",
    "flow_fraction",
    "near_limit",
    "thickness_regime",
    "o2_above_limiting",
    "flow_above_blowoff",
    "self_sustaining_60s",
    "spread_beyond_ignition",
    "extinction",
    "marginal",
    "sustained_spread",
    "severity",
)


def answers_to_vector(answers: dict[str, Any] | None, margin_values: list[float] | None = None) -> list[float]:
    """Margins first, then the eight engine signals. `answers=None` zeroes the
    model block so the margin-only path can run without any engine."""
    computed = [float(value) for value in (margin_values or [0.0, 0.0, 0.0, 0.0])]
    if answers is None:
        model = [0.0] * 8
    else:
        choice = answers["flame_class"]["probabilities"]
        model = [
            float(answers["o2_above_limiting"]),
            float(answers["flow_above_blowoff"]),
            float(answers["self_sustaining_60s"]),
            float(answers["spread_beyond_ignition"]),
            float(choice["extinction"]),
            float(choice["marginal"]),
            float(choice["sustained_spread"]),
            float(answers["severity"]["score"]) / 3.0,
        ]
    vector = computed + model
    if len(vector) != len(FEATURE_ORDER):
        raise ValueError("feature vector length drifted")
    return vector


def _pipeline(n_min: int) -> CalibratedClassifierCV:
    folds = max(2, min(5, n_min, 5))
    estimator = LogisticRegression(max_iter=2000, C=1.0)
    return CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
    )


def fit(vectors: list[list[float]], labels: list[str]) -> dict[str, Any]:
    counts = Counter(labels)
    missing = [name for name in core.CLASSES if counts.get(name, 0) == 0]
    if missing:
        raise ValueError(f"cannot fit combiner: classes absent from training split: {missing}")
    n_min = min(counts[name] for name in core.CLASSES)
    if n_min < 2:
        raise ValueError(f"cannot calibrate: smallest class has {n_min} rows")
    pipeline = _pipeline(n_min)
    pipeline.fit(np.asarray(vectors, dtype=float), np.asarray(labels))
    return {
        "pipeline": pipeline,
        "feature_order": list(FEATURE_ORDER),
        "classes": [str(c) for c in pipeline.classes_],
        "n_train": len(labels),
        "class_counts": dict(sorted(counts.items())),
        "schema_version": core.SCHEMA_VERSION,
        "trained_at": core.iso_now(),
    }


def oof_probabilities(vectors: list[list[float]], labels: list[str]) -> tuple[np.ndarray, list[str]]:
    counts = Counter(labels)
    n_min = min(counts[name] for name in core.CLASSES)
    pipeline = _pipeline(n_min)
    proba = cross_val_predict(
        pipeline,
        np.asarray(vectors, dtype=float),
        np.asarray(labels),
        cv=StratifiedKFold(n_splits=max(2, min(5, n_min)), shuffle=True, random_state=42),
        method="predict_proba",
    )
    classes = sorted(set(labels))
    return np.asarray(proba, dtype=float), [str(c) for c in classes]


def predict(bundle: dict[str, Any], vector: list[float]) -> dict[str, Any]:
    pipeline = bundle["pipeline"]
    scores = pipeline.predict_proba(np.asarray([vector], dtype=float))[0]
    classes = [str(c) for c in pipeline.classes_]
    probabilities = {name: float(scores[classes.index(name)]) if name in classes else 0.0 for name in core.CLASSES}
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("combiner returned an all-zero distribution")
    probabilities = {name: value / total for name, value in probabilities.items()}
    verdict = max(probabilities, key=probabilities.get)
    return {
        "verdict": verdict,
        "confidence": probabilities[verdict],
        "probabilities": probabilities,
    }


def save(bundle: dict[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path or core.paths()["combiner"])
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=target.name)
    os.close(fd)
    try:
        joblib.dump(bundle, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def load(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or core.paths()["combiner"])
    if not target.exists():
        raise FileNotFoundError(f"no combiner at {target}; run: python -m flameguard.evaluate")
    bundle = joblib.load(target)
    if bundle.get("schema_version") != core.SCHEMA_VERSION:
        raise ValueError(
            f"combiner schema {bundle.get('schema_version')!r} != {core.SCHEMA_VERSION!r}; refit required"
        )
    if list(bundle.get("feature_order", [])) != list(FEATURE_ORDER):
        raise ValueError("combiner feature order drifted; refit required")
    return bundle