"""FlameGuard SI — training pipeline.

    python -m flameguard.train [--data PATH] [--target COLUMN] [--seed N]

Discovers the organizer CSV under data/raw/, cross-validates, decides whether
calibration earns its place, tunes the abstention threshold, fits the final
model and writes artifacts/{model.joblib,features.json,metrics.json,validation.md}.

If no CSV is present a clearly-labelled SYNTHETIC placeholder is generated so
the pipeline stays exercisable end to end; its numbers are never presented as
NASA validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import core

SEED = 7
TAU_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
REVIEW_BAND = (0.10, 0.30)
MIN_ROWS = 12

MATERIALS = ("pmma", "pe_insulation", "cotton", "silicone", "nomex")
MATERIAL_FLAMMABILITY = {"pmma": 2.0, "pe_insulation": 2.5, "cotton": 1.5, "silicone": 1.0, "nomex": 0.2}


def make_estimator(mask: np.ndarray, seed: int = SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=False,
        categorical_features=mask,
        random_state=seed,
    )


def synthesise(n: int, seed: int, logger: logging.Logger) -> list[dict[str, str]]:
    """Placeholder data with physics-plausible structure. NOT NASA measurements."""
    rng = random.Random(seed)
    drafts = []
    for _ in range(n):
        material = rng.choice(MATERIALS)
        o2 = round(rng.uniform(14.0, 30.0), 1)
        pressure = round(rng.uniform(50.0, 101.0), 1)
        velocity = round(rng.uniform(0.0, 25.0), 1)
        geometry = rng.choice(("open", "confined"))
        index = (
            1.0 * (o2 - 17.0)
            - 0.45 * velocity
            + 1.0 * (pressure - 50.0) / 50.0
            + MATERIAL_FLAMMABILITY[material]
            + (1.0 if geometry == "confined" else 0.0)
            + rng.gauss(0.0, 1.5)
        )
        drafts.append((index, material, o2, pressure, velocity, geometry))
    lower, upper = np.quantile([draft[0] for draft in drafts], [0.38, 0.70])
    rows = []
    for index, material, o2, pressure, velocity, geometry in drafts:
        label = "sustained_spread" if index > upper else "marginal" if index > lower else "extinction"
        rows.append(
            {
                "material": material,
                "o2_percent": str(o2),
                "pressure_kpa": str(pressure),
                "opposed_flow_cm_s": str(velocity),
                "geometry": geometry,
                "flame_behavior": label,
            }
        )
    core.log_event(
        logger,
        "synthetic_dataset",
        rows=len(rows),
        seed=seed,
        class_counts={c: sum(1 for r in rows if r["flame_behavior"] == c) for c in core.CLASSES},
        warning="placeholder data, not NASA measurements",
    )
    return rows


def oof_probabilities(
    factory: Callable[[], object],
    x: np.ndarray,
    y: np.ndarray,
    splitter,
    classes: np.ndarray,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    proba = np.zeros((len(y), len(classes)))
    split_args = (x, y, groups) if groups is not None else (x, y)
    for train_idx, test_idx in splitter.split(*split_args):
        model = factory().fit(x[train_idx], y[train_idx])
        fold = model.predict_proba(x[test_idx])
        index = {label: position for position, label in enumerate(model.classes_)}
        for column, label in enumerate(classes):
            if label in index:
                proba[test_idx, column] = fold[:, index[label]]
    if not np.allclose(proba.sum(axis=1), 1.0):
        raise ValueError("out-of-fold probabilities are incomplete; check the split or grouping")
    return proba


def brier_macro(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    return float(np.mean([brier_score_loss((y == c).astype(int), proba[:, i]) for i, c in enumerate(classes)]))


def tune_tau(y: np.ndarray, proba: np.ndarray, classes: np.ndarray, logger: logging.Logger) -> dict:
    predictions = classes[proba.argmax(axis=1)]
    candidates = []
    for tau in TAU_GRID:
        flags = np.array([core.abstain(proba[i], tau, core.TAU_MARGIN)[0] for i in range(len(y))])
        auto = ~flags
        coverage = float(auto.mean())
        accuracy = float((predictions[auto] == y[auto]).mean()) if auto.any() else 0.0
        candidates.append({"tau": tau, "coverage": coverage, "review_rate": 1 - coverage, "accuracy_auto": accuracy})
    in_band = [c for c in candidates if REVIEW_BAND[0] <= c["review_rate"] <= REVIEW_BAND[1]]
    if in_band:
        pool = in_band
    else:
        nearest_edge = min(candidates, key=lambda c: min(abs(c["review_rate"] - edge) for edge in REVIEW_BAND))
        pool = [nearest_edge]
        core.log_event(
            logger,
            "tau_outside_target_band",
            review_rate=round(nearest_edge["review_rate"], 4),
            band=list(REVIEW_BAND),
            level=logging.ERROR,
        )
    chosen = max(pool, key=lambda c: (c["accuracy_auto"], -c["review_rate"]))
    core.log_event(
        logger,
        "tau_tuned",
        tau=chosen["tau"],
        review_rate=round(chosen["review_rate"], 4),
        accuracy_auto_clear=round(chosen["accuracy_auto"], 4),
        inside_target_band=bool(in_band),
        candidates=[{k: round(v, 4) if isinstance(v, float) else v for k, v in c.items()} for c in candidates],
    )
    return {**chosen, "inside_target_band": bool(in_band), "grid": candidates}


def find_group_column(rows: list[dict[str, str]], meta: dict, n_splits: int) -> str | None:
    """An identifier column whose values repeat signals multiple runs of one experiment."""
    for name in meta["skipped_columns"]:
        values = [r.get(name, "") for r in rows]
        present = [v for v in values if v != ""]
        if len(present) < 0.5 * len(values):
            continue
        distinct = set(present)
        if len(distinct) >= max(2, n_splits) and len(distinct) < len(values):
            return name
    return None


def write_validation_md(path: Path, metrics: dict, meta: dict) -> None:
    lines = [
        "# FlameGuard SI — validation table",
        "",
        f"- **Data source:** `{metrics['data_source']}` — {metrics['provenance']}",
        f"- **Rows:** {metrics['n_rows']} · **Classes:** {', '.join(metrics['classes'])}",
        f"- **Model:** {metrics['model']} · **Seed:** {metrics['seed']}",
        f"- **Split:** {metrics['split']}",
        "",
        "## Held-out cross-validated performance",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in metrics["classes"]:
        row = metrics["per_class"][name]
        lines.append(
            f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {row['support']} |"
        )
    lines += [
        f"| **macro avg** | {metrics['per_class']['macro avg']['precision']:.3f} | "
        f"{metrics['per_class']['macro avg']['recall']:.3f} | {metrics['per_class']['macro avg']['f1-score']:.3f} | "
        f"{metrics['n_rows']} |",
        "",
        f"- Accuracy (argmax, no abstention): **{metrics['accuracy']:.3f}**",
        f"- Macro F1: **{metrics['macro_f1']:.3f}** vs majority-class baseline **{metrics['baseline_macro_f1']:.3f}**",
        "",
        "## Confusion matrix (rows = actual, columns = predicted)",
        "",
        "| actual \\ predicted | " + " | ".join(metrics["classes"]) + " |",
        "| --- | " + " | ".join("---" for _ in metrics["classes"]) + " |",
    ]
    for i, name in enumerate(metrics["classes"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in metrics["confusion"][i]) + " |")
    lines += [
        "",
        "## Calibration",
        "",
        f"- Brier (macro, uncalibrated): **{metrics['brier_uncalibrated']:.4f}**",
        f"- Brier (macro, calibrated): **{metrics['brier_calibrated']:.4f}**",
        f"- Decision: **{'calibration kept' if metrics['calibration_kept'] else 'calibration DROPPED (no Brier gain)'}**",
        "",
        "## Abstention trade-off",
        "",
        "Selection rule: highest accuracy on the auto-clear subset among thresholds whose review",
        "rate falls in the 10–30% band, ties broken toward fewer deferrals.",
        "",
        "| tau | auto-clear coverage | review rate | accuracy on auto-clear |",
        "| --- | --- | --- | --- |",
    ]
    for cand in metrics["abstention"]["grid"]:
        marker = " **(chosen)**" if cand["tau"] == metrics["abstention"]["tau"] else ""
        lines.append(
            f"| {cand['tau']:.2f} | {cand['coverage']:.3f} | {cand['review_rate']:.3f} | "
            f"{cand['accuracy_auto']:.3f}{marker} |"
        )
    lines += [
        "",
        f"Chosen tau = **{metrics['abstention']['tau']:.2f}** (margin {core.TAU_MARGIN:.2f}) · "
        f"review rate {metrics['abstention']['review_rate']:.3f} · "
        f"accuracy on auto-clear {metrics['abstention']['accuracy_auto']:.3f} "
        f"({metrics['abstention']['review_rate'] * metrics['n_rows']:.0f} of {metrics['n_rows']} runs deferred to a human).",
        "",
        "The suppression strategy table is rule-based from published NASA fire-safety guidance and is not learned from data.",
        "",
    ]
    if metrics["data_source"] != "provided":
        lines += [
            "> **PROVENANCE WARNING:** these numbers come from SYNTHETIC placeholder data,",
            "> not from NASA experimental runs. Drop the organizer CSV into `data/raw/` and",
            "> re-run `python -m flameguard.train` to produce real validation numbers.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flameguard.train")
    parser.add_argument("--data", type=Path, default=None, help="CSV path (default: data/raw/*.csv)")
    parser.add_argument("--target", default=None, help="target column name")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--synthetic-rows", type=int, default=180)
    args = parser.parse_args(argv)
    if args.synthetic_rows < MIN_ROWS:
        raise SystemExit(f"--synthetic-rows must be at least {MIN_ROWS}")

    logger = core.get_logger()
    core.log_event(logger, "train_start", data=str(args.data) if args.data else "auto")

    csv_path = args.data
    if csv_path is None:
        found = sorted((core.ROOT / "data" / "raw").glob("*.csv"))
        csv_path = found[0] if found else None
        if len(found) > 1:
            core.log_event(logger, "multiple_csv_found", using=str(csv_path), all=[str(p) for p in found], level=logging.WARNING)

    if csv_path is not None:
        try:
            rows = core.read_rows(csv_path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read dataset {csv_path}: {exc}") from exc
        data_source, provenance = "provided", f"organizer CSV `{csv_path.name}`"
        fingerprint = hashlib.sha1(Path(csv_path).read_bytes()).hexdigest()[:12]
    else:
        rows = synthesise(args.synthetic_rows, args.seed, logger)
        data_source = "synthetic"
        provenance = "synthetic placeholder generated by flameguard.train (no CSV found in data/raw/)"
        fingerprint = f"synthetic-{args.seed}"

    target = core.detect_target(list(rows[0].keys()), args.target, logger)
    meta = core.infer_meta(rows, target, logger)
    meta["data_source"] = data_source
    meta["model_version"] = f"{data_source}-{fingerprint}"

    usable = core.labelled_rows(rows, target)
    if len(usable) < MIN_ROWS:
        unmapped = sorted({r.get(target, "") for r in rows if core.normalize_class(r.get(target, "")) is None})
        raise SystemExit(
            f"only {len(usable)} labelled rows parsed from column {target!r} (need {MIN_ROWS}); "
            f"unmapped values: {unmapped[:12]}"
        )
    y = core.label_vector(usable, meta)
    classes = np.array(sorted(set(y)))
    if len(classes) < 2:
        raise SystemExit(f"need at least 2 outcome classes, found {classes.tolist()}")
    if len(classes) < len(core.CLASSES):
        core.log_event(logger, "reduced_class_set", classes=classes.tolist(), expected=list(core.CLASSES), level=logging.WARNING)
    counts = {c: int((y == c).sum()) for c in classes}
    core.log_event(logger, "class_counts", counts=counts, rows=len(usable))
    if min(counts.values()) < 10:
        core.log_event(logger, "small_class_warning", minimum=min(counts.values()), level=logging.WARNING)

    x_model = core.encode_model(usable, meta)
    x_knn = core.encode_knn(x_model, meta)
    mask = core.categorical_mask(meta)

    n_splits = max(2, min(5, min(counts.values())))
    group_column = find_group_column(usable, meta, n_splits)
    groups = None
    if group_column:
        groups = np.array([r.get(group_column, "") for r in usable])
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        split_name = f"StratifiedGroupKFold(k={n_splits}) grouped by `{group_column}`"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        split_name = f"StratifiedKFold(k={n_splits})"
    core.log_event(logger, "splitter_selected", split=split_name, group_column=group_column)

    proba_uncal = oof_probabilities(lambda: make_estimator(mask, args.seed), x_model, y, splitter, classes, groups)
    inner_cv = max(2, min(3, min(counts.values())))
    calibrated_factory = lambda: CalibratedClassifierCV(make_estimator(mask, args.seed), method="sigmoid", cv=inner_cv)
    proba_cal = oof_probabilities(calibrated_factory, x_model, y, splitter, classes, groups)

    brier_uncal = brier_macro(y, proba_uncal, classes)
    brier_cal = brier_macro(y, proba_cal, classes)
    calibration_kept = brier_cal < brier_uncal
    chosen_proba = proba_cal if calibration_kept else proba_uncal
    core.log_event(
        logger,
        "calibration_decision",
        brier_uncalibrated=round(brier_uncal, 4),
        brier_calibrated=round(brier_cal, 4),
        kept=calibration_kept,
    )

    abstention = tune_tau(y, chosen_proba, classes, logger)
    predictions = classes[chosen_proba.argmax(axis=1)]

    dummy = DummyClassifier(strategy="most_frequent").fit(x_model, y)
    baseline_macro_f1 = float(f1_score(y, dummy.predict(x_model), average="macro", zero_division=0))
    report = classification_report(y, predictions, labels=list(classes), zero_division=0, output_dict=True)
    matrix = confusion_matrix(y, predictions, labels=list(classes))

    final_model = calibrated_factory() if calibration_kept else make_estimator(mask, args.seed)
    final_model.fit(x_model, y)
    core.log_event(logger, "final_model_fitted", calibrated=calibration_kept, estimator=type(final_model).__name__)

    metrics = {
        "data_source": data_source,
        "provenance": provenance,
        "model_version": meta["model_version"],
        "model": type(final_model).__name__,
        "calibrated": calibration_kept,
        "seed": args.seed,
        "split": split_name,
        "group_column": group_column,
        "n_rows": len(usable),
        "classes": list(classes),
        "class_counts": counts,
        "features": {n: s["kind"] for n, s in meta["features"].items()},
        "accuracy": float((predictions == y).mean()),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "baseline_macro_f1": baseline_macro_f1,
        "per_class": {k: report[k] for k in list(classes) + ["macro avg"]},
        "confusion": matrix.tolist(),
        "brier_uncalibrated": brier_uncal,
        "brier_calibrated": brier_cal,
        "calibration_kept": calibration_kept,
        "abstention": abstention,
        "tau_global": abstention["tau"],
        "tau_margin": core.TAU_MARGIN,
        "k_neighbours": core.K_NEIGHBOURS,
    }

    target_paths = core.paths()
    core.save_bundle(
        {
            "model": final_model,
            "meta": meta,
            "x_knn": x_knn,
            "rows": usable,
            "labels": y,
            "classes": classes,
            "tau_global": abstention["tau"],
            "tau_margin": core.TAU_MARGIN,
            "model_version": meta["model_version"],
        }
    )
    target_paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    target_paths["metrics"].write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_validation_md(target_paths["validation"], metrics, meta)
    core.log_event(
        logger,
        "train_complete",
        artifacts=[str(p) for p in (target_paths["bundle"], target_paths["meta"], target_paths["metrics"], target_paths["validation"])],
        accuracy=round(metrics["accuracy"], 4),
        macro_f1=round(metrics["macro_f1"], 4),
    )
    print(f"accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f} "
          f"tau={abstention['tau']:.2f} review_rate={abstention['review_rate']:.3f} "
          f"calibrated={calibration_kept} source={data_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())