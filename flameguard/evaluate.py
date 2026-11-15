"""FlameGuard SI — fit the combiner and produce the honest validation report.

Protocol:
  1. Every run must have a cached Jev response (--fetch fills gaps, resumable).
  2. Canonical demo cases are forced into the held-out split.
  3. The logistic combiner + Platt calibration are fitted on 70% only.
  4. Tau is chosen on out-of-fold train probabilities; the test set is untouched.
  5. Accuracy, macro-F1, ECE, Brier, abstention and per-family/per-basis tables
     are written to artifacts/metrics.json and VALIDATION.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from . import combiner, core, dataset, features, jev, llm

TAU_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
REVIEW_BAND = (0.10, 0.30)
SEED = "flameguard-2026"


def state_of(run: dict[str, Any]) -> dict[str, Any]:
    return {name: run[name] for name in dataset.CONDITION_FIELDS}


def ensure_cache(runs: list[dict[str, Any]], engine: Any, fetch: bool) -> int:
    missing: list[str] = []
    fetched = 0
    for run in runs:
        key = engine.key_for(state_of(run))
        entry = core.cache_get(key, engine.cache_path)
        if entry and entry.get("questions_version") == jev.QUESTIONS_VERSION:
            continue
        if not fetch:
            missing.append(run["run_id"])
            continue
        try:
            engine.answer(state_of(run), allow_live=True)
        except jev.JevAuthError as exc:
            raise SystemExit(f"{exc}; export JEV_API_KEY and rerun with --fetch") from exc
        except jev.JevError as exc:
            raise SystemExit(f"engine fetch failed at {run['run_id']}: {exc}") from exc
        fetched += 1
        time.sleep(0.2)
    if missing:
        preview = ", ".join(missing[:8])
        raise SystemExit(f"{len(missing)} runs not cached ({preview}…); rerun with --fetch")
    return fetched


def load_vectors(
    runs: list[dict[str, Any]], engine: Any, no_engine: bool = False
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any] | None]]:
    vectors: list[list[float]] = []
    labels: list[str] = []
    answers: list[dict[str, Any] | None] = []
    for run in runs:
        state = state_of(run)
        margin_values = features.vector(state)
        if no_engine:
            parsed = None
        else:
            entry = core.cache_get(engine.key_for(state), engine.cache_path)
            if not entry:
                raise SystemExit(f"missing cache entry for {run['run_id']}")
            parsed = jev.parse_answers(entry["answers"])
        vectors.append(combiner.answers_to_vector(parsed, margin_values))
        labels.append(run["label"])
        answers.append(parsed)
    return np.asarray(vectors, dtype=float), np.asarray(labels), answers


def split_runs(
    runs: list[dict[str, Any]], canonical: list[str], test_fraction: float = 0.3
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run["label"]].append(run)
    canonical_set = set(canonical)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label, items in sorted(groups.items()):
        items = sorted(items, key=lambda r: hashlib.sha256((SEED + r["run_id"]).encode()).hexdigest())
        target = max(1, round(test_fraction * len(items)))
        chosen = [r for r in items if r["run_id"] in canonical_set]
        chosen_ids = {r["run_id"] for r in chosen}
        for run in items:
            if len(chosen) >= target:
                break
            if run["run_id"] not in chosen_ids:
                chosen.append(run)
                chosen_ids.add(run["run_id"])
        test.extend(chosen)
        train.extend(r for r in items if r["run_id"] not in chosen_ids)
    missing = canonical_set - {r["run_id"] for r in test}
    if missing:
        raise SystemExit(f"canonical cases did not land in test: {sorted(missing)}")
    for name in core.CLASSES:
        if not any(r["label"] == name for r in train):
            raise SystemExit(f"training split has no {name} rows")
    return train, sorted(test, key=lambda r: r["run_id"])


def tune_tau(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, Any]:
    class_array = np.asarray(classes)
    predicted = class_array[proba.argmax(axis=1)]
    ordered = np.sort(proba, axis=1)
    margin_ok = (ordered[:, -1] - ordered[:, -2]) >= core.TAU_MARGIN
    grid = []
    for tau in TAU_GRID:
        auto = (proba.max(axis=1) >= tau) & margin_ok
        coverage = float(auto.mean())
        accuracy = float((predicted[auto] == y[auto]).mean()) if auto.any() else 0.0
        grid.append(
            {
                "tau": tau,
                "coverage": coverage,
                "review_rate": 1.0 - coverage,
                "accuracy_auto": accuracy,
            }
        )
    in_band = [item for item in grid if REVIEW_BAND[0] <= item["review_rate"] <= REVIEW_BAND[1]]
    pool = in_band or [
        min(
            grid,
            key=lambda item: min(abs(item["review_rate"] - REVIEW_BAND[0]), abs(item["review_rate"] - REVIEW_BAND[1])),
        )
    ]
    chosen = max(pool, key=lambda item: (item["accuracy_auto"], -item["review_rate"]))
    return {
        "tau": chosen["tau"],
        "margin": core.TAU_MARGIN,
        "review_rate": chosen["review_rate"],
        "accuracy_auto": chosen["accuracy_auto"],
        "inside_target_band": bool(in_band),
        "grid": grid,
    }


def ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    confidence = proba.max(axis=1)
    correct = (np.asarray(core.CLASSES)[proba.argmax(axis=1)] == y).astype(float)
    order = np.argsort(confidence)
    total = len(y)
    value = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        value += len(chunk) / total * abs(correct[chunk].mean() - confidence[chunk].mean())
    return float(value)


def brier(y: np.ndarray, proba: np.ndarray) -> float:
    one_hot = np.zeros_like(proba)
    for index, name in enumerate(core.CLASSES):
        one_hot[:, index] = (y == name).astype(float)
    return float(np.mean((proba - one_hot) ** 2))


def reliability(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    confidence = proba.max(axis=1)
    correct = (np.asarray(core.CLASSES)[proba.argmax(axis=1)] == y).astype(float)
    order = np.argsort(confidence)
    rows = []
    for index, chunk in enumerate(np.array_split(order, n_bins)):
        if len(chunk) == 0:
            continue
        rows.append(
            {
                "bin": index + 1,
                "n": int(len(chunk)),
                "confidence": round(float(confidence[chunk].mean()), 4),
                "accuracy": round(float(correct[chunk].mean()), 4),
            }
        )
    return rows


def bootstrap(y: np.ndarray, predicted: np.ndarray, resamples: int = 1000) -> dict[str, list[float]]:
    rng = np.random.default_rng(42)
    accuracy: list[float] = []
    macro_f1: list[float] = []
    for _ in range(resamples):
        index = rng.integers(0, len(y), len(y))
        accuracy.append(float((predicted[index] == y[index]).mean()))
        macro_f1.append(
            float(f1_score(y[index], predicted[index], labels=list(core.CLASSES), average="macro", zero_division=0))
        )
    return {
        "accuracy": [round(float(np.percentile(accuracy, 2.5)), 4), round(float(np.percentile(accuracy, 97.5)), 4)],
        "macro_f1": [round(float(np.percentile(macro_f1, 2.5)), 4), round(float(np.percentile(macro_f1, 97.5)), 4)],
    }


def group_metrics(
    runs: list[dict[str, Any]],
    y: np.ndarray,
    predicted: np.ndarray,
    abstained: np.ndarray,
    key: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, run in enumerate(runs):
        groups[str(run[key])].append(index)
    rows = []
    for name, indices in sorted(groups.items()):
        idx = np.asarray(indices)
        present = sorted(set(y[idx].tolist()))
        rows.append(
            {
                "name": name,
                "n": int(len(idx)),
                "accuracy": round(float((predicted[idx] == y[idx]).mean()), 4),
                "macro_f1": round(
                    float(f1_score(y[idx], predicted[idx], labels=present, average="macro", zero_division=0)),
                    4,
                ),
                "classes_present": present,
                "abstention_rate": round(float(abstained[idx].mean()), 4),
                "insufficient_n": bool(len(idx) < 5),
            }
        )
    return rows


def build_metrics(
    runs: list[dict[str, Any]],
    report: dict[str, Any],
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    tau: dict[str, Any],
    answers_test: list[dict[str, Any] | None],
    engine: Any,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    predicted = np.asarray(core.CLASSES)[proba_test.argmax(axis=1)]
    abstained = np.asarray(
        [core.abstain(dict(zip(core.CLASSES, row)), tau["tau"], tau["margin"])[0] for row in proba_test]
    )
    accuracy = float((predicted == y_test).mean())
    macro = float(f1_score(y_test, predicted, labels=list(core.CLASSES), average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=list(core.CLASSES), zero_division=0
    )
    per_class = {
        name: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, name in enumerate(core.CLASSES)
    }
    majority = Counter(y_train.tolist()).most_common(1)[0][0]
    majority_pred = np.full(len(y_test), majority)
    if answers_test and answers_test[0] is not None:
        engine_choice = np.asarray(
            [answers["flame_class"]["choice"] for answers in answers_test if answers is not None]
        )
        engine_choice_accuracy = round(float((engine_choice == y_test).mean()), 4)
        engine_choice_macro_f1 = round(
            float(f1_score(y_test, engine_choice, labels=list(core.CLASSES), average="macro", zero_division=0)),
            4,
        )
    else:
        engine_choice = None
        engine_choice_accuracy = None
        engine_choice_macro_f1 = None
    selective = predicted[~abstained]
    selective_truth = y_test[~abstained]
    cache_models: Counter = Counter()
    if engine is not None:
        cache_models = Counter(
            entry.get("model")
            for entry in (
                core.cache_get(engine.key_for(state_of(run)), engine.cache_path)
                for run in runs
            )
            if entry
        )
    misclassified = []
    for index in np.where(predicted != y_test)[0][:10]:
        run = test[int(index)]
        misclassified.append(
            {
                "run_id": run["run_id"],
                "family": run["family"],
                "actual": str(y_test[index]),
                "predicted": str(predicted[index]),
                "confidence": round(float(proba_test[index].max()), 4),
                "label_basis": run["label_basis"],
                "source": run["source"],
            }
        )
    return {
        "generated_at": core.iso_now(),
        "schema_version": core.SCHEMA_VERSION,
        "data_version": report["data_version"],
        "n_runs": report["n_runs"],
        "gate": report["gate"],
        "reduced_claims": report["reduced_claims"],
        "provenance_warning": report["provenance_warning"],
        "tier_a": report["tier_a"],
        "derived_share": report["derived_share"],
        "label_counts": report["label_counts"],
        "basis_counts": report["basis_counts"],
        "families": report["families"],
        "state_conflicts": report["state_conflict_count"],
        "split": {
            "seed": SEED,
            "train": len(train),
            "test": len(test),
            "train_counts": dict(sorted(Counter(y_train.tolist()).items())),
            "test_counts": dict(sorted(Counter(y_test.tolist()).items())),
            "canonical_in_test": True,
        },
        "model": {
            "name": "multinomial LogisticRegression + sigmoid calibration",
            "features": list(combiner.FEATURE_ORDER),
            "n_train": len(train),
            "classes": list(core.CLASSES),
            "tau_global": tau["tau"],
            "tau_margin": tau["margin"],
            "engine_kind": type(engine).__name__ if engine is not None else "PhysicsMargins",
            "engine_model": engine.model if engine is not None else "published-threshold margins",
            "engine_endpoint": getattr(engine, "base", jev.ENDPOINT) if engine is not None else "none",
            "engine_models": dict(cache_models),
            "cache_entries": core.cache_size(engine.cache_path) if engine is not None else 0,
            "engine_provider": getattr(engine, "provider", "physics") if engine is not None else "physics",
            "engine_label": (
                jev.engine_label(getattr(engine, "provider", "physics"))
                if engine is not None
                else "published-threshold margins"
            ),
            "feature_mode": bundle.get("feature_mode", "engine+margins"),
        },
        "accuracy": round(accuracy, 4),
        "accuracy_ci": bootstrap(y_test, predicted)["accuracy"],
        "macro_f1": round(macro, 4),
        "macro_f1_ci": bootstrap(y_test, predicted)["macro_f1"],
        "ece": round(ece(y_test, proba_test), 4),
        "brier": round(brier(y_test, proba_test), 4),
        "abstention_rate": round(float(abstained.mean()), 4),
        "coverage": round(float((~abstained).mean()), 4),
        "selective_accuracy": round(float((selective == selective_truth).mean()), 4) if len(selective) else None,
        "review_count": int(abstained.sum()),
        "per_class": per_class,
        "confusion": confusion_matrix(y_test, predicted, labels=list(core.CLASSES)).tolist(),
        "reliability": reliability(y_test, proba_test),
        "abstention": tau,
        "per_family": group_metrics(test, y_test, predicted, abstained, "family"),
        "per_basis": group_metrics(test, y_test, predicted, abstained, "label_basis"),
        "baseline": {
            "majority_class": majority,
            "majority_accuracy": round(float((majority_pred == y_test).mean()), 4),
            "majority_macro_f1": round(
                float(f1_score(y_test, majority_pred, labels=list(core.CLASSES), average="macro", zero_division=0)),
                4,
            ),
            "engine_choice_accuracy": engine_choice_accuracy,
            "engine_choice_macro_f1": engine_choice_macro_f1,
        },
        "misclassified": misclassified,
        "reproduce": (
            "python -m flameguard.evaluate --no-engine"
            if bundle.get("feature_mode") == "margins_only"
            else "python -m flameguard.evaluate --no-fetch"
        ),
    }


def render_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# FlameGuard SI — validation",
        "",
        f"Generated **{metrics['generated_at']}** · schema `{metrics['schema_version']}` · data `{metrics['data_version']}`",
        f"Engine **{metrics['model'].get('engine_label', metrics['model'].get('engine_kind', 'unknown'))}** · model `{metrics['model'].get('engine_model', 'unknown')}` · feature mode `{metrics['model'].get('feature_mode', 'engine+margins')}` · endpoint `{metrics['model'].get('engine_endpoint', 'unknown')}`",
        "",
        "> These metrics are measured on the held-out 30% only. Calibration is fitted on the",
        "> 70% combiner-fit split; it is not inherited from the decision model. Published and derived labels",
        "> are reported separately because FLEX/BASS numeric outcomes need a label policy.",
        "",
        "## Gate and provenance",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| labelled runs | **{metrics['n_runs']}** (gate {metrics['gate']}) |",
        f"| reduced_claims | **{metrics['reduced_claims']}** |",
        f"| provenance_warning | **{metrics['provenance_warning']}** |",
        f"| published / derived | {metrics['tier_a']} / {metrics['basis_counts'].get('derived', 0)} |",
        f"| duplicate-state conflicts | {metrics['state_conflicts']} |",
        "",
        "## Split",
        "",
        f"70/30 stratified by label; the six canonical cases are forced into test.",
        "",
        "| split | n | class counts |",
        "| --- | --- | --- |",
        f"| train | {metrics['split']['train']} | {metrics['split']['train_counts']} |",
        f"| test | {metrics['split']['test']} | {metrics['split']['test_counts']} |",
        "",
        "## Held-out headline",
        "",
        "| metric | value | 95% CI |",
        "| --- | --- | --- |",
        f"| accuracy | **{metrics['accuracy']:.3f}** | {metrics['accuracy_ci']} |",
        f"| macro-F1 | **{metrics['macro_f1']:.3f}** | {metrics['macro_f1_ci']} |",
        f"| ECE (10 bins) | **{metrics['ece']:.4f}** | — |",
        f"| Brier (macro) | **{metrics['brier']:.4f}** | — |",
        f"| abstention rate | **{metrics['abstention_rate']:.3f}** | — |",
        f"| selective accuracy | **{metrics['selective_accuracy'] if metrics['selective_accuracy'] is not None else 'n/a'}** | — |",
        "",
        "| baseline | accuracy | macro-F1 |",
        "| --- | --- | --- |",
        f"| majority ({metrics['baseline']['majority_class']}) | {metrics['baseline']['majority_accuracy']:.3f} | {metrics['baseline']['majority_macro_f1']:.3f} |",
        f"| engine choice alone | {metrics['baseline']['engine_choice_accuracy'] if metrics['baseline']['engine_choice_accuracy'] is not None else 'n/a'} | {metrics['baseline']['engine_choice_macro_f1'] if metrics['baseline']['engine_choice_macro_f1'] is not None else 'n/a'} |",
        "",
        "## Per class (held-out)",
        "",
        "| class | precision | recall | F1 | support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in core.CLASSES:
        row = metrics["per_class"][name]
        lines.append(f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |")
    lines += [
        "",
        "## Confusion matrix (rows = actual, columns = predicted)",
        "",
        "| actual \\ predicted | " + " | ".join(core.CLASSES) + " |",
        "| --- | " + " | ".join("---" for _ in core.CLASSES) + " |",
    ]
    for index, name in enumerate(core.CLASSES):
        lines.append("| " + name + " | " + " | ".join(str(v) for v in metrics["confusion"][index]) + " |")
    lines += [
        "",
        "## Calibration (held-out)",
        "",
        f"- ECE: **{metrics['ece']:.4f}** · Brier: **{metrics['brier']:.4f}**",
        "",
        "| bin | n | mean confidence | accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for row in metrics["reliability"]:
        lines.append(f"| {row['bin']} | {row['n']} | {row['confidence']:.3f} | {row['accuracy']:.3f} |")
    lines += [
        "",
        "## Abstention (tau chosen on out-of-fold train only)",
        "",
        f"- tau = **{metrics['abstention']['tau']:.2f}**, margin = **{metrics['abstention']['margin']:.2f}**",
        f"- train review rate = {metrics['abstention']['review_rate']:.3f}, train selective accuracy = {metrics['abstention']['accuracy_auto']:.3f}",
        "",
        "| tau | coverage | review rate | accuracy on auto-clear |",
        "| --- | --- | --- | --- |",
    ]
    for row in metrics["abstention"]["grid"]:
        marker = " **(chosen)**" if row["tau"] == metrics["abstention"]["tau"] else ""
        lines.append(
            f"| {row['tau']:.2f} | {row['coverage']:.3f} | {row['review_rate']:.3f} | {row['accuracy_auto']:.3f}{marker} |"
        )
    lines += [
        "",
        "## Per family (held-out)",
        "",
        "| family | n | accuracy | macro-F1 | abstention | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in metrics["per_family"]:
        note = "insufficient n" if row["insufficient_n"] else ""
        lines.append(
            f"| {row['name']} | {row['n']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | {row['abstention_rate']:.3f} | {note} |"
        )
    lines += [
        "",
        "Macro-F1 in this table is computed over the classes present in the family, not a fixed 3-class macro.",
        "",
        "## Published vs derived labels (held-out)",
        "",
        "| basis | n | accuracy | macro-F1 | abstention |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in metrics["per_basis"]:
        lines.append(
            f"| {row['name']} | {row['n']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | {row['abstention_rate']:.3f} |"
        )
    lines += [
        "",
        "## Misclassified examples",
        "",
    ]
    if not metrics["misclassified"]:
        lines.append("None in this split.")
    else:
        lines += [
            "| run | family | actual | predicted | confidence | basis | source |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in metrics["misclassified"]:
            source = row["source"]
            lines.append(
                f"| {row['run_id']} | {row['family']} | {row['actual']} | {row['predicted']} | "
                f"{row['confidence']:.3f} | {row['label_basis']} | {source['doc_id']} table {source['table']} |"
            )
    lines += [
        "",
        "## Limitations",
        "",
        "- The 167 rows are a convenience extraction, not a random sample of all microgravity combustion data.",
        "- 143 of 167 labels are derived from published numeric outcomes (FLEX burn time, BASS-II spread rate, Saffire burn length); only 24 are explicitly categorical. The published/derived table above is the honest split.",
        "- The engine contract includes a microgravity reference with the published thresholds, and the four margin features are computed from the same policy. Derived-label metrics therefore measure policy agreement, not independent prediction; the 24 published labels are the independent check.",
        "- FLEX `Extinction` is the normal end of a droplet burn; it is not a failed ignition. The burn-time rule is documented in `data/LABEL_POLICY.md`.",
        "- SoFIE GEL rows are extinction-point conditions, not initial conditions; they are limit tests.",
        "- The held-out set is small (about 50 rows), so the confidence intervals are wide.",
        "- The decision model is frozen; its own confidence is not calibrated for this task. The combiner is calibrated on the 70% split.",
        "- Suppression strategies are a deterministic policy from published fire-safety guidance, not a learned classifier, and are not validated here.",
        "- Nothing here is a flight-qualified procedure.",
        "",
        "## Reproduce",
        "",
        "```bash",
        metrics["reproduce"],
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flameguard.evaluate")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fetch", dest="fetch", action="store_true", help="fetch missing Jev responses (resumable)")
    group.add_argument("--no-fetch", dest="fetch", action="store_false", help="fail on a cache miss instead of calling Jev")
    parser.set_defaults(fetch=False)
    parser.add_argument(
        "--no-engine",
        action="store_true",
        help="fit/evaluate on published-threshold margins only; make no engine calls",
    )
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args(argv)

    logger = core.get_logger()
    runs = dataset.load_runs(args.data)
    report = dataset.audit(runs)
    schema_path = core.paths()["schema"]
    if not schema_path.exists():
        core.write_json(schema_path, dataset.build_schema(runs, report))
    engine = None if args.no_engine else llm.build_engine()
    fetched = 0 if engine is None else ensure_cache(runs, engine, args.fetch)
    if fetched:
        core.log_event(logger, "evaluate_fetched", count=fetched)

    x_all, y_all, answers_all = load_vectors(runs, engine, no_engine=args.no_engine)
    canonical = dataset.load_cases()["canonical"]
    train, test = split_runs(runs, canonical)
    train_ids = {run["run_id"] for run in train}
    test_ids = {run["run_id"] for run in test}
    by_id = {run["run_id"]: index for index, run in enumerate(runs)}
    train_index = np.asarray([by_id[run["run_id"]] for run in train])
    test_index = np.asarray([by_id[run["run_id"]] for run in test])
    x_train, y_train = x_all[train_index], y_all[train_index]
    x_test, y_test = x_all[test_index], y_all[test_index]
    answers_test = [answers_all[i] for i in test_index]

    bundle = combiner.fit(x_train.tolist(), y_train.tolist())
    oof, oof_classes = combiner.oof_probabilities(x_train.tolist(), y_train.tolist())
    tau = tune_tau(y_train, oof, oof_classes)
    bundle.update(
        {
            "tau_global": tau["tau"],
            "tau_margin": tau["margin"],
            "jev_model": engine.model if engine is not None else "published-threshold margins",
            "feature_mode": "margins_only" if engine is None else "engine+margins",
        }
    )
    combiner.save(bundle)

    predictions = [combiner.predict(bundle, vector.tolist()) for vector in x_test]
    proba_test = np.asarray([[item["probabilities"][name] for name in core.CLASSES] for item in predictions])
    metrics = build_metrics(runs, report, train, test, y_train, y_test, proba_test, tau, answers_test, engine, bundle)
    core.write_json(core.paths()["metrics"], metrics)
    core.write_json(
        core.paths()["split"],
        {
            "seed": SEED,
            "train": sorted(train_ids),
            "test": sorted(test_ids),
            "canonical": canonical,
            "train_counts": metrics["split"]["train_counts"],
            "test_counts": metrics["split"]["test_counts"],
        },
    )
    core.paths()["validation"].write_text(render_markdown(metrics), encoding="utf-8")
    core.log_event(
        logger,
        "evaluate_done",
        n=metrics["n_runs"],
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        ece=metrics["ece"],
        brier=metrics["brier"],
        abstention=metrics["abstention_rate"],
        reduced=metrics["reduced_claims"],
    )
    print(
        f"n={metrics['n_runs']} test={len(test)} accuracy={metrics['accuracy']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f} ece={metrics['ece']:.4f} brier={metrics['brier']:.4f} "
        f"abstention={metrics['abstention_rate']:.3f} tau={tau['tau']:.2f}"
    )
    if metrics["provenance_warning"]:
        print(f"PROVENANCE: {metrics['tier_a']} published / {metrics['n_runs']} total — see VALIDATION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())