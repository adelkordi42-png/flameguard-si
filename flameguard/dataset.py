"""FlameGuard SI — labelled-run ingestion and audit.

`data/runs.jsonl` is the only source of truth for the model and the UI. This
module validates every row, computes the gate and provenance counts, and writes
the request schema consumed by the API. It never invents a row.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import core

CONDITION_FIELDS = (
    "material",
    "geometry",
    "thickness_mm",
    "o2_percent",
    "pressure_kpa",
    "opposed_flow_cm_s",
    "igniter",
    "gravity",
)
FIELD_ORDER = (
    ("material", "categorical", ""),
    ("geometry", "categorical", ""),
    ("thickness_mm", "numeric", "mm"),
    ("o2_percent", "numeric", "%"),
    ("pressure_kpa", "numeric", "kPa"),
    ("opposed_flow_cm_s", "numeric", "cm/s"),
    ("igniter", "categorical", ""),
    ("gravity", "categorical", ""),
)
GATE = 150


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    table: str
    url: str
    page: int | None = None


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=3)
    family: str = Field(min_length=1)
    source: Source
    material: str = Field(min_length=1)
    geometry: str = Field(min_length=1)
    thickness_mm: float = Field(gt=0.0, le=100.0)
    o2_percent: float = Field(ge=0.0, le=100.0)
    pressure_kpa: float = Field(gt=0.0, le=1000.0)
    opposed_flow_cm_s: float = Field(ge=0.0, le=200.0)
    igniter: str = Field(min_length=1)
    gravity: str = Field(min_length=1)
    outcome_raw: str = Field(min_length=1)
    outcome_quote: str = Field(min_length=1)
    label: Literal["sustained_spread", "marginal", "extinction"]
    label_basis: Literal["published", "derived"]
    notes: str = ""


class DataError(RuntimeError):
    pass


def load_runs(path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(path or core.paths()["runs"])
    if not target.exists():
        raise DataError(f"no labelled runs at {target}")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(Run.model_validate(json.loads(line)).model_dump())
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"line {number}: {exc}")
    if errors:
        raise DataError("invalid runs.jsonl:\n" + "\n".join(errors[:10]))
    if not rows:
        raise DataError(f"no rows parsed from {target}")
    ids = [row["run_id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted(k for k, v in Counter(ids).items() if v > 1)
        raise DataError(f"duplicate run_id values: {duplicates}")
    return rows


def conditions(run: dict[str, Any]) -> dict[str, Any]:
    return {name: run[name] for name in CONDITION_FIELDS}


def condition_hash(run: dict[str, Any]) -> str:
    return core.canonical_hash(conditions(run))


def audit(runs: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(run["label"] for run in runs)
    basis = Counter(run["label_basis"] for run in runs)
    families = Counter(run["family"] for run in runs)
    sources = Counter(run["source"]["doc_id"] for run in runs)
    by_state: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        by_state[condition_hash(run)].add(run["label"])
    conflicts = [
        {"state_hash": key[:12], "labels": sorted(value), "rows": sum(1 for r in runs if condition_hash(r) == key)}
        for key, value in by_state.items()
        if len(value) > 1
    ]
    complete = len(runs)
    tier_a = basis.get("published", 0)
    derived_share = basis.get("derived", 0) / complete if complete else 0.0
    return {
        "n_runs": complete,
        "n_complete": complete,
        "gate": GATE,
        "reduced_claims": complete < GATE,
        "provenance_warning": tier_a < 50 or derived_share > 0.5,
        "tier_a": tier_a,
        "derived_share": round(derived_share, 4),
        "label_counts": dict(sorted(labels.items())),
        "basis_counts": dict(sorted(basis.items())),
        "families": dict(sorted(families.items())),
        "sources": dict(sorted(sources.items())),
        "state_conflict_count": len(conflicts),
        "state_conflicts": conflicts[:5],
        "data_version": core.canonical_hash(
            [(run["run_id"], run["label"], run["label_basis"]) for run in sorted(runs, key=lambda r: r["run_id"])]
        )[:16],
    }


def build_schema(runs: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for name, kind, unit in FIELD_ORDER:
        values = [run[name] for run in runs]
        spec: dict[str, Any] = {"name": name, "kind": kind, "unit": unit}
        if kind == "numeric":
            numbers = [float(value) for value in values]
            spec.update(
                min=round(min(numbers), 4),
                max=round(max(numbers), 4),
                mean=round(sum(numbers) / len(numbers), 4),
            )
        else:
            spec["categories"] = sorted({str(value) for value in values})
        fields.append(spec)
    return {
        "schema_version": core.SCHEMA_VERSION,
        "generated_at": core.iso_now(),
        "n_runs": report["n_runs"],
        "gate": report["gate"],
        "reduced_claims": report["reduced_claims"],
        "provenance_warning": report["provenance_warning"],
        "tier_a": report["tier_a"],
        "derived_share": report["derived_share"],
        "label_counts": report["label_counts"],
        "basis_counts": report["basis_counts"],
        "families": report["families"],
        "data_version": report["data_version"],
        "fields": fields,
    }


def load_schema() -> dict[str, Any]:
    schema = core.read_json(core.paths()["schema"])
    if not isinstance(schema, dict) or not isinstance(schema.get("fields"), list):
        raise DataError(f"missing or invalid schema at {core.paths()['schema']}; run: python -m flameguard.dataset")
    return schema


def load_cases(path: str | Path | None = None) -> dict[str, Any]:
    cases = core.read_json(path or core.paths()["cases"])
    if not isinstance(cases, dict) or not isinstance(cases.get("canonical"), list) or not isinstance(cases.get("novel"), list):
        raise DataError(f"missing or invalid cases file at {path or core.paths()['cases']}")
    return cases


def canonical_cases(runs: list[dict[str, Any]], cases: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {run["run_id"]: run for run in runs}
    out = []
    for run_id in cases["canonical"]:
        run = by_id.get(run_id)
        if run is None:
            raise DataError(f"canonical case {run_id!r} absent from runs.jsonl")
        out.append(
            {
                "run_id": run_id,
                "family": run["family"],
                "scenario": conditions(run),
                "actual_label": run["label"],
                "label_basis": run["label_basis"],
                "outcome_raw": run["outcome_raw"],
                "source": run["source"],
            }
        )
    if len(out) < 6:
        raise DataError(f"need 6 canonical cases, found {len(out)}")
    return out


def novel_cases(cases: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in cases["novel"]:
        if not isinstance(item, dict) or "id" not in item or "scenario" not in item:
            raise DataError("each novel case needs id and scenario")
        out.append({"id": item["id"], "title": item.get("title", item["id"]), "scenario": item["scenario"]})
    if len(out) < 8:
        raise DataError(f"need 8 novel cases, found {len(out)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flameguard.dataset")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--audit", action="store_true", help="print the audit (default action)")
    args = parser.parse_args(argv)

    logger = core.get_logger()
    try:
        runs = load_runs(args.data)
        report = audit(runs)
        schema = build_schema(runs, report)
        target = args.schema or core.paths()["schema"]
        core.write_json(target, schema)
        core.log_event(logger, "data_audit", **{k: report[k] for k in ("n_complete", "gate", "reduced_claims", "label_counts", "basis_counts")})
    except DataError as exc:
        print(f"data audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"schema: {target}")
    if report["reduced_claims"]:
        print(f"WARNING: {report['n_complete']} < {report['gate']}; reduced_claims=true")
    if report["provenance_warning"]:
        print(f"WARNING: provenance — {report['tier_a']} published / {report['n_complete']} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())