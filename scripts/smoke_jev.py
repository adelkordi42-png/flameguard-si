#!/usr/bin/env python3
"""One real Jev call: proves the key, base URL, model id, response parsing and cache."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flameguard import core, dataset, jev


def main() -> int:
    runs = dataset.load_runs()
    scenario = dataset.canonical_cases(runs, dataset.load_cases())[0]["scenario"]
    engine = jev.JevEngine()
    print(f"endpoint={jev.ENDPOINT}")
    print(f"model={engine.model} mode={jev.mode()} key={'set' if engine.key else 'MISSING'} timeout={jev.TIMEOUT_S:g}s")
    if not engine.key:
        print("JEV_API_KEY (or TYPESAFE_API_KEY / BEATAPI_API_KEY) is not set", file=sys.stderr)
        return 1
    try:
        response = engine.answer(scenario, allow_live=True)
    except jev.JevError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"engine={response.engine} model={response.model} latency_ms={response.latency_ms:.1f}")
    for name, value in sorted(response.signals().items()):
        print(f"  {name}: {value:.4f}")
    print(f"cached entries: {core.cache_size()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())