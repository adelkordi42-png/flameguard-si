"""Precompute the 6 canonical + 8 novel demo cases into engine_cache.json.

    python -m flameguard.precompute            # fetch missing, cache-first
    python -m flameguard.precompute --check    # verify every case is cached

The demo serves from this cache with JEV_MODE=cache, so it never depends on the
network. A miss is reported, never replaced by a fabricated answer.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from . import core, dataset, jev, llm


def demo_states() -> list[tuple[str, dict[str, Any]]]:
    runs = dataset.load_runs()
    cases = dataset.load_cases()
    states: list[tuple[str, dict[str, Any]]] = []
    for item in dataset.canonical_cases(runs, cases):
        states.append((item["run_id"], item["scenario"]))
    for item in dataset.novel_cases(cases):
        states.append((item["id"], item["scenario"]))
    return states


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flameguard.precompute")
    parser.add_argument("--check", action="store_true", help="only verify the cache")
    args = parser.parse_args(argv)
    logger = core.get_logger()
    if (os.environ.get("ENGINE_PROVIDER") or "").strip().lower() == "physics":
        print("physics margins engine: no engine cache needed; canonical cases are computed from margins")
        return 0
    engine = llm.build_engine()
    states = demo_states()
    missing: list[str] = []
    for name, state in states:
        key = engine.key_for(state)
        entry = core.cache_get(key, engine.cache_path)
        if entry and entry.get("questions_version") == jev.QUESTIONS_VERSION:
            continue
        if args.check:
            missing.append(name)
            continue
        try:
            engine.answer(state, allow_live=True)
        except jev.JevAuthError as exc:
            print(f"cannot precompute {name}: {exc}; export JEV_API_KEY")
            return 1
        except jev.JevError as exc:
            print(f"cannot precompute {name}: {exc}")
            return 1
        time.sleep(0.2)
    if missing:
        print(f"{len(missing)} demo cases not cached: {', '.join(missing)}")
        return 1
    print(f"{len(states)} demo cases cached")
    core.log_event(logger, "precompute_ok", cases=len(states), check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())