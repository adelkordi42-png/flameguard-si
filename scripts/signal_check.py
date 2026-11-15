#!/usr/bin/env python3
"""10-row signal gate for the selected engine.

Runs 4 sustained / 3 marginal / 3 extinction rows through whatever
`llm.build_engine()` selects (OpenJev, Jev-compatible host, or the LLM wrapper)
and exits 1 unless at least 6/10 choices match the label.

    python scripts/signal_check.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flameguard import dataset, jev, llm


def main() -> int:
    runs = dataset.load_runs()
    by_label: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_label[run["label"]].append(run)
    sample = []
    for label, count in (("sustained_spread", 4), ("marginal", 3), ("extinction", 3)):
        items = sorted(by_label[label], key=lambda run: run["run_id"])
        step = max(1, len(items) // count)
        sample.extend(items[::step][:count])

    engine = llm.build_engine()
    provider = getattr(engine, "provider", "llm")
    base = getattr(engine, "base", jev.BASE_URL)
    source = jev.key_source() if isinstance(engine, jev.JevEngine) else "LLM_API_KEY / BEATAPI_API_KEY"
    print(
        f"engine={type(engine).__name__} provider={provider} model={getattr(engine, 'model', '?')} "
        f"base={base} key_source={source or 'MISSING'}"
    )
    correct = 0
    for run in sample:
        state = {name: run[name] for name in dataset.CONDITION_FIELDS}
        try:
            response = engine.answer(state, allow_live=True)
        except jev.JevAuthError as exc:
            print(f"auth error for {run['run_id']}: {exc}")
            print(f"key/base mismatch: base={base} key_source={source or 'MISSING'}; set the key for this provider")
            return 1
        except jev.JevError as exc:
            print(f"engine error for {run['run_id']}: {exc}")
            return 1
        choice = response.answers["flame_class"]["choice"]
        probabilities = response.answers["flame_class"]["probabilities"]
        ok = choice == run["label"]
        correct += int(ok)
        print(
            f"{run['run_id']:26s} actual={run['label']:16s} predicted={choice:16s} "
            f"p_true={probabilities.get(run['label'], 0):.3f} "
            f"conf={response.answers['flame_class']['confidence']:.3f} "
            f"o2={response.answers['o2_above_limiting']:.2f} "
            f"flow={response.answers['flow_above_blowoff']:.2f} "
            f"sev={response.answers['severity']['score']:.2f} {'OK' if ok else 'MISS'}"
        )
    print(f"\ncorrect {correct}/{len(sample)}")
    return 0 if correct >= 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())