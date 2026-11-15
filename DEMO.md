# Judge demo — 60 seconds

**Precondition:** the shipped artifacts are committed; no key is needed.

```bash
python -m flameguard.evaluate --no-engine   # margins-only fit + VALIDATION.md (already run)
python -m flameguard.precompute --check     # physics mode: nothing to cache
python -m pytest -q                          # 19 tests
```

Start the demo (offline, zero keys, zero tokens):

```bash
ENGINE_PROVIDER=physics JEV_MODE=cache uvicorn flameguard.api:app --port 7860
```

Warm the page and the health endpoint before the judge arrives:

```bash
curl -s localhost:7860/api/health
curl -s localhost:7860/api/cases | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["canonical"]), len(d["novel"]))'
```

Expect `{"status":"ok","engine_provider":"physics",...}` and `6 8`.

> Keyed engine instead: set `JEV_API_BASE`/`JEV_MODEL`/`JEV_API_KEY` (or
> `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY`), run `python scripts/signal_check.py`;
> only if it prints 6/10 or better, run `evaluate --fetch` and `precompute`.

---

## Beat 1 — problem (0–10 s)

> “A fire in freefall has no truck bay and no escape. NASA has decades of
> microgravity combustion data, but the data says what happened in a test — not
> what to do about the cabin in front of you.”

Point at the header: **167 runs · 24 published / 143 derived** and the provenance
warning. Say the warning is intentional: the report separates published labels
from labels derived by a locked policy.

## Beat 2 — one call (10–25 s)

Press the first canonical preset / leave the default. Point at the verdict card.

> “One request carries the scenario and six typed Jev questions: four threshold
> questions, one three-way class question, one severity score. A calibrated
> logistic layer combines the six answers. The card shows the class, the
> calibrated probability, the typed reason codes, and one of five suppression
> strategies.”

Read the card; do not read a number from this script. The card is generated from
the committed cache.

## Beat 3 — predicted vs actual (25–40 s)

Point at section 3.

> “These six are held out, not training rows. Each row is a real NASA table with
> its source. The table shows predicted versus actual, the probability, whether it
> matched, and whether the label was published or derived.”

If one is wrong, say so. The match column is the honest scorecard.

## Beat 4 — the differentiator (40–50 s)

Set **Material: PMMA**, **O2: 99**, **Pressure: 101.3**, **Opposed flow: 10**,
**Thickness: 1**, then Assess. Or pick an uncached novel scenario with the key
removed.

> “This input is outside the observed envelope. The system refuses to answer:
> `needs_human_review`, reason `input_outside_observed_range`, strategy
> `abort_configuration`. A dashboard that always answers is a dashboard you stop
> trusting.”

Mention the other triggers: low calibrated probability, top-two classes too close,
and a cache miss with no network (`jev_unavailable_no_cache`).

## Beat 5 — validation and limits (50–60 s)

Open `/metrics`.

> “Accuracy, macro-F1, ECE, Brier, abstention, per-family and published-vs-derived
> tables are generated from the held-out 30% only. Calibration is fitted and
> validated on NASA-derived runs; it is not inherited from Jev. The suppression
> strategy is a rule-based policy, not a learned classifier. This is not a
> flight-qualified procedure.”

Stop. Do not claim a number that is not on the page.

---

## If something breaks

| symptom | cause | recovery |
| --- | --- | --- |
| canonical table is empty | `combiner.joblib` missing | run `evaluate --no-engine`; do not fabricate a verdict |
| verdict is `needs_human_review` | the abstention rule fired | read the reason; that is the product working |
| `/metrics` says “metrics unavailable” | `evaluate` not run | run `evaluate --no-engine`; the page reads `artifacts/metrics.json` |
| first Render request hangs | free-tier cold start | warm `/api/health` before the demo |
| page is blank | JS error | open the console; the page has no external dependencies |

## Numbers to have ready

Only from `artifacts/metrics.json` / `/metrics`:

- held-out accuracy and 95% CI
- macro-F1 and 95% CI
- ECE and Brier
- abstention rate and selective accuracy
- published vs derived counts (24 / 143)
- full-data gate (167 ≥ 150)

Never quote a number from this file; it is a script, not a measurement.