# FlameGuard SI — PROJECT_MAP

Calibrated flammability decision layer for spacecraft cabins. Jev (TypeSafe
System One) supplies six typed sub-answers; a small calibrated logistic combiner
turns them into a three-class verdict, typed reason codes, and a suppression
strategy. A cache-first service keeps the demo offline-safe.

**Build date:** 2026-11-15 · **Repo:** `github.com/adelkordi42-png/flameguard-si`
(public) · **Runtime:** Python 3.14.4, CPU only.

## [TECH_STACK]

Pinned in `requirements.txt`; all wheels verified installable on cp314 with
`pip check` clean. `requirements-dev.txt` adds test and build-only packages.

| Component | Version | Why this and not more |
| --- | --- | --- |
| Python | 3.14.4 | System interpreter is PEP 668-locked; venv built at `~/.venvs/flameguard` on the Linux filesystem (importing scikit-learn from `/mnt/c` is slow) |
| fastapi | 0.163.0 | Free 422 with field paths, lifespan, FileResponse; no extra HTTP framework |
| uvicorn | 0.54.0 | ASGI server, no `[standard]` extras |
| pydantic | 2.14.3 | Request model generated from `artifacts/schema.json` at startup |
| numpy | 2.5.4 | Arrays for the combiner and metrics; pandas deliberately excluded |
| scikit-learn | 1.9.1 | `LogisticRegression`, `CalibratedClassifierCV`, metrics; no LightGBM/XGBoost |
| joblib | 1.6.0 | `combiner.joblib` atomic serialization |
| httpx2 / pytest | 2.16.0 / 9.2.0 | dev only — Starlette 1.10 TestClient needs httpx2 |
| pdfplumber | 0.11.10 | dev/build only — parses FLEX Table IX from the NTRS PDF; not in the runtime image |
| Frontend | — | Two static HTML files, vanilla JS/CSS, no Node, no npm, no build step |

## [SYSTEM_FLOW]

Build time:
```
NTRS .txt/.pdf tables
  ├─ BASS-II 5.1 (20) · BASS SIBAL 7.1 (31)
  ├─ Saffire I/II I (11) · SoFIE GEL 1 (5)
  └─ FLEX IX (276 parsed -> 100 stratified)
             |
             v
data/runs.jsonl (167 rows, source quote, label, label_basis)
             |
             +--> python -m flameguard.dataset --audit
             |      artifacts/schema.json · gate · provenance · state conflicts
             |
             +--> python -m flameguard.evaluate --fetch
             |      1 engine request/state → engine_cache.json
             |      70/30 split (canonical in test) → LR+Platt → combiner.joblib
             |      artifacts/metrics.json · VALIDATION.md · split.json
             |
             +--> python -m flameguard.precompute
                    6 canonical + 8 novel cases → engine_cache.json
```

Demo time:
```
GET /                        → static/index.html
  ├─ GET /api/meta           → fields, classes, strategies, gate, provenance
  ├─ GET /api/cases          → 6 canonical + 8 novel, computed from cache
  └─ POST /api/predict       → schema validation → cache hit → combiner
                                  │ miss
                                  ├─ JEV_MODE=auto/live + key + probe → Jev
                                  └─ otherwise 200 needs_human_review
GET /metrics                 → static/metrics.html → /api/metrics
GET /VALIDATION.md           → generated report
GET /deck                    → five-slide deck, fills from /api/metrics
```

Every stage above is on the judge path. There is no database, queue, auth, or
frontend framework.

## [ARCHITECTURE]

| File | Role | Measured lines |
| --- | --- | --- |
| `flameguard/core.py` | Constants, non-blocking QueueHandler logger, atomic JSON/cache I/O with `fcntl` lock, bands, abstention, suppression policy, dynamic request model | 320 |
| `flameguard/jev.py` | Six-question contract, state rendering, stdlib HTTP client, retries, response validation, cache-first engine | 411 |
| `flameguard/dataset.py` | `runs.jsonl` validation, provenance audit, gate, schema generation, canonical/novel case loading | 251 |
| `flameguard/combiner.py` | 8-feature extraction, multinomial LR + sigmoid calibration, atomic joblib persistence, OOF probabilities | 140 |
| `flameguard/evaluate.py` | Cache fetch, canonical-forced split, OOF tau, held-out metrics/ECE/Brier/CI, per-family/basis, markdown report | 585 |
| `flameguard/api.py` | Lifespan, health/meta/cases/predict/metrics, static mount, `/deck`, typed abstention on every failure | 294 |
| `flameguard/precompute.py` | Cache the 14 demo cases; `--check` fails on a miss | 63 |
| `static/index.html` | Form → verdict card → canonical table → 8 novel presets | 294 |
| `static/metrics.html` | `/metrics` page, no hardcoded numbers | 118 |
| `docs/deck.html` | Five-slide deck served at `/deck`; fills its numbers from `/api/metrics` | 132 |
| `tests/test_smoke.py` | Data gate, schema 422, cache corruption, parser, combiner, suppression, API offline/422/OOD, no-secret | 263 |
| `scripts/build_runs.py` | Build-time extraction/regeneration; not imported by the service | 407 |

Total measured: **3,279 lines**. The line counts are evidence, not targets; no
function is reachable only from a test.

Key decisions:

- **Endpoint correction.** `POST /v1/jev:run` returns 404. The live OpenAPI 0.2.0
  exposes `POST /v1/systemone`; the client uses that. `docs/typesafe-openapi.json`
  is the committed evidence.
- **Gateway-compatible.** `JEV_API_BASE` points the same client at any host that
  exposes the native `/v1/systemone` schema. Verified against published docs:
  Vercel AI Gateway (`https://ai-gateway.vercel.sh/typesafe`, model
  `typesafe-ai/jev`, Bearer `AI_GATEWAY_API_KEY`) and BeatAPI
  (`https://api.beatapi.io`, models `jev-1.14`/`jev-1.13`). Only the base URL,
  model id and key change; the cache key still covers every provider.
- **Six small questions, not one big one.** Four `noul` thresholds, one `choice`,
  one `score`, one request. The combiner consumes the fixed 8-vector.
- **Cache is a first-class path.** `engine_cache.json` is keyed by SHA-256 of
  schema + model + state + question contract; writes are locked and atomic;
  corrupt files are quarantined. A cache miss with no live path is a typed
  abstention, never a crash or a fabricated answer.
- **Calibration earns its place.** The LR+Platt pipeline is fitted on 70%; tau is
  chosen on OOF train probabilities; the test set is untouched. ECE/Brier are
  measured, not asserted from Jev confidence.
- **Published vs derived is visible.** 24 labels are explicit; 143 are derived by
  the locked policy. Every report/UI table separates them; `provenance_warning`
  is true.
- **Suppression is a policy, not a classifier.** The five strategies are a
  deterministic lookup; the UI and README say so.
- **Logging is non-blocking.** QueueHandler + daemon QueueListener, one JSON line
  per event; no secrets, no log files.

## Verification log

| Check | Evidence | Status |
| --- | --- | --- |
| Data gate | `python -m flameguard.dataset --audit` → 167 rows, 76/52/39, gate 150, reduced_claims false, provenance_warning true, 3 duplicate-state conflicts | **PASS** |
| Data provenance | every row has doc_id + table + URL + `outcome_quote`; test asserts it | **PASS** |
| Schema 422 | `tests/test_smoke.py::test_scenario_model...`, `test_api_422...` | **PASS** |
| Parser | 6 typed answers; missing/short-sum/out-of-range rejected | **PASS** |
| Combiner | 3-class fit, probabilities sum to 1, OOF tau | **PASS** |
| Suppression totality | exhaustive verdict × O2 × pressure × flow × severity × abstain | **PASS** |
| Offline path | margins mode: `/api/health` provider `physics`; `/api/predict` 200; OOD 200 `input_outside_observed_range`; uncached keyed mode 200 `needs_human_review` | **PASS** |
| Full pipeline | `evaluate --no-engine` → `artifacts/metrics.json` + `VALIDATION.md`, split 116/51, no NaN | **PASS** |
| Tests | `pytest -q` → **19 passed** (hermetic: real shell keys are cleared) | **PASS** |
| Canonical demo | 6 held-out cases → **5/6** correct with the margins engine; the miss (M20) is shown in the table | **PASS** |
| Metrics honesty | accuracy 0.431, macro-F1 0.381, ECE 0.200, Brier 0.216, abstention 0.392; published subset n=9 acc 0.444 / macro-F1 0.416 | **PASS (low, disclosed)** |
| Engine gate | `scripts/signal_check.py`; local jevos 4/10, Codiv 4/10, local+primer 3/10; real Jev not run here | **BLOCKED — no provider key in this shell** |
| Deploy/push | commit + push after the docs pass; `ENGINE_PROVIDER=physics`, `JEV_MODE=cache` needs no secret | **PENDING** |
| 60 s video | user screen recording; `DEMO.md` shot list | **PENDING (user)** |

## [ORPHANS & PENDING]

| # | Item | Status | Unblock / next action |
| --- | --- | --- | --- |
| 1 | Real Jev/OpenJev gate | **OPEN (external)** | Export a provider key, run `python scripts/signal_check.py`. ≥6/10 → `evaluate --fetch`, `precompute`, refit; the report labels the engine. <6/10 → keep the shipped margins engine. |
| 2 | Derived-label limitation | **KNOWN, DISCLOSED** | 143/167 labels depend on outcomes (burn time, spread rate) that are not inputs; the published 24 are the independent check. The report separates them; no blended number is called NASA accuracy. |
| 3 | Render production deploy | **PENDING** | Commit and push; `render.yaml` runs `ENGINE_PROVIDER=physics`, `JEV_MODE=cache`, no secret required. The physics artifacts are committed. |
| 4 | 1-minute demo video | **PENDING (user)** | Record offline (`ENGINE_PROVIDER=physics`); shot list in `DEMO.md`. No `ffmpeg`/Chromium on this machine. |
| 5 | Five-slide deck | **DONE (dynamic)** | `docs/deck.html` served at `/deck`; numbers fill from `/api/metrics`, so it cannot overclaim. |

Items 1 and 4 are the only open work; 2 is a disclosed scientific limit, not a
bug. The product is complete and runs offline with zero keys and zero tokens.