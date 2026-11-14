# FlameGuard SI — PROJECT_MAP

Calibrated flammability decision layer for spacecraft cabin fire safety. Turns
labelled microgravity combustion runs into a typed verdict + suppression
strategy with an explicit abstention path.

## [TECH_STACK]

Pinned in `requirements.txt`; every wheel verified installable on **cp314** with
zero source builds. `requirements-dev.txt` adds the test-only packages so the
container does not ship pytest.

| Component | Version | Why this and not more |
| --- | --- | --- |
| Python | 3.14.4 (venv) | System interpreter is PEP 668-locked and lacks `ensurepip`; venv created with `--without-pip` + `get-pip.py`, kept on the Linux-native filesystem because importing scikit-learn from `/mnt/c` costs ~40 s vs ~3 s natively |
| scikit-learn | 1.9.1 | `HistGradientBoostingClassifier` (native categorical + NaN) and `CalibratedClassifierCV` come free — LightGBM/XGBoost rejected as extra dependencies with no gain at this scale |
| numpy | 2.5.4 | Only array layer needed; **pandas deliberately excluded** (stdlib `csv` + numpy) |
| pydantic | 2.14.3 | Request schema generated at runtime from `features.json` via `create_model` |
| fastapi | 0.162.7 | Free 422-on-invalid-payload with field paths (criterion F1) |
| uvicorn | 0.54.0 | ASGI server, no `[standard]` extras |
| joblib | 1.6.0 | Model bundle serialisation (atomic write) |
| httpx2 / pytest | 2.16.0 / 9.2.0 | **dev only** — `httpx` is deprecated by Starlette 1.10, hence httpx2 |
| Frontend | — | One static HTML file, vanilla JS/CSS, no Node, no npm, no build step |

Environment: WSL2, CPU-only. Build date **2026-11-14**.

## [SYSTEM_FLOW]

```
data/raw/*.csv  (organizer ground truth; SYNTHETIC placeholder if absent)
        |
        v
train.py : detect target -> infer schema -> class normalisation -> leakage check
           -> stratified CV -> calibration decision -> tau tuning -> fit -> persist
        v
artifacts/ model.joblib · features.json · metrics.json · validation.md
        |
        +---------------------------+
        v                           v
api.py  /api/meta  -->  drives the SPA form (field kinds, ranges, categories, roles)
        /api/health
        /api/predict
        v
static/index.html  --POST /api/predict {scenario}-->
        v
verdict card: class + confidence + abstain flag + suppression strategy
              + k nearest LABELLED runs with their actual outcomes
```

Every stage is on the judge's demo path; nothing else exists.

## [ARCHITECTURE]

One shared layer, because `train` and `api` both need schema + encoding and must
never disagree.

| File | Role | Measured lines |
| --- | --- | --- |
| `flameguard/core.py` | Schema discovery, CSV I/O, encoding, abstention, suppression lookup, k-NN, logger, artifact bundle, runtime request schema | 403 |
| `flameguard/train.py` | Data discovery, class normalisation, leakage-aware split, CV, calibration decision, tau tuning, metrics + validation report | 407 |
| `flameguard/api.py` | FastAPI surface, role detection, OOD abstention, typed response | 226 |
| `static/index.html` | The SPA: form → verdict card → comparison table | 323 |
| `tests/test_smoke.py` | 11 tests covering F1, F4–F8 and the sampled regressions | 158 |
| `Dockerfile` | HF Spaces (Docker SDK), digest-pinned base, retrains at build | 24 |

**Budget deviation, recorded rather than hidden:** the plan set caps of
200/160/130/280 lines. Actual is 403/407/226/323. The overrun is not speculative
code — the validation-report renderer, the synthetic fallback, the schema
hardening (encoding fallback, finite-stat guards, token-based identifier
detection) and role/OOD handling all landed inside these modules instead of
being split into micro-files. No function is unreachable. The caps were wrong
estimates and are superseded by the measured values above.

Key decisions:

- **Schema is discovered, never assumed.** `infer_meta` classifies each column
  from the data and records categories/min/max/mean/std in `features.json`; the
  API request model is generated from that file. Feature *names* are never
  hard-coded. The four **physical roles** (O2, pressure, confinement, flow) that
  the suppression lookup needs are detected by token heuristic in `api.py`; if a
  role is not found the strategy degrades to an "unknown" band rather than
  guessing, and the UI hides presets that depend on it.
- **Calibration must earn its place.** Platt scaling is kept only if it beats
  the uncalibrated model on macro Brier; otherwise it is dropped (F3).
- **Abstention is rule-based and tunable without retraining:**
  `max_prob < tau_global` OR `(p1 - p2) < tau_margin` OR an input outside the
  observed training range. `tau` is chosen for highest accuracy on the
  auto-clear subset among thresholds whose review rate falls in the 10–30% band,
  ties broken toward fewer deferrals.
- **Suppression mapping is deterministic and explicitly NOT learned**, sourced
  from published NASA fire-safety guidance and labelled as such in the UI.
- **Logging is non-blocking** (`QueueHandler` + daemon `QueueListener`), one JSON
  line per event, no log files, no dependency.

## Verification log

| ID | Criterion | Evidence |
| --- | --- | --- |
| F0 | Env reproducible | `pip check` clean, all cp314 wheels, no source builds; digest-pinned base image |
| F1 | Ingestion validation | 422 with `loc` paths for missing field, bad type, unknown category, extra field |
| F2 | Model beats baseline | macro-F1 **0.788** vs majority baseline **0.185** |
| F3 | Calibration earns place | Brier **0.1133 → 0.1059**, kept; dropped automatically otherwise |
| F4 | Abstention | flat/low-margin/OOD vectors abstain (unit-tested); τ=0.55 → **21.7%** review, 85.8% accuracy on auto-clear |
| F5 | Suppression total | exhaustive over 4 verdicts × 5 O2 × 4 pressures × 3 confinement states; unit normalisation tested |
| F6 | Nearest runs | ordered, exact self-match at distance 0, only feature columns exposed |
| F7 | Verdict API | typed contract asserted; **p95 51.3 ms** over 60 calls |
| F8 | SPA | headless Chrome render with **no console errors**; abstain branch + escaping verified against the shipping script |
| F9 | Deploy | **live** at <https://flameguard-si.onrender.com> — health, meta, predict, 422 and OOD abstention verified from outside, and the SPA renders in a real browser with no console errors; warm p95 **314 ms** |

Accuracy figures are on **SYNTHETIC** data and are labelled as such everywhere
(`metrics.json`, `validation.md`, UI badge).

Production latency note: warm p95 is 314 ms over the network versus 51 ms on
localhost, so ~260 ms is round-trip time, not inference (model scoring is ~1 ms).
The host reports `gcp-us-west1`; recreating the service in Frankfurt would
roughly halve it. Still comfortably inside the "under a second" claim.

## [ORPHANS & PENDING]

| # | Item | Status | Notes |
| --- | --- | --- | --- |
| 1 | Real PSI dataset (BASS PSI-26, BASS-II PSI-25) in `data/raw/` | **BLOCKED ON USER** | The challenge designates NASA PSI, whose datasets need an account to download. Pipeline runs on a clearly-labelled SYNTHETIC placeholder. Drop the files and re-run `python -m flameguard.train`. |
| 2 | NASA-validated numbers | **BLOCKED ON #1** | Current metrics are synthetic and marked as such in three places. |
| 3 | PSI→feature adapter | **BLOCKED ON #1** | Cannot be written meaningfully until the real column layout is visible. |
| 4 | NTRS paper extraction (SoFIE/BASS/FLEX) | **OUT OF SCOPE** | Rescinded by the <24 h decision. |
| 5 | Stretch goals (2D/3D view, what-if slider, confusion-matrix UI) | **OUT OF SCOPE** | Confusion matrix is emitted to `validation.md` instead. |

Items 1–3 cannot be closed from this machine: they need PSI files only the user
can download. Everything that does not depend on them is finished and verified.
The container image is no longer a risk — Render built it and the service runs.