# FlameGuard SI

A fire-safety decision assistant for spacecraft cabins. Enter a microgravity
scenario — material, geometry/thickness, O2, pressure, opposed flow — and get a
calibrated flammability verdict (`sustained_spread` / `marginal` / `extinction`),
typed reason codes, and one of five suppression strategies. When the evidence is
not sufficient, it returns `needs_human_review` instead of guessing.

**The typed decision contract** is four `noul` threshold questions, one `choice`
for the flame class, and one `score` for severity. Three engines can answer it —
TypeSafe Jev, an OpenJev host, or a free OpenAI-compatible model — and a fourth
mode, **published-threshold margins**, computes the physics directly. In every
mode a small calibrated logistic regression turns the signals plus the computed
margins into the final distribution. The combiner is fitted on NASA-derived runs.

## Status — shipped engine

The repository ships with the **published-threshold margins** engine
(`feature_mode=margins_only`): zero API keys, zero tokens, fully offline. It
computes the O2 margin against the published limiting concentration, the flow
fraction against blow-off, a near-limit flag and a thickness regime, then
calibrates on the 167 runs. `VALIDATION.md` and `artifacts/metrics.json` are
generated; the six canonical cases score **5/6**.

The TypeSafe Jev, OpenJev and OpenAI-compatible engines are selectable (see
below) but are **not** what the shipped artifacts use. The 10-row signal gate
(`scripts/signal_check.py`) showed the decision models answering with a constant
class on this data, because the marginal/extinction labels depend on outcomes
burn time and spread rate that are not inputs. Switching to a keyed engine is
one environment block plus a refit, and the gate must pass 6/10 first.

```bash
# reproduce the shipped, keyless artifacts
python -m flameguard.evaluate --no-engine   # fit + validate on margins, no engine calls
python -m flameguard.precompute --check     # physics mode: no cache needed
python -m pytest -q                          # 19 tests
uvicorn flameguard.api:app --port 7860

# or use a keyed engine: set its block, then
python -m flameguard.evaluate --fetch        # 157 unique states -> cache -> fit -> VALIDATION.md
python -m flameguard.precompute              # +8 novel cases
```

`evaluate` labels the actual engine and feature mode in the report; `precompute
--check` fails closed on a missing cache.

**Estimated API cost.** Jev 1.14/1.13 costs **$42 per billion input tokens
($0.042 per million)** and output tokens are free ([docs.typesafe.ai/models](https://docs.typesafe.ai/models)).
One request here is ~2.3 KB (~600 tokens), so the 167-run fetch plus the 14-case
precompute is well under **$0.02**; even with retries and a pessimistic tokenizer
it stays under $0.10. Credit is not the constraint — a key is.

## Why this exists

NASA has decades of microgravity combustion data (FLEX, BASS/BASS-II, Saffire,
SoFIE). The data says what happened in a test; it does not directly answer what
to do about the cabin in front of you. FlameGuard turns published run tables into
a typed decision with a calibrated probability, an abstention path, and a
suppression policy.

## Data

`data/runs.jsonl` contains **167 complete labelled runs**:

| family | rows | source | label basis |
| --- | --- | --- | --- |
| BASS-II | 20 | NASA/TM-20210011385 table 5.1 | derived from spread rate |
| BASS SIBAL | 31 | NASA/TM-20210011385 table 7.1 | published comments + derived blanks |
| Saffire I/II | 11 | NTRS 20170008805 table I | published extinction + derived burn length |
| SoFIE GEL | 5 | NTRS 20240007628 table 1 | published extinction mode |
| FLEX | 100 | NTRS 20150023456 table IX | derived from burn time / test end |

- **24 published** labels, **143 derived** by the policy in
  [`data/LABEL_POLICY.md`](data/LABEL_POLICY.md). The report prints the two
  groups separately; a derived label is never called a NASA outcome.
- Every row has a source document, table, URL, and the exact source line in
  `outcome_quote`. `data/sources.md` records the extraction method.
- The gate is 150. This dataset passes it; because most labels are derived,
  `provenance_warning=true` is shown in the UI and report. If extraction had
  fallen below 150, `reduced_claims=true` would be set everywhere.
- Duplicate condition states with conflicting labels are counted by the audit,
  not silently merged.

## Jev contract (verified, not assumed)

The challenge brief cited `https://beta.typesafe.ai/docs/jev/get-started` and
`POST /v1/jev:run`. On 2026-11-15 the docs host does not resolve and the endpoint
returns **404**. The live OpenAPI 0.2.0 exposes:

- `POST https://api.typesafe.ai/v1/systemone`
- `GET https://api.typesafe.ai/v1/models`
- `Authorization: Bearer $JEV_API_KEY`

### Gateway options (Vercel AI Gateway or BeatAPI)

Both expose the native TypeSafe `/v1/systemone` request/response shape, so only
the base URL, model id and key change — no adapter.

**Vercel AI Gateway** (preferred here). Vercel serves a TypeSafe-compatible base
at `https://ai-gateway.vercel.sh/typesafe`; the documented model ids are
`typesafe-ai/jev`, `typesafe-ai/jev-1.14.0` and `typesafe-ai/jev-1.13.0`, with
`Authorization: Bearer $AI_GATEWAY_API_KEY`. Pricing is **$0.042 per 1M input
tokens** and output is free, so the whole 167-run build costs far less than a
cent.

```bash
export JEV_API_BASE=https://ai-gateway.vercel.sh/typesafe
export JEV_MODEL=typesafe-ai/jev
export JEV_API_KEY="$AI_GATEWAY_API_KEY"      # or paste the Vercel AI Gateway key
curl -s -o /tmp/models -w '%{http_code}\n' \
  -H "Authorization: Bearer $JEV_API_KEY" \
  https://ai-gateway.vercel.sh/typesafe/v1/models
```

**BeatAPI**. Base `https://api.beatapi.io`, model ids `jev-1.14` (default) and
`jev-1.13`, auth `Authorization: Bearer $BEATAPI_API_KEY`.

```bash
export JEV_API_BASE=https://api.beatapi.io
export JEV_MODEL=jev-1.14
export JEV_API_KEY="$BEATAPI_API_KEY"
```

`api_key()` accepts `JEV_API_KEY`, `TYPESAFE_API_KEY`, `BEATAPI_API_KEY`,
`AI_GATEWAY_API_KEY` or `VERCEL_AI_GATEWAY_KEY`, in that order. When `JEV_MODEL`
is unset, the base URL picks the provider default (`typesafe-ai/jev` for the
Vercel gateway, `jev-1.14` for BeatAPI, `jev-latest` for TypeSafe direct). A
`200` from the models call means the key works; then run the two build commands.
Each run spends that provider's balance, so smoke-test one scenario first.

### Other typed-judgment hosts

Any host serving the same `state` + `questions` (`noul`/`choice`/`score`) schema
works; set `JEV_API_PATH` when its path differs. Example, the independent
`jevtypesafe.org` host (not affiliated with TypeSafe), which uses
`POST /api/v1/decide` and `model: "jev"`:

```bash
export JEV_API_BASE=https://jevtypesafe.org
export JEV_API_PATH=/api/v1/decide
export JEV_MODEL=jev
export JEV_API_KEY="jv_live_..."
```

`jev-agent.com` ("Jagent", also independent) serves the same shape at
`POST /api/v1/systemone` with `model: "jev-latest"`:

```bash
export JEV_API_BASE=https://jev-agent.com
export JEV_API_PATH=/api/v1/systemone
export JEV_MODEL=jev-latest
export JEV_API_KEY="jv_live_..."
```

The endpoint actually used is recorded in every cache entry and printed in
`VALIDATION.md`, so the report always states which host served the answers.

The OpenAPI document is committed at [`docs/typesafe-openapi.json`](docs/typesafe-openapi.json).
The client uses `JEV_API_KEY` and accepts `TYPESAFE_API_KEY` as a legacy alias.
Every request is cached in `engine_cache.json` by a hash of state + question
contract; a cache miss with no key or network returns a typed abstention, never
a fabricated answer.

## The six questions

| id | type | what it asks |
| --- | --- | --- |
| `o2_above_limiting` | `noul` | Is O2 above the limiting concentration for this material/geometry? |
| `flow_above_blowoff` | `noul` | Is opposed flow above the blow-off threshold? |
| `self_sustaining_60s` | `noul` | Would a flame persist for 60 s without further energy? |
| `spread_beyond_ignition` | `noul` | Would the flame spread beyond the ignition region? |
| `flame_class` | `choice` | Three-way flame behaviour. |
| `severity` | `score` | 0 none · 1 localised/self-extinguishing · 2 steady/limited · 3 self-propagating. |

The combiner consumes the fixed 8-vector:
`[4 noul probabilities, P(extinction), P(marginal), P(sustained_spread), severity/3]`.

## Calibration and validation

- 70/30 split stratified by label; the six canonical cases are forced into the
  held-out split.
- The combiner is `LogisticRegression` + sigmoid (`CalibratedClassifierCV`) fitted
  on the 70% only. Tau is chosen on out-of-fold train probabilities; the test set
  is untouched.
- The report contains accuracy, macro-F1, ECE (10 equal-frequency bins), Brier,
  abstention rate, coverage, selective accuracy, per-class P/R/F1, confusion
  matrix, per-family and published-vs-derived tables, bootstrap 95% CIs, and
  misclassified examples.
- **Calibration is validated against NASA-derived outcomes; it is not inherited
  from Jev.** Jev returns valid typed answers, not guaranteed-correct ones.
- The suppression strategy is a deterministic policy from published fire-safety
  guidance, not a learned classifier, and is not validated here.

Run `python -m flameguard.evaluate --fetch` to generate `artifacts/metrics.json`
and `VALIDATION.md`. The `/metrics` page reads the JSON; the SPA never hardcodes
a number.

## Run locally

```bash
python3 -m venv --without-pip ~/.venvs/flameguard
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/.venvs/flameguard/bin/python /tmp/get-pip.py
~/.venvs/flameguard/bin/python -m pip install -r requirements-dev.txt

export JEV_API_KEY=...
~/.venvs/flameguard/bin/python -m flameguard.dataset --audit
~/.venvs/flameguard/bin/python -m flameguard.evaluate --fetch
~/.venvs/flameguard/bin/python -m flameguard.precompute
~/.venvs/flameguard/bin/python -m pytest -q
~/.venvs/flameguard/bin/python -m uvicorn flameguard.api:app --port 7860
```

Open <http://127.0.0.1:7860>. The page shows the data gate, the six canonical
predicted-vs-actual rows, and eight novel in-envelope scenarios. `/metrics` is
the validation page; `/VALIDATION.md` is the generated report.

**Offline demo (the path that must never fail):**

```bash
env -u JEV_API_KEY -u TYPESAFE_API_KEY JEV_MODE=cache \
  ~/.venvs/flameguard/bin/python -m uvicorn flameguard.api:app --port 7860
```

The cached path is milliseconds; live Jev latency is reported separately and is
not claimed to be under a second.

## API

| endpoint | purpose |
| --- | --- |
| `GET /api/health` | status, engine mode, cache entries, gate; never calls Jev |
| `GET /api/meta` | field schema, classes, strategies, tau, gate/provenance |
| `GET /api/cases` | 6 canonical (actual + predicted) and 8 novel scenarios |
| `POST /api/predict` | typed verdict, calibrated probability, reason codes, strategy |
| `GET /api/metrics` | generated `artifacts/metrics.json` |
| `GET /VALIDATION.md` | generated report |
| `GET /metrics` | validation page |
| `GET /deck` | five-slide deck, fills its numbers from `/api/metrics` |

Invalid payloads return 422 with field paths. A Jev/network failure returns 200
with `needs_human_review` and a typed reason; it never returns 500.

## Deploy

The Dockerfile installs the pinned runtime, copies the committed artifacts, and
runs uvicorn; there is no training at build time. `render.yaml` sets
`JEV_MODE=cache` and a secret `JEV_API_KEY`. Deploy only after the cache,
combiner and `VALIDATION.md` have been generated; otherwise the canonical table
correctly shows “not cached”.

## Limits

See [`VALIDATION.md`](VALIDATION.md) once generated. The short version: the
dataset is a convenience extraction, most labels are derived, the held-out set is
small, FLEX `Extinction` is a normal burn-out unless the burn-time rule says
otherwise, SoFIE rows are extinction-point conditions, and the suppression
policy is rule-based. Nothing here is a flight-qualified procedure.

## License

MIT. NASA NTRS documents are U.S. Government works; see `data/sources.md` for
attribution.