#!/usr/bin/env bash
# End-to-end build: one Jev call, 167-run fetch, combiner fit + VALIDATION.md,
# then the 14-case demo cache and the test suite.
#
#   bash scripts/run_build.sh
#
# Requires a key in the environment. Gateways need their own base/path/model:
#   BeatAPI:        export JEV_API_BASE=https://api.beatapi.io          JEV_MODEL=jev-1.14
#   jevtypesafe.org export JEV_API_BASE=https://jevtypesafe.org JEV_API_PATH=/api/v1/decide JEV_MODEL=jev
#   TypeSafe direct: export JEV_API_KEY=<type-safe key>   (JEV_API_BASE/MODEL unset)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-$HOME/.venvs/flameguard/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi
export JEV_TIMEOUT_S="${JEV_TIMEOUT_S:-60}"

if [ -z "${JEV_API_KEY:-}" ] && [ -z "${TYPESAFE_API_KEY:-}" ] && [ -z "${BEATAPI_API_KEY:-}" ]; then
  echo "no key in the environment: set JEV_API_KEY (or TYPESAFE_API_KEY / BEATAPI_API_KEY)" >&2
  echo "gateways: BeatAPI -> JEV_API_BASE=https://api.beatapi.io JEV_MODEL=jev-1.14 ;" >&2
  echo "          jevtypesafe.org -> JEV_API_BASE=https://jevtypesafe.org JEV_API_PATH=/api/v1/decide JEV_MODEL=jev" >&2
  exit 1
fi

echo "== 1/4 one real Jev call =="
"$PY" scripts/smoke_jev.py

echo "== 2/4 fetch 167 runs (resumable; this is the long step) =="
"$PY" -m flameguard.evaluate --fetch

echo "== 3/4 cache the 6 canonical + 8 novel demo cases =="
"$PY" -m flameguard.precompute
"$PY" -m flameguard.precompute --check

echo "== 4/4 tests =="
"$PY" -m pytest -q

echo "== headline =="
"$PY" -c "import json; m=json.load(open('artifacts/metrics.json')); print({k: m[k] for k in ('n_runs','accuracy','macro_f1','ece','brier','abstention_rate','reduced_claims','provenance_warning')})"
echo "done: artifacts/combiner.joblib, artifacts/metrics.json, VALIDATION.md and jev_cache.json are ready"