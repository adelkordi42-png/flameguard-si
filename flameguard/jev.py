"""FlameGuard SI — Jev (TypeSafe System One) client.

One request carries the scenario plus six typed questions: four `noul`
thresholds, one `choice` for the flame class, one `score` for severity. The
response is validated before it can influence a verdict, and every response is
cached on disk by a hash of the state + question contract.

Verified contract (2026-11-15): POST https://api.typesafe.ai/v1/systemone.
The brief's /v1/jev:run returns 404; it is deliberately not used.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from . import core, features


def _api_base() -> str:
    """TypeSafe directly, or a host that exposes the same typed decision schema."""
    override = (os.environ.get("JEV_API_BASE") or "").strip().rstrip("/")
    return override or "https://api.typesafe.ai"


def _api_path() -> str:
    override = (os.environ.get("JEV_API_PATH") or "").strip()
    if not override:
        return "/v1/systemone"
    return override if override.startswith("/") else "/" + override


BASE_URL = _api_base()
API_PATH = _api_path()
ENDPOINT = f"{BASE_URL}{API_PATH}"
MODELS_ENDPOINT = f"{BASE_URL}/v1/models"
PROBE_URL = f"{BASE_URL}/openapi.json"
DEFAULT_MODEL = "jev-latest"
QUESTIONS_VERSION = "v2"


def _timeout() -> float:
    try:
        return max(5.0, float(os.environ.get("JEV_TIMEOUT_S", "30")))
    except ValueError:
        return 30.0


TIMEOUT_S = _timeout()

PRIMER = (
    "Microgravity reference from published NASA runs: PMMA flat sheets sustain spread above about "
    "18% O2 and quench or fail below about 17%; SIBAL cotton-fiberglass fabric quenches at about "
    "17-18.7% O2 and fails to ignite at about 16.4-16.8%; droplet fuels extinguish below about 16% "
    "O2. Opposed flow below about 5 cm/s does not blow off; above about 40 cm/s it can blow off a "
    "developing flame, especially for thin samples. Thick PMMA (above about 3 mm) spreads slowly, "
    "and a spread rate below about 0.02 mm/s is near-limit. Droplet burn time grows with the square "
    "of the initial diameter; small droplets (below about 1.5 mm) at low O2 can fail within seconds. "
    "Treat a scenario within about 1% O2 of the limit, or within about 20% of the blow-off flow, as "
    "near-limit rather than clearly flammable."
)

CRITERIA = {
    "sustained_spread": (
        "The material ignites and combustion is self-sustaining, spreading across or along the "
        "surface. Oxygen is clearly above the limiting value, flow is clearly below blow-off, and "
        "the flame keeps advancing rather than dying after ignition."
    ),
    "marginal": (
        "The material ignites but behaviour is near the flammability limit: within about 1% O2 of the "
        "limiting concentration, within about 20% of the blow-off flow, a thick sample spreading below "
        "about 0.02 mm/s, a disruptive or intermittent droplet burn, or a burn lasting 2-5 s. Choose "
        "this only when spread is genuinely ambiguous between sustained and extinction."
    ),
    "extinction": (
        "The material fails to sustain combustion: O2 below the limiting concentration, opposed flow "
        "above the blow-off threshold, no ignition, a quench, or a burn shorter than about 2 s."
    ),
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "o2_above_limiting": {
        "type": "noul",
        "instructions": (
            "The oxygen concentration in this scenario is above the limiting oxygen concentration "
            "for this material and geometry."
        ),
        "criteria": {
            "true": "Oxygen is high enough to support a self-sustaining flame.",
            "false": "Oxygen is below the limit; ignition cannot sustain combustion.",
        },
    },
    "flow_above_blowoff": {
        "type": "noul",
        "instructions": (
            "The opposed-flow velocity is high enough to blow off or extinguish a developing flame."
        ),
        "criteria": {
            "true": "The forced flow exceeds the blow-off threshold for this material.",
            "false": "The forced flow is below the blow-off threshold.",
        },
    },
    "self_sustaining_60s": {
        "type": "noul",
        "instructions": (
            "If ignited at the igniter, a flame would still be burning 60 seconds later without "
            "further energy input."
        ),
        "criteria": {
            "true": "Combustion persists for at least 60 s unaided.",
            "false": "The flame extinguishes before 60 s or never establishes.",
        },
    },
    "spread_beyond_ignition": {
        "type": "noul",
        "instructions": (
            "The flame would spread beyond the ignition region along the material surface."
        ),
        "criteria": {
            "true": "The flame advances beyond the igniter footprint.",
            "false": "The flame remains localised or does not establish.",
        },
    },
    "flame_class": {
        "type": "choice",
        "instructions": "Classify the most likely flame behaviour for this scenario.",
        "criteria": CRITERIA,
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the fire-safety outcome for this scenario?",
        "criteria": [
            "No ignition or immediate extinction; no hazard.",
            "Localised burning that self-extinguishes.",
            "Steady burning with limited spread.",
            "Self-propagating spread with growth potential.",
        ],
    },
}

log = core.get_logger()


class JevError(RuntimeError):
    """The Jev call failed or returned an unusable answer."""


class JevUnavailable(JevError):
    """No cached answer and live calls are not allowed or not possible."""


class JevAuthError(JevError):
    """The credential is missing or rejected."""


@dataclass
class JevResponse:
    answers: dict[str, Any]
    model: str
    usage: dict[str, Any]
    latency_ms: float
    engine: str
    source_latency_ms: float | None = None
    provider: str = "jev"

    def signals(self) -> dict[str, float]:
        signals = {
            name: float(value)
            for name, value in self.answers.items()
            if isinstance(value, (int, float))
        }
        signals["severity"] = float(self.answers["severity"]["score"])
        return signals

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "source_latency_ms": round(self.source_latency_ms, 2) if self.source_latency_ms else None,
            "usage": self.usage,
            "signals": {key: round(value, 4) for key, value in self.signals().items()},
        }


KEY_ENVS = {
    "beatapi": ("BEATAPI_API_KEY",),
    "ai-gateway.vercel.sh": ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_KEY"),
    "codiv": ("CODIV_API_KEY",),
    "typesafe": ("TYPESAFE_API_KEY",),
}


def key_env_names() -> tuple[str, ...]:
    """Provider-specific key names first, then the generic JEV_API_KEY."""
    base = BASE_URL.lower()
    specific: tuple[str, ...] = ()
    for marker, names in KEY_ENVS.items():
        if marker in base:
            specific = names
            break
    return tuple(dict.fromkeys(specific + ("JEV_API_KEY",)))


def key_source() -> str | None:
    for name in key_env_names():
        if (os.environ.get(name) or "").strip():
            return name
    return None


def api_key() -> str | None:
    source = key_source()
    return (os.environ.get(source) or "").strip() if source else None


def model_name() -> str:
    explicit = os.environ.get("JEV_MODEL", "").strip()
    if explicit:
        return explicit
    base = BASE_URL.lower()
    if "beatapi" in base:
        return "jev-1.14"
    if "ai-gateway.vercel.sh" in base:
        return "typesafe-ai/jev"
    return DEFAULT_MODEL


def mode() -> str:
    value = os.environ.get("JEV_MODE", "auto").strip().lower()
    return value if value in {"cache", "auto", "live"} else "auto"


def provider_name() -> str:
    """Which engine is behind the Jev-compatible wire API."""
    explicit = (os.environ.get("ENGINE_PROVIDER") or "").strip().lower()
    if explicit:
        return explicit
    base = BASE_URL.lower()
    if "127.0.0.1" in base or "localhost" in base or "codiv" in base:
        return "openjev"
    if "beatapi" in base or "typesafe" in base:
        return "jev"
    return "custom"


def engine_label(provider: str | None = None) -> str:
    name = provider or provider_name()
    return {
        "openjev": "OpenJev (Jev-compatible)",
        "jev": "Jev-compatible hosted",
        "llm": "LLM wrapper",
    }.get(name, name)


def render_state(values: dict[str, Any]) -> dict[str, Any]:
    order = (
        "material",
        "geometry",
        "thickness_mm",
        "o2_percent",
        "pressure_kpa",
        "opposed_flow_cm_s",
        "igniter",
        "gravity",
    )
    state: dict[str, Any] = {}
    for name in order:
        value = values.get(name)
        if value is None or value == "":
            continue
        state[name] = float(value) if isinstance(value, (int, float)) else value
    state["microgravity_reference"] = PRIMER
    state["derived"] = features.margins(state)
    return state


def cache_key(state: dict[str, Any], model: str) -> str:
    return core.canonical_hash(
        {
            "schema": core.SCHEMA_VERSION,
            "provider": provider_name(),
            "model": model,
            "questions_version": QUESTIONS_VERSION,
            "state": render_state(state),
            "questions": QUESTIONS,
        }
    )


def build_request(state: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    return {"model": model or model_name(), "state": render_state(state), "questions": QUESTIONS}


def parse_answers(answers: Any) -> dict[str, Any]:
    if not isinstance(answers, dict):
        raise JevError("response has no answers object")
    missing = sorted(set(QUESTIONS) - set(answers))
    if missing:
        raise JevError(f"missing answers: {missing}")
    extra = sorted(set(answers) - set(QUESTIONS))
    if extra:
        raise JevError(f"unexpected answers: {extra}")
    parsed: dict[str, Any] = {}
    for name, spec in QUESTIONS.items():
        answer = answers[name]
        if not isinstance(answer, dict):
            raise JevError(f"{name}: answer is not an object")
        kind = spec["type"]
        if kind == "noul":
            if answer.get("type") != "noul":
                raise JevError(f"{name}: expected noul, got {answer.get('type')!r}")
            value = _unit(answer.get("noul"), name)
            parsed[name] = value
        elif kind == "choice":
            expected = set(spec["criteria"])
            probabilities = answer.get("probabilities")
            choice = str(answer.get("choice") or "")
            if not isinstance(probabilities, dict) or not probabilities:
                # Some gateways return only the selected option. The raw answer is
                # still cached; encode the selection as a one-hot so the pipeline
                # keeps the model's actual decision instead of failing the run.
                if choice not in expected:
                    raise JevError(f"{name}: no probabilities and choice {choice!r} not in criteria")
                cleaned = {key: (1.0 if key == choice else 0.0) for key in spec["criteria"]}
            else:
                if set(probabilities) != expected:
                    raise JevError(f"{name}: probability keys {sorted(probabilities)} != {sorted(expected)}")
                cleaned = {key: _unit(value, f"{name}.{key}") for key, value in probabilities.items()}
                if abs(sum(cleaned.values()) - 1.0) > 0.05:
                    raise JevError(f"{name}: probabilities sum to {sum(cleaned.values()):.3f}")
                choice = choice or max(cleaned, key=cleaned.get)
                if choice not in cleaned:
                    raise JevError(f"{name}: choice {choice!r} absent from probabilities")
            confidence = answer.get("confidence")
            parsed[name] = {
                "choice": choice,
                "confidence": _unit(cleaned[choice] if confidence is None else confidence, f"{name}.confidence"),
                "probabilities": cleaned,
            }
        elif kind == "score":
            score = float(answer.get("score"))
            if not 0.0 <= score <= len(spec["criteria"]) - 1:
                raise JevError(f"{name}: score {score} outside rubric")
            confidence = answer.get("confidence")
            parsed[name] = {
                "score": score,
                "confidence": _unit(0.0 if confidence is None else confidence, f"{name}.confidence"),
                "probabilities": {
                    str(k): _unit(v, f"{name}.probabilities.{k}")
                    for k, v in (answer.get("probabilities") or {}).items()
                },
            }
        else:  # pragma: no cover - the spec above is fixed
            raise JevError(f"{name}: unsupported question type {kind}")
    return parsed


def _unit(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JevError(f"{label}: {value!r} is not numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise JevError(f"{label}: {number} outside [0, 1]")
    return number


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = TIMEOUT_S,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "flameguard-si/2.0", **(headers or {})},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        if exc.code in (401, 403):
            raise JevAuthError(f"HTTP {exc.code} from Jev: {detail}") from exc
        raise JevError(f"HTTP {exc.code} from Jev: {detail}") from exc
    except urllib.error.URLError as exc:
        raise JevError(f"cannot reach Jev: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise JevError(f"Jev returned non-JSON: {exc}") from exc


class JevEngine:
    """Cache-first client. `transport` is injectable for tests and offline runs."""

    def __init__(
        self,
        key: str | None = None,
        model: str | None = None,
        transport: Callable[[str, dict[str, Any] | None, dict[str, str], float], dict[str, Any]] | None = None,
        cache_path: str | None = None,
    ) -> None:
        self.key = key if key is not None else api_key()
        self.model = model or model_name()
        self.transport = transport or _http_json
        self.cache_path = cache_path or str(core.CACHE_PATH)
        self.provider = provider_name()
        self._probe_at = 0.0
        self._probe_ok = False

    # -------------------------------------------------------------- public API
    def key_for(self, state: dict[str, Any]) -> str:
        return cache_key(state, self.model)

    def probe(self, force: bool = False) -> bool:
        if not force and time.monotonic() - self._probe_at < 60.0:
            return self._probe_ok
        self._probe_at = time.monotonic()
        try:
            self.transport(PROBE_URL, None, {}, 1.5)
            self._probe_ok = True
        except JevError as exc:
            # A gateway may not serve /openapi.json; any HTTP status proves the
            # host is reachable. Only a transport failure means the network is down.
            self._probe_ok = str(exc).startswith("HTTP ")
        return self._probe_ok

    def models(self) -> list[dict[str, Any]]:
        if not self.key:
            raise JevAuthError("JEV_API_KEY is not set")
        body = self.transport(MODELS_ENDPOINT, None, {"Authorization": f"Bearer {self.key}"}, TIMEOUT_S)
        models = None
        if isinstance(body, dict):
            models = body.get("models")
            if not isinstance(models, list):
                models = body.get("data")
        if not isinstance(models, list):
            raise JevError("models response has no 'models' or 'data' list")
        return models

    def answer(self, state: dict[str, Any], allow_live: bool | None = None) -> JevResponse:
        started = time.perf_counter()
        key = self.key_for(state)
        entry = core.cache_get(key, self.cache_path)
        if entry and entry.get("questions_version") == QUESTIONS_VERSION:
            try:
                answers = parse_answers(entry.get("answers"))
            except JevError as exc:
                core.log_event(log, "cache_entry_invalid", key=key[:12], error=str(exc), level=30)
            else:
                core.log_event(log, "jev_cache_hit", key=key[:12], model=entry.get("model"))
                return JevResponse(
                    answers=answers,
                    model=str(entry.get("model") or self.model),
                    usage=entry.get("usage") or {},
                    latency_ms=(time.perf_counter() - started) * 1000,
                    engine="jev_cache",
                    source_latency_ms=entry.get("latency_ms"),
                    provider=self.provider,
                )
        core.log_event(log, "jev_cache_miss", key=key[:12], model=self.model)
        live_allowed = mode() != "cache" if allow_live is None else allow_live
        if not live_allowed:
            raise JevUnavailable("cache miss and live calls disabled")
        if not self.key:
            raise JevAuthError("JEV_API_KEY is not set")
        if allow_live is None and mode() == "auto" and not self.probe():
            raise JevUnavailable("cache miss and network probe failed")
        body = self._call(build_request(state, self.model))
        answers = parse_answers(body.get("answers"))
        latency_ms = (time.perf_counter() - started) * 1000
        actual_model = str(body.get("model") or self.model)
        entry = {
            "provider": self.provider,
            "requested_model": self.model,
            "model": actual_model,
            "endpoint": ENDPOINT,
            "questions_version": QUESTIONS_VERSION,
            "state": render_state(state),
            "answers": body["answers"],
            "usage": body.get("usage") or {},
            "fetched_at": core.iso_now(),
            "latency_ms": round(latency_ms, 2),
        }
        core.cache_put(key, entry, self.cache_path)
        core.log_event(
            log,
            "jev_call",
            key=key[:12],
            model=actual_model,
            latency_ms=round(latency_ms, 1),
            usage=entry["usage"],
        )
        return JevResponse(
            answers=answers,
            model=actual_model,
            usage=entry["usage"],
            latency_ms=latency_ms,
            engine="jev_live",
            source_latency_ms=latency_ms,
            provider=self.provider,
        )

    # --------------------------------------------------------------- transport
    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.key is not None
        headers = {"Authorization": f"Bearer {self.key}"}
        last: JevError | None = None
        for attempt in range(3):
            try:
                body = self.transport(ENDPOINT, payload, headers, TIMEOUT_S)
                if not isinstance(body, dict):
                    raise JevError("Jev returned a non-object response")
                return body
            except JevAuthError:
                raise
            except JevError as exc:
                last = exc
                text = str(exc)
                retryable = "HTTP 429" in text or "HTTP 5" in text or "cannot reach" in text
                if not retryable or attempt == 2:
                    raise
                time.sleep(0.5 * (2**attempt))
        raise last or JevError("Jev call failed")