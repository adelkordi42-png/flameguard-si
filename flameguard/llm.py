"""OpenAI-compatible substitute engine.

Jev's free variants were retired (2026-11-05) and its minimum top-up is out of
reach for this demo, so this engine preserves the architecture:

    six typed signals per state -> calibrated logistic combiner -> verdict

Any OpenAI-compatible chat endpoint that can return JSON works: BeatAPI free
models, Groq, Google AI Studio, Cerebras, OpenRouter, GitHub Models, or a local
Ollama server. The model supplies the signals; the combiner supplies the
calibration, exactly as with Jev. Set:

    LLM_BASE_URL=https://api.beatapi.io/v1
    LLM_MODEL=deepseek-v4.1-flash-free
    LLM_API_KEY=...            # falls back to BEATAPI_API_KEY

When LLM_BASE_URL/LLM_MODEL are unset, build_engine() returns the Jev client.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable

from . import core, jev

PROMPT_VERSION = "v1"
DEFAULT_BASE = "https://api.beatapi.io/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash-free"
MAX_TOKENS = 400

SHAPE = (
    '{"o2_above_limiting":0.0,"flow_above_blowoff":0.0,"self_sustaining_60s":0.0,'
    '"spread_beyond_ignition":0.0,'
    '"flame_class":{"extinction":0.0,"marginal":0.0,"sustained_spread":0.0},'
    '"severity":0.0}'
)


def _definitions() -> str:
    lines = []
    for name, spec in jev.QUESTIONS.items():
        if spec["type"] == "noul":
            lines.append(f"- {name} (0..1, probability the statement is true): {spec['instructions']}")
            criteria = spec.get("criteria") or {}
            lines.append(f"    true means: {criteria.get('true', '')}")
            lines.append(f"    false means: {criteria.get('false', '')}")
        elif spec["type"] == "choice":
            lines.append(f"- flame_class (probabilities summing to 1): {spec['instructions']}")
            for key, text in spec["criteria"].items():
                lines.append(f"    {key}: {text}")
        elif spec["type"] == "score":
            lines.append(f"- severity (0..3): {spec['instructions']}")
            for level, text in enumerate(spec["criteria"]):
                lines.append(f"    {level}: {text}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a decision model for spacecraft cabin fire safety in microgravity. "
    "Answer six questions about ONE scenario. Return ONLY a JSON object, no prose, "
    "no markdown fences, with exactly these keys and numeric values:\n"
    f"{SHAPE}\n"
    "Rules: each noul is a probability from 0 to 1; the three flame_class "
    "probabilities must sum to 1; severity is 0 to 3 and may be fractional.\n"
    "Question definitions:\n"
    f"{_definitions()}"
)


class LlmEngine:
    """Cache-first engine with the same public surface as jev.JevEngine."""

    def __init__(
        self,
        base: str | None = None,
        key: str | None = None,
        model: str | None = None,
        transport: Callable[[str, dict[str, Any] | None, dict[str, str], float], dict[str, Any]] | None = None,
        cache_path: str | None = None,
    ) -> None:
        self.base = (base or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.key = key if key is not None else (
            os.environ.get("LLM_API_KEY") or os.environ.get("BEATAPI_API_KEY") or ""
        )
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.transport = transport or jev._http_json
        self.cache_path = cache_path or str(core.CACHE_PATH)
        self.provider = "llm"
        self.min_interval = max(0.0, float(os.environ.get("LLM_MIN_INTERVAL_S", "0.2")))
        self._last_call = 0.0
        self._probe_at = 0.0
        self._probe_ok = False

    def _pace(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    # ------------------------------------------------------------------ cache
    def key_for(self, state: dict[str, Any]) -> str:
        return core.canonical_hash(
            {
                "schema": core.SCHEMA_VERSION,
                "engine": "llm",
                "base": self.base,
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                "state": jev.render_state(state),
                "questions": jev.QUESTIONS,
            }
        )

    def probe(self, force: bool = False) -> bool:
        if not force and time.monotonic() - self._probe_at < 60.0:
            return self._probe_ok
        self._probe_at = time.monotonic()
        try:
            self.transport(f"{self.base}/models", None, self._headers(), 1.5)
            self._probe_ok = True
        except jev.JevError as exc:
            self._probe_ok = str(exc).startswith("HTTP ")
        return self._probe_ok

    def models(self) -> list[dict[str, Any]]:
        if not self.key:
            raise jev.JevAuthError("LLM_API_KEY / BEATAPI_API_KEY is not set")
        body = self.transport(f"{self.base}/models", None, self._headers(), jev.TIMEOUT_S)
        models = None
        if isinstance(body, dict):
            models = body.get("models") or body.get("data")
        if not isinstance(models, list):
            raise jev.JevError("models response has no 'data' or 'models' list")
        return models

    # ----------------------------------------------------------------- answer
    def answer(self, state: dict[str, Any], allow_live: bool | None = None) -> jev.JevResponse:
        started = time.perf_counter()
        key = self.key_for(state)
        entry = core.cache_get(key, self.cache_path)
        if entry and entry.get("questions_version") == jev.QUESTIONS_VERSION:
            try:
                answers = jev.parse_answers(entry.get("answers"))
            except jev.JevError as exc:
                core.log_event(jev.log, "llm_cache_invalid", key=key[:12], error=str(exc), level=30)
            else:
                core.log_event(jev.log, "llm_cache_hit", key=key[:12], model=entry.get("model"))
                return jev.JevResponse(
                    answers=answers,
                    model=str(entry.get("model") or self.model),
                    usage=entry.get("usage") or {},
                    latency_ms=(time.perf_counter() - started) * 1000,
                    engine="llm_cache",
                    source_latency_ms=entry.get("latency_ms"),
                    provider=self.provider,
                )
        core.log_event(jev.log, "llm_cache_miss", key=key[:12], model=self.model)
        live_allowed = jev.mode() != "cache" if allow_live is None else allow_live
        if not live_allowed:
            raise jev.JevUnavailable("cache miss and live calls disabled")
        if not self.key:
            raise jev.JevAuthError("LLM_API_KEY / BEATAPI_API_KEY is not set")
        if allow_live is None and jev.mode() == "auto" and not self.probe():
            raise jev.JevUnavailable("cache miss and network probe failed")
        raw, usage = self._chat(state)
        answers = jev.parse_answers(raw)
        latency_ms = (time.perf_counter() - started) * 1000
        core.cache_put(
            key,
            {
                "provider": self.provider,
                "requested_model": self.model,
                "model": self.model,
                "questions_version": jev.QUESTIONS_VERSION,
                "state": jev.render_state(state),
                "answers": raw,
                "usage": usage,
                "fetched_at": core.iso_now(),
                "latency_ms": round(latency_ms, 2),
            },
            self.cache_path,
        )
        core.log_event(
            jev.log,
            "llm_call",
            key=key[:12],
            model=self.model,
            latency_ms=round(latency_ms, 1),
            usage=usage,
        )
        return jev.JevResponse(
            answers=answers,
            model=self.model,
            usage=usage,
            latency_ms=latency_ms,
            engine="llm_live",
            source_latency_ms=latency_ms,
            provider=self.provider,
        )

    # -------------------------------------------------------------- transport
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def _body(self, state: dict[str, Any], strict: bool) -> dict[str, Any]:
        user = "Scenario:\n" + json.dumps(jev.render_state(state), indent=1)
        if strict:
            user += f"\nReturn ONLY this JSON object: {SHAPE}"
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        }

    def _chat(self, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        last: jev.JevError | None = None
        for attempt in range(2):
            payload = self._body(state, strict=attempt > 0)
            self._pace()
            try:
                body = self._post(payload)
            except jev.JevError as exc:
                text = str(exc)
                if "response_format" in text and "response_format" in json.dumps(payload):
                    payload.pop("response_format", None)
                    try:
                        body = self._post(payload)
                    except jev.JevError as inner:
                        last = inner
                        continue
                else:
                    last = exc
                    continue
            try:
                return self._extract(body)
            except jev.JevError as exc:
                last = exc
                continue
        raise last or jev.JevError("LLM call failed")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        attempts = 0
        while True:
            try:
                body = self.transport(url, payload, headers, jev.TIMEOUT_S)
                if not isinstance(body, dict):
                    raise jev.JevError("LLM returned a non-object response")
                return body
            except jev.JevAuthError:
                raise
            except jev.JevError as exc:
                text = str(exc)
                retryable = "HTTP 429" in text or "HTTP 5" in text or "cannot reach" in text
                if not retryable or attempts == 2:
                    raise
                time.sleep(max(self.min_interval, 0.5 * (2**attempts)))
                attempts += 1

    @staticmethod
    def _content(body: dict[str, Any]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise jev.JevError("LLM response has no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict):
            message = {}
        content = message.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise jev.JevError("LLM response has no text content")
        return content

    def _extract(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        raw = self._content(body)
        answers = build_answers(raw)
        usage = body.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return answers, {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        }


def _json_from_text(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise jev.JevError(f"no JSON object in LLM output: {text[:120]!r}")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise jev.JevError(f"LLM output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise jev.JevError("LLM JSON is not an object")
    return value


def _prob(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise jev.JevError(f"{label}: {value!r} is not numeric") from exc
    return min(1.0, max(0.0, number))


def _score(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise jev.JevError(f"{label}: {value!r} is not numeric") from exc
    return min(3.0, max(0.0, number))


def build_answers(text: str) -> dict[str, Any]:
    """Turn one model JSON object into the typed answer shape jev.parse_answers reads."""
    data = _json_from_text(text)
    out: dict[str, Any] = {}
    for name in ("o2_above_limiting", "flow_above_blowoff", "self_sustaining_60s", "spread_beyond_ignition"):
        if name not in data:
            raise jev.JevError(f"missing {name} in LLM JSON")
        out[name] = {"type": "noul", "noul": _prob(data[name], name)}
    class_raw = data.get("flame_class")
    if isinstance(class_raw, str):
        class_raw = {class_raw: 1.0}
    elif not isinstance(class_raw, dict):
        class_raw = {}
    probabilities = {key: _prob(class_raw.get(key, 0.0), f"flame_class.{key}") for key in jev.CRITERIA}
    total = sum(probabilities.values())
    if total <= 0:
        probabilities = {key: 1.0 / len(jev.CRITERIA) for key in jev.CRITERIA}
    else:
        probabilities = {key: value / total for key, value in probabilities.items()}
    choice = max(probabilities, key=probabilities.get)
    out["flame_class"] = {
        "type": "choice",
        "choice": choice,
        "confidence": probabilities[choice],
        "probabilities": probabilities,
    }
    if "severity" not in data:
        raise jev.JevError("missing severity in LLM JSON")
    severity = _score(data["severity"], "severity")
    out["severity"] = {
        "type": "score",
        "score": severity,
        "confidence": 0.0,
        "legend": {str(i): text for i, text in enumerate(jev.QUESTIONS["severity"]["criteria"])},
        "probabilities": {},
    }
    return out


def build_engine():
    """An explicit ENGINE_PROVIDER wins; otherwise LLM_* selects the wrapper."""
    provider = (os.environ.get("ENGINE_PROVIDER") or "").strip().lower()
    if provider == "llm":
        return LlmEngine()
    if provider:
        return jev.JevEngine()
    if os.environ.get("LLM_BASE_URL") or os.environ.get("LLM_MODEL"):
        return LlmEngine()
    return jev.JevEngine()