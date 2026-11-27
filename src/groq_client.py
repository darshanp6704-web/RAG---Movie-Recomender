"""Shared Groq API wrapper with call logging, reasoning token budget, and strict errors."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import groq
from groq import Groq
from pydantic import BaseModel

from src import config

logger = logging.getLogger("movie_buddy.groq")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class GroqServiceError(RuntimeError):
    """Raised when Groq is unavailable, unconfigured, or returns an invalid response."""

    def __init__(self, user_message: str, status_code: int | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


class GroqCallLog(BaseModel):
    """Telemetry record captured for every Groq API invocation."""

    purpose: str
    model: str
    status_code: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0


def is_reasoning_model(model_name: str) -> bool:
    """Return True if the selected Groq model emits reasoning tokens."""
    lower = model_name.lower()
    return any(tok in lower for tok in ("oss", "qwen", "r1", "reason", "thinking"))


def resolve_max_tokens(model_name: str, requested_max_tokens: int) -> int:
    """Ensure reasoning models always receive at least 1024 max_tokens."""
    if is_reasoning_model(model_name):
        return max(1024, requested_max_tokens)
    return requested_max_tokens


def get_groq_client(api_key: str | None = None) -> Groq:
    """Build a Groq client or fail clearly if GROQ_API_KEY is missing."""
    key = (api_key if api_key is not None else config.GROQ_API_KEY).strip()
    if not key or key == "your_groq_api_key_here":
        raise GroqServiceError(
            "Groq API key is missing. Set a valid GROQ_API_KEY in .env to use Movie Buddy.",
            status_code=None,
        )
    return Groq(api_key=key, timeout=20.0, max_retries=0)


def call_groq_json(
    purpose: str,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.1,
    client: Groq | None = None,
) -> tuple[dict[str, Any], GroqCallLog]:
    """Execute a Groq JSON chat completion, log status/tokens, and fail loudly on errors."""
    target_model = (model or config.GROQ_MODEL).strip()
    target_tokens = resolve_max_tokens(
        target_model, max_tokens or config.GROQ_MAX_TOKENS
    )
    groq_client = client or get_groq_client()

    t0 = time.perf_counter()
    create_kwargs: dict[str, Any] = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": target_tokens,
        "temperature": temperature,
    }
    if is_reasoning_model(target_model):
        create_kwargs["reasoning_effort"] = "low"

    try:
        raw_resp = groq_client.chat.completions.with_raw_response.create(
            **create_kwargs
        )
        status_code = getattr(raw_resp, "status_code", 200)
        parsed = raw_resp.parse()
    except groq.AuthenticationError as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        sc = getattr(exc, "status_code", 401)
        logger.error(
            "Groq call failed | purpose=%s model=%s status=%s latency_ms=%.2f error=%s",
            purpose,
            target_model,
            sc,
            latency_ms,
            exc,
        )
        raise GroqServiceError(
            f"Groq authentication failed (HTTP {sc}). Check GROQ_API_KEY in .env.",
            status_code=sc,
        ) from exc
    except groq.RateLimitError as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        sc = getattr(exc, "status_code", 429)
        logger.error(
            "Groq call failed | purpose=%s model=%s status=%s latency_ms=%.2f error=%s",
            purpose,
            target_model,
            sc,
            latency_ms,
            exc,
        )
        raise GroqServiceError(
            f"Groq rate limit reached (HTTP {sc}). Please wait a moment and try again.",
            status_code=sc,
        ) from exc
    except groq.APIStatusError as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        sc = getattr(exc, "status_code", None)
        logger.error(
            "Groq call failed | purpose=%s model=%s status=%s latency_ms=%.2f error=%s",
            purpose,
            target_model,
            sc,
            latency_ms,
            exc,
        )
        raise GroqServiceError(
            f"Groq API error (HTTP {sc}): unable to process your request right now.",
            status_code=sc,
        ) from exc
    except (groq.APIConnectionError, groq.APITimeoutError) as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.error(
            "Groq connection/timeout failed | purpose=%s model=%s status=None latency_ms=%.2f error=%s",
            purpose,
            target_model,
            latency_ms,
            exc,
        )
        raise GroqServiceError(
            "Unable to reach Groq API (network error or timeout). Please try again.",
            status_code=None,
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    usage = getattr(parsed, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)

    call_log = GroqCallLog(
        purpose=purpose,
        model=target_model,
        status_code=status_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        latency_ms=latency_ms,
    )
    logger.info(
        "Groq call | purpose=%s model=%s status=%s prompt_tokens=%d completion_tokens=%d total_tokens=%d reasoning_tokens=%d latency_ms=%.2f",
        call_log.purpose,
        call_log.model,
        call_log.status_code,
        call_log.prompt_tokens,
        call_log.completion_tokens,
        call_log.total_tokens,
        call_log.reasoning_tokens,
        call_log.latency_ms,
    )

    content = (parsed.choices[0].message.content or "").strip()
    if not content:
        raise GroqServiceError(
            f"Groq returned an empty response (HTTP {status_code}). Please try again.",
            status_code=status_code,
        )

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GroqServiceError(
            f"Groq returned malformed JSON (HTTP {status_code}). Please try again.",
            status_code=status_code,
        ) from exc

    if not isinstance(payload, dict):
        raise GroqServiceError(
            f"Groq returned unexpected JSON structure (HTTP {status_code}).",
            status_code=status_code,
        )

    return payload, call_log
