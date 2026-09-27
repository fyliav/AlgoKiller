from __future__ import annotations

import json
import os
import re
import time
from itertools import cycle
from threading import Lock
from typing import Any, Iterator

import litellm

_api_key_lock = Lock()
_api_key_rotators: dict[tuple[str, ...], Iterator[str]] = {}

# Client-side read timeout for one model request. This only bounds how long the
# local process waits; upstream reverse proxies enforce their own limits.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 9999.0

# Upper bound for the delay between model request retries.
MAX_RETRY_DELAY_SECONDS = 120.0

_RETRY_AFTER_PATTERN = re.compile(r"retry[-_]after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", re.IGNORECASE)


def message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "".join(text_parts)
    return ""


def clean_text(value: str) -> str:
    """Remove Unicode surrogate code points before sending text to LiteLLM."""
    return value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def clean_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [clean_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {clean_jsonable(key): clean_jsonable(item) for key, item in value.items()}
    return value


def _is_anthropic_model(model: str) -> bool:
    return model.split("/", 1)[0].lower() == "anthropic"


def _is_gpt5_family_model(model: str) -> bool:
    return model.split("/", 1)[0].lower() == "openai"


def is_deepseek(model: str) -> bool:
    model_name = model.rsplit("/", 1)[-1].lower()
    return model_name.startswith("deepseek")


def _is_kimi_model(model: str) -> bool:
    model_name = model.rsplit("/", 1)[-1].lower()
    return model_name.startswith("kimi-")


def api_kwargs(*, api_key: str, api_base: str) -> dict[str, str]:
    kwargs = {}
    rotated_api_key = _next_api_key(api_key)
    if rotated_api_key:
        kwargs["api_key"] = rotated_api_key
    if api_base:
        kwargs["api_base"] = api_base
    return kwargs


def _api_keys(api_key: str) -> tuple[str, ...]:
    return tuple(key.strip() for key in api_key.split(",") if key.strip())


def _next_api_key(api_key: str) -> str:
    keys = _api_keys(api_key)
    if not keys:
        return ""
    if len(keys) == 1:
        return keys[0]
    with _api_key_lock:
        rotator = _api_key_rotators.setdefault(keys, cycle(keys))
        return next(rotator)


def temperature_kwargs(*, model: str, temperature: float) -> dict[str, float]:
    if (_is_gpt5_family_model(model) or _is_kimi_model(model)) and temperature != 1:
        return {}
    return {"temperature": temperature}


def reasoning_effort_kwargs(*, model: str, reasoning_effort: str) -> dict[Any, Any] | dict[str, str | list[str]] | dict[
    str, dict[str, str]]:
    if not reasoning_effort or reasoning_effort.lower() in {"none", "off", "disabled"}:
        return {}
    if _is_kimi_model(model) or is_deepseek(model):
        return {}
    if _is_gpt5_family_model(model):
        return {"reasoning_effort": reasoning_effort, "allowed_openai_params": ["reasoning_effort"]}
    elif _is_anthropic_model(model):
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": reasoning_effort}}

    return {}


def extra_body_kwargs(*, model: str) -> dict[str, dict[str, Any]]:
    if _is_kimi_model(model):
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if is_deepseek(model):
        return {"extra_body": {"thinking": {"type": "enabled"}}}
    return {}


def streaming_enabled() -> bool:
    """Streaming keeps bytes flowing through upstream proxies so their read
    timeout (for example Cloudflare's 120s proxy read timeout -> HTTP 524)
    does not kill long reasoning requests."""
    value = os.getenv("HARNESS_STREAM", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def request_timeout_seconds() -> float:
    raw = os.getenv("HARNESS_REQUEST_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            raw = headers.get("retry-after")
        except Exception:
            raw = None
        if raw:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    match = _RETRY_AFTER_PATTERN.search(str(exc))
    if match:
        return max(0.0, float(match.group(1)))
    return None


def _looks_like_complete_response(result: Any) -> bool:
    if isinstance(result, litellm.ModelResponse):
        return True
    choices = getattr(result, "choices", None)
    if not choices:
        return False
    first = choices[0]
    if isinstance(first, dict):
        return "message" in first
    return hasattr(first, "message")


def _completion_once(**kwargs: Any) -> Any:
    if not streaming_enabled():
        kwargs.setdefault("timeout", request_timeout_seconds())
        return litellm.completion(**kwargs)

    stream_kwargs = dict(kwargs)
    stream_kwargs.setdefault("timeout", request_timeout_seconds())
    stream_kwargs["stream"] = True
    stream_kwargs.setdefault("stream_options", {"include_usage": True})
    result = litellm.completion(**stream_kwargs)
    if _looks_like_complete_response(result) or not hasattr(result, "__iter__"):
        # Some gateways ignore stream=True and answer with a complete response.
        return result
    chunks = list(result)
    if not chunks:
        raise RuntimeError("streaming model response produced no chunks")
    response = litellm.stream_chunk_builder(chunks, messages=kwargs.get("messages"))
    if response is None:
        raise RuntimeError("failed to rebuild a model response from stream chunks")
    return response


def completion_with_retries(*, max_attempts: int, retry_delay_seconds: float = 1.0, **kwargs: Any) -> Any:
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        try:
            return _completion_once(**kwargs)
        except Exception as exc:
            print(exc)
            if _is_non_retryable_model_error(exc) or attempt >= attempts:
                raise
            delay = _retry_delay_seconds(exc=exc, base_delay=retry_delay_seconds, attempt=attempt)
            print(
                "> model_request_retry("
                + json.dumps(
                    {
                        "model": kwargs.get("model"),
                        "attempt": attempt + 1,
                        "max_attempts": attempts,
                        "stream": streaming_enabled(),
                        "retry_in_seconds": round(delay, 1),
                    },
                    ensure_ascii=False,
                )
                + ")"
            )
            if delay > 0:
                time.sleep(delay)

    raise RuntimeError("unreachable model retry state")


def _retry_delay_seconds(*, exc: Exception, base_delay: float, attempt: int) -> float:
    if base_delay <= 0:
        return 0.0
    server_delay = _retry_after_seconds(exc)
    if server_delay is not None:
        return min(server_delay, MAX_RETRY_DELAY_SECONDS)
    return min(base_delay * (2 ** (attempt - 1)), MAX_RETRY_DELAY_SECONDS)

def _is_non_retryable_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "api_key",
        "authentication",
        "model not found",
        "模型选择错误",
        "all available accounts exhausted",
    )
    return any(marker in text for marker in markers)
