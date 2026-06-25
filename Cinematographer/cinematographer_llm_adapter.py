"""Local LLM adapter for Cinematographer board selection and evaluation.

Standalone LLM adapter copy — does NOT import from the main directory.
Uses AnyLLM when available.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Optional

try:
    from any_llm import AnyLLM
    from any_llm.constants import LLMProvider
except Exception:  # pragma: no cover
    AnyLLM = None
    LLMProvider = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


# ── noisy log filters ────────────────────────────────────────────────
class _ThinkingPartFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "non-text parts in the response" not in record.getMessage()


class _AFCFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "AFC is enabled" not in record.getMessage()


logging.getLogger("google_genai.types").addFilter(_ThinkingPartFilter())
logging.getLogger("google_genai.models").addFilter(_AFCFilter())


# ── readiness check ──────────────────────────────────────────────────
MODEL_LANGUAGE_POLICY = (
    "Language policy: all instructions, field descriptions, free-form reasons, "
    "and natural-language output must be written in English."
)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def _contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def _iter_model_text(value: Any, path: str = "payload"):
    if isinstance(value, str):
        if value.startswith("data:"):
            return
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_model_text(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_model_text(item, f"{path}[{index}]")


def _assert_english_model_io(messages: list[dict[str, Any]]) -> None:
    for path, text in _iter_model_text(messages, "messages"):
        if _contains_cjk(text):
            raise ValueError(f"model_input_must_be_english: cjk_text_found_at_{path}")


def _with_language_policy(system_prompt: str) -> str:
    if MODEL_LANGUAGE_POLICY in system_prompt:
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n{MODEL_LANGUAGE_POLICY}"


def llm_ready(*, model: str, api_key: str) -> bool:
    return bool((AnyLLM or OpenAI) and model and api_key)


def _openai_base_url(api_base: Optional[str]) -> Optional[str]:
    if not api_base:
        return None
    base = str(api_base).rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


# ── image encoding ───────────────────────────────────────────────────
def image_path_to_data_url(path: str | Path) -> str:
    source = Path(path)
    mime_type, _ = mimetypes.guess_type(str(source))
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime_type or 'image/png'};base64,{encoded}"


# ── client lifecycle ─────────────────────────────────────────────────
def _close_client(llm: Any) -> None:
    client = getattr(llm, "client", None)
    if client is None:
        return
    if hasattr(client, "close"):
        try:
            client.close()
        except Exception:
            pass
    close_async = None
    if hasattr(client, "aio") and hasattr(client.aio, "aclose"):
        close_async = client.aio.aclose
    elif hasattr(client, "_api_client") and hasattr(client._api_client, "aclose"):
        close_async = client._api_client.aclose
    if close_async is None:
        return
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(close_async())
        finally:
            loop.close()
    except Exception:
        pass


# ── raw completion ───────────────────────────────────────────────────
def completion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client_args: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    _assert_english_model_io(messages)
    if AnyLLM is not None and LLMProvider is not None:
        if provider is None:
            provider_key, model_id = AnyLLM.split_model_provider(model)
        else:
            provider_key = LLMProvider.from_string(provider)
            model_id = model

        if provider_key == LLMProvider.from_string("gemini") and api_base:
            if client_args is None:
                client_args = {}
            http_options = client_args.setdefault("http_options", {})
            if "base_url" not in http_options:
                http_options["base_url"] = api_base
                http_options.setdefault("headers", {})["x-goog-api-key"] = api_key

        llm = AnyLLM.create(provider_key, api_key=api_key, api_base=api_base, **(client_args or {}))
        try:
            return llm.completion(model=model_id, messages=messages, **kwargs)
        finally:
            _close_client(llm)

    if OpenAI is None:
        raise RuntimeError("No compatible LLM runtime is available. Install openai or legacy any_llm.")

    request_timeout = kwargs.pop("request_timeout_seconds", None)
    client = OpenAI(api_key=api_key, base_url=_openai_base_url(api_base), timeout=request_timeout or 120.0)
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


# ── response parsing ─────────────────────────────────────────────────
def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    try:
        return str(response.choices[0].message.content or "")
    except Exception:
        return str(response)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json|JSON)?\s*([\s\S]*?)\s*```$", stripped)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _extract_first_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            _, end_index = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end_index].strip()
    return ""


def _parse_json_candidates(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    candidates: list[str] = []
    raw = text.strip()
    if raw:
        candidates.append(raw)
    stripped = _strip_code_fence(raw)
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    extracted = _extract_first_json_object(stripped or raw)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    last_error: Optional[str] = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"llm_invalid_json: {exc}"
            continue
        if isinstance(payload, dict):
            return payload, None
        last_error = f"llm_invalid_json_type: expected object got {type(payload).__name__}"
    return None, last_error or "llm_invalid_json: no_decodable_json_object_found"


def _repair_json_response(
    *,
    model: str,
    raw_response_text: str,
    api_key: str,
    api_base: str,
    provider: str,
) -> tuple[Optional[dict[str, Any]], Optional[str], str]:
    repair_prompt = (
        "Convert the following assistant reply into one strict JSON object.\n"
        "Rules:\n"
        "- Return raw JSON only\n"
        "- Do not use markdown fences\n"
        "- Preserve the original content as much as possible\n"
        "- If some field is uncertain, keep the field but use an empty object, empty list, null, or a reasonable scalar placeholder\n"
        "- The top-level result must be a JSON object\n"
    )
    try:
        response = completion(
            model=model,
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            messages=[
                {"role": "system", "content": _with_language_policy(repair_prompt)},
                {"role": "user", "content": raw_response_text},
            ],
            reasoning_effort="medium",
        )
    except Exception as exc:
        return None, f"llm_repair_call_failed: {exc}", ""

    repaired_text = _extract_response_text(response).strip()
    if not repaired_text:
        return None, "llm_repair_empty_response", repaired_text
    payload, error = _parse_json_candidates(repaired_text)
    return payload, error, repaired_text


# ── high-level JSON call ─────────────────────────────────────────────
def call_json_response(
    *,
    model: str,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    api_key: str,
    api_base: str,
    provider: str,
    reasoning_effort: str = "high",
    retry_count: int = 0,
    retry_backoff_seconds: float = 1.0,
) -> tuple[Optional[dict[str, Any]], Optional[str], str]:
    """Call one LLM request and parse a JSON object from the reply."""

    if not llm_ready(model=model, api_key=api_key):
        return None, "llm_unavailable", ""

    response = None
    last_exc: Exception | None = None
    attempts = max(1, int(retry_count) + 1)
    for attempt_index in range(attempts):
        try:
            response = completion(
                model=model,
                provider=provider,
                api_key=api_key,
                api_base=api_base,
                messages=[
                    {"role": "system", "content": _with_language_policy(system_prompt)},
                    {"role": "user", "content": user_content},
                ],
                reasoning_effort=reasoning_effort,
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt_index >= attempts - 1:
                break
            time.sleep(max(0.0, retry_backoff_seconds) * (2 ** attempt_index))
    if last_exc is not None:
        return None, f"llm_transport_error: {last_exc}", ""

    response_text = _extract_response_text(response).strip()
    if not response_text:
        return None, "llm_parse_error: empty_response", response_text
    payload, error = _parse_json_candidates(response_text)
    if payload is not None:
        return payload, None, response_text

    repaired_payload, repaired_error, repaired_text = _repair_json_response(
        model=model,
        raw_response_text=response_text,
        api_key=api_key,
        api_base=api_base,
        provider=provider,
    )
    if repaired_payload is not None:
        return repaired_payload, None, repaired_text or response_text
    return None, f"llm_parse_error: {repaired_error or error}", response_text
