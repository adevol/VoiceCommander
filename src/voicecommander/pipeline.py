from __future__ import annotations

import base64
import logging
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import httpx
from openrouter import OpenRouter
from openrouter.components import ChatResult
from openrouter.errors import OpenRouterError, ResponseValidationError

from .prompts import postprocess_prompt
from .settings import Settings

logger = logging.getLogger(__name__)


@contextmanager
def _openrouter_client(api_key: str):
    if not api_key:
        raise RuntimeError("OpenRouter is selected but no API key is configured")
    try:
        with OpenRouter(api_key=api_key, timeout_ms=300_000, retry_config=None) as client:
            yield client
    except ResponseValidationError as error:
        raise RuntimeError("OpenRouter returned an invalid response") from error
    except OpenRouterError as error:
        detail = error.body[:200].strip()
        raise RuntimeError(f"OpenRouter returned HTTP {error.status_code}: {detail}") from error
    except httpx.TransportError as error:
        raise RuntimeError(f"OpenRouter could not be reached: {error}") from error


def _openrouter(api_key: str, payload: dict[str, Any]) -> ChatResult:
    with _openrouter_client(api_key) as client:
        for attempt in range(2):
            try:
                return client.chat.send(**payload, stream=False)
            except httpx.TransportError as error:
                logger.warning("OpenRouter connection failed: %s", error)
                if attempt:
                    raise


def _message_text(response: ChatResult, what: str) -> str:
    content = response.choices[0].message.content if response.choices else None
    text = content.strip() if isinstance(content, str) else ""
    if not text:
        raise RuntimeError(f"OpenRouter returned an empty {what}")
    return text


def transcribe_openrouter(path: Path, settings: Settings, api_key: str) -> str:
    language_instruction = (
        "Detect the spoken language and transcribe the audio verbatim."
        if settings.language == "auto"
        else f"Transcribe this {settings.language} audio verbatim."
    )
    payload = {
        "model": settings.openrouter_asr_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            language_instruction
                            + " Reply with the transcript only, with no commentary."
                            + (
                                " These words may appear and are spelled like this: "
                                f"{settings.vocabulary}."
                                if settings.vocabulary
                                else ""
                            )
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "provider": {"data_collection": "deny"},
    }
    return _message_text(_openrouter(api_key, payload), "transcript")


def postprocess_openrouter(
    text: str,
    settings: Settings,
    api_key: str,
    markdown: bool = False,
) -> str:
    return _message_text(
        _openrouter(api_key, _postprocess_payload(text, settings, markdown)),
        "post-processed text",
    )


def _postprocess_payload(text: str, settings: Settings, markdown: bool) -> dict[str, Any]:
    return {
        "model": settings.postprocess_model,
        "messages": [
            {"role": "system", "content": postprocess_prompt(settings, markdown)},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "provider": {"data_collection": "deny"},
    }


def postprocess_openrouter_stream(
    text: str,
    settings: Settings,
    api_key: str,
    on_update: Callable[[str], None],
    markdown: bool = False,
) -> str:
    payload = _postprocess_payload(text, settings, markdown)
    result = ""
    emitted = ""
    last_update = monotonic()
    done = False
    with _openrouter_client(api_key) as client:
        with client.chat.send(**payload, stream=True) as stream:
            for chunk in stream:
                if chunk.error:
                    raise RuntimeError(f"OpenRouter stream failed: {chunk.error.message}")
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason == "error":
                    raise RuntimeError("OpenRouter stream failed")
                # The SDK consumes [DONE]; require the model's completion event.
                done = done or choice.finish_reason is not None
                content = choice.delta.content
                if not isinstance(content, str):
                    continue
                result += content
                now = monotonic()
                if now - last_update >= 0.05:
                    on_update(result)
                    emitted = result
                    last_update = now
    if not done:
        raise RuntimeError("OpenRouter stream ended before completion")
    result = result.strip()
    if not result:
        raise RuntimeError("OpenRouter returned an empty post-processed text")
    if result != emitted:
        on_update(result)
    return result


def refine_transcript(
    raw: str,
    settings: Settings,
    api_key: str,
    markdown: bool = False,
    on_refine: Callable[[str], None] | None = None,
) -> str:
    if not markdown and settings.postprocess_strength == 0:
        return raw
    if on_refine is not None:
        on_refine("")
        try:
            return postprocess_openrouter_stream(
                raw, settings, api_key, on_refine, markdown=markdown
            )
        except Exception:
            logger.exception("Streaming post-processing failed; retrying without streaming")
    try:
        result = postprocess_openrouter(raw, settings, api_key, markdown=markdown)
        if on_refine is not None:
            on_refine(result)
        return result
    except Exception:
        if markdown:
            raise
        logger.exception("Post-processing failed; using raw transcript")
        return raw
