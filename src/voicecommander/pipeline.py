from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from time import monotonic
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .local_asr import LocalAsrEngine, PreviewText
from .prompts import postprocess_prompt
from .settings import Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1"
logger = logging.getLogger(__name__)


def _openrouter_request(path: str, api_key: str, payload: dict[str, Any]) -> Request:
    if not api_key:
        raise RuntimeError("OpenRouter is selected but no API key is configured")
    return Request(
        f"{OPENROUTER_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )


def _openrouter(path: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = _openrouter_request(path, api_key, payload)
    last: OSError | None = None
    for _ in range(2):
        try:
            with urlopen(request, timeout=300) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(200).decode("utf-8", errors="replace").strip()
            message = f"OpenRouter returned HTTP {error.code}{f': {detail}' if detail else ''}"
            logger.error(message)
            raise RuntimeError(message) from error
        except OSError as error:
            last = error
            logger.warning("OpenRouter connection failed: %s", error)
    logger.error("OpenRouter request failed twice: %s", last)
    raise RuntimeError(f"OpenRouter could not be reached: {last}") from last


def _message_text(response: dict[str, Any], what: str) -> str:
    try:
        text = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise RuntimeError(f"OpenRouter returned an invalid {what} response") from error
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
    return _message_text(_openrouter("/chat/completions", api_key, payload), "transcript")


def postprocess_openrouter(
    text: str,
    settings: Settings,
    api_key: str,
    markdown: bool = False,
) -> str:
    return _message_text(
        _openrouter("/chat/completions", api_key, _postprocess_payload(text, settings, markdown)),
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
    payload = _postprocess_payload(text, settings, markdown) | {"stream": True}
    request = _openrouter_request("/chat/completions", api_key, payload)
    result = ""
    emitted = ""
    last_update = monotonic()
    done = False
    try:
        with urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if not isinstance(chunk, dict):
                    raise RuntimeError("OpenRouter returned an invalid stream event")
                if chunk.get("error"):
                    raise RuntimeError(f"OpenRouter stream failed: {chunk['error']}")
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                try:
                    content = choices[0]["delta"].get("content")
                except (KeyError, TypeError, AttributeError) as error:
                    raise RuntimeError("OpenRouter returned an invalid stream event") from error
                if content is None:
                    continue
                if not isinstance(content, str):
                    raise RuntimeError("OpenRouter returned an invalid stream event")
                result += content
                now = monotonic()
                if now - last_update >= 0.05:
                    on_update(result)
                    emitted = result
                    last_update = now
    except HTTPError as error:
        detail = error.read(200).decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"OpenRouter returned HTTP {error.code}{f': {detail}' if detail else ''}"
        ) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"OpenRouter stream failed: {error}") from error
    if not done:
        raise RuntimeError("OpenRouter stream ended before completion")
    result = result.strip()
    if not result:
        raise RuntimeError("OpenRouter returned an empty post-processed text")
    if result != emitted:
        on_update(result)
    return result


def run_pipeline(
    path: Path,
    settings: Settings,
    loaded_model: LocalAsrEngine | None,
    api_key: str,
    markdown: bool = False,
    on_refine: Callable[[str], None] | None = None,
    preview: PreviewText | None = None,
) -> str:
    if settings.asr_provider == "local":
        if loaded_model is None:
            raise RuntimeError("Local ASR model is not loaded")
        raw = loaded_model.transcribe(path, settings, preview)
    else:
        raw = transcribe_openrouter(path, settings, api_key)
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
