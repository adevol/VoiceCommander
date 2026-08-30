from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .local_asr import LocalAsrEngine
from .prompts import postprocess_prompt
from .settings import Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1"
logger = logging.getLogger(__name__)


def _openrouter(path: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("OpenRouter is selected but no API key is configured")
    request = Request(
        f"{OPENROUTER_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
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
    payload = {
        "model": settings.postprocess_model,
        "messages": [
            {"role": "system", "content": postprocess_prompt(settings, markdown)},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "provider": {"data_collection": "deny"},
    }
    return _message_text(_openrouter("/chat/completions", api_key, payload), "post-processed text")


def run_pipeline(
    path: Path,
    settings: Settings,
    loaded_model: LocalAsrEngine | None,
    api_key: str,
    markdown: bool = False,
) -> str:
    if settings.asr_provider == "local":
        if loaded_model is None:
            raise RuntimeError("Local ASR model is not loaded")
        raw = loaded_model.transcribe(path, settings)
    else:
        raw = transcribe_openrouter(path, settings, api_key)
    if markdown:
        return postprocess_openrouter(raw, settings, api_key, markdown=True)
    if settings.postprocess_strength == 0:
        return raw
    try:
        return postprocess_openrouter(raw, settings, api_key)
    except Exception:
        logger.exception("Post-processing failed; using raw transcript")
        return raw
