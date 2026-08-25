from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .local_asr import LocalAsrEngine
from .settings import POSTPROCESS_STYLES, Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1"
logger = logging.getLogger(__name__)


def _openrouter(path: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST a payload to OpenRouter.

    Args:
        path: The route to call, appended to `OPENROUTER_URL`.
        api_key: The OpenRouter key.
        payload: The request body, serialised as JSON.

    Returns:
        The decoded JSON response.

    Raises:
        RuntimeError: If the key is missing, the response is an HTTP error, or the
            connection fails on both attempts.
    """
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


def _vocabulary_hint(vocabulary: str) -> str:
    return f" These words may appear and are spelled like this: {vocabulary}." if vocabulary else ""


def transcribe_openrouter(path: Path, settings: Settings, api_key: str) -> str:
    """Transcribe a WAV file with an OpenRouter model.

    Args:
        path: The WAV file to transcribe.
        settings: Supplies the model, language, and custom vocabulary.
        api_key: The OpenRouter key.

    Returns:
        The transcript, stripped of surrounding whitespace.

    Raises:
        RuntimeError: If the request fails or the reply holds no usable text.
    """
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
                            + _vocabulary_hint(settings.vocabulary)
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


def _strength_instruction(strength: int) -> str:
    """Turn the editing-strength slider into an instruction.

    Args:
        strength: The slider position, from 0 to 100.

    Returns:
        One of three instructions, for light, moderate, or free editing.
    """
    if strength <= 25:
        return "Only fix punctuation and obvious mis-hearings; keep the wording exactly as spoken."
    if strength <= 75:
        return "Lightly clean up the text, keeping the speaker's wording and sentence order."
    return "Rewrite freely for clarity while preserving the meaning."


def postprocess_openrouter(
    text: str,
    settings: Settings,
    api_key: str,
    markdown: bool = False,
) -> str:
    """Edit a transcript with an OpenRouter model.

    Args:
        text: The raw transcript.
        settings: Supplies the model, editing strength, style, language, and custom
            vocabulary.
        api_key: The OpenRouter key.
        markdown: Whether to turn the dictation into finished Markdown rather than
            lightly editing it as prose.

    Returns:
        The edited text.

    Raises:
        RuntimeError: If the request fails or the reply holds no usable text.
    """
    number_format_rule = (
        "Write numbers, dates, and times naturally for the language of the dictated text."
        if settings.language == "auto"
        else f"Write numbers, dates, and times naturally for the {settings.language} locale."
    )
    if markdown:
        rules = [
            "Turn this dictated description into the finished Markdown content the speaker intends;"
            " do not merely transcribe the description.",
            "Use valid, conventional Markdown with paragraphs, headings, lists, blockquotes, tables,"
            " and code blocks when the requested content calls for them.",
            "Interpret spoken layout and hierarchy instructions instead of including those instructions"
            " in the result.",
            "Format mathematical expressions as LaTeX: use $...$ for inline math and $$...$$ for"
            " display math.",
            "Preserve the speaker's meaning and supplied facts; do not invent content.",
            f"Style: {POSTPROCESS_STYLES[settings.postprocess_style]}",
            "Reply in the language of the dictated text.",
            'Apply spoken self-corrections: when the speaker says "scratch that" or "correction"'
            " or restarts a phrase, keep only the corrected version and never write the command itself.",
            number_format_rule,
            "Return only raw Markdown, without commentary or an outer Markdown code fence.",
        ]
    else:
        rules = [
            "Edit this dictated text without changing its meaning or adding information.",
            _strength_instruction(settings.postprocess_strength),
            f"Style: {POSTPROCESS_STYLES[settings.postprocess_style]}",
            "Reply in the language of the dictated text.",
            'Apply spoken self-corrections: when the speaker says "scratch that" or "correction"'
            " or restarts a phrase, keep only the corrected version and never write the command itself.",
            'Convert spoken commands ("new paragraph", "bullet point", "quote ... unquote") and'
            " formatting cues into paragraphs, bullets, numbered lists, and punctuation.",
            number_format_rule,
            "Return only the finished text.",
        ]
    if settings.vocabulary:
        rules.insert(2, f"Keep these terms exactly as spelled here: {settings.vocabulary}")
    prompt = "\n".join(rules)
    payload = {
        "model": settings.postprocess_model,
        "messages": [
            {"role": "system", "content": prompt},
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
        raw = loaded_model.start(settings).finish(path)
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
