from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .settings import APP_DIR, POSTPROCESS_STYLES, WHISPER_MODELS, WHISPER_REVISION, Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1"
WHISPER_DIR = APP_DIR / "whisper.cpp"
WHISPER_RUNTIME_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
)
WHISPER_RUNTIME_SHA256 = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"
logger = logging.getLogger(__name__)
LoadedModel = tuple[Path, Path]


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=300) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"Downloaded file failed verification: {destination.name}")
        temporary.replace(destination)
    except OSError as error:
        raise RuntimeError(f"Could not download {destination.name}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_whisper(local_asr_model: str) -> tuple[Path, Path]:
    """Return the whisper.cpp runtime and model, downloading either if absent.

    Args:
        local_asr_model: A key of `WHISPER_MODELS`.

    Returns:
        The paths to the `whisper-cli` executable and the model file.

    Raises:
        RuntimeError: If a download fails verification or the runtime archive is
            missing files.
    """
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    executable = WHISPER_DIR / "whisper-cli.exe"
    runtime_marker = WHISPER_DIR / f".runtime-{WHISPER_RUNTIME_SHA256}"
    if not (runtime_marker.exists() and executable.exists()):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "whisper.zip"
            _download(WHISPER_RUNTIME_URL, archive_path, WHISPER_RUNTIME_SHA256)
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    name = Path(member.filename).name
                    if not member.is_dir() and (
                        name == "whisper-cli.exe" or name.lower().endswith(".dll")
                    ):
                        with archive.open(member) as source, (WHISPER_DIR / name).open("wb") as output:
                            shutil.copyfileobj(source, output)
        if not all(
            (WHISPER_DIR / name).exists()
            for name in ("whisper-cli.exe", "whisper.dll", "ggml.dll")
        ):
            raise RuntimeError("The whisper.cpp runtime archive was incomplete")
        runtime_marker.touch()

    filename, expected_sha256 = WHISPER_MODELS[local_asr_model]
    model = WHISPER_DIR / filename
    if not model.exists():
        url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_REVISION}/{filename}"
        _download(url, model, expected_sha256)
    return executable, model


def load_local_model(local_asr_model: str = "base") -> LoadedModel:
    if local_asr_model not in WHISPER_MODELS:
        raise RuntimeError(f"Unsupported local ASR model: {local_asr_model}")
    executable, model = _ensure_whisper(local_asr_model)
    logger.info("Loaded multilingual Whisper model %s", model)
    return executable, model


def transcribe_local(path: Path, settings: Settings, loaded_model: LoadedModel) -> str:
    """Transcribe a WAV file with whisper.cpp.

    Args:
        path: The WAV file to transcribe.
        settings: Supplies the language, custom vocabulary, and recording limit that
            sets the subprocess timeout.
        loaded_model: The executable and model paths from `load_local_model`.

    Returns:
        The transcript, stripped of surrounding whitespace.

    Raises:
        RuntimeError: If whisper.cpp exits non-zero or returns an empty transcript.
    """
    executable, model = loaded_model
    output_base = path.with_suffix(path.suffix + ".whisper")
    output_path = Path(str(output_base) + ".txt")
    command = [
        str(executable),
        "-m",
        str(model),
        "-f",
        str(path),
        "-l",
        settings.language.split("-", 1)[0].lower(),
        "-t",
        str(min(8, os.cpu_count() or 4)),
        "-otxt",
        "-of",
        str(output_base),
        "-np",
    ]
    if settings.vocabulary:
        command += ["--prompt", settings.vocabulary, "--carry-initial-prompt"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(300, settings.max_seconds * 4),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            detail = result.stderr.strip()[-500:]
            raise RuntimeError(f"whisper.cpp failed{f': {detail}' if detail else ''}")
        text = output_path.read_text(encoding="utf-8").strip()
    finally:
        output_path.unlink(missing_ok=True)
    if not text:
        raise RuntimeError("Local ASR returned an empty transcript")
    return text


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
    payload = {
        "model": settings.openrouter_asr_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Transcribe this {settings.language} audio verbatim."
                        " Reply with the transcript only, with no commentary."
                        + _vocabulary_hint(settings.vocabulary),
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


def transcribe(path: Path, settings: Settings, loaded_model: LoadedModel | None, api_key: str) -> str:
    if settings.asr_provider == "local":
        if loaded_model is None:
            raise RuntimeError("Local ASR model is not loaded")
        return transcribe_local(path, settings, loaded_model)
    return transcribe_openrouter(path, settings, api_key)


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
            f"Write numbers, dates, and times naturally for the {settings.language} locale.",
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
            f"Write numbers, dates, and times naturally for the {settings.language} locale.",
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
    loaded_model: LoadedModel | None,
    api_key: str,
    markdown: bool = False,
) -> str:
    raw = transcribe(path, settings, loaded_model, api_key)
    if markdown:
        return postprocess_openrouter(raw, settings, api_key, markdown=True)
    if settings.postprocess_strength == 0:
        return raw
    try:
        return postprocess_openrouter(raw, settings, api_key)
    except Exception:
        logger.exception("Post-processing failed; using raw transcript")
        return raw
