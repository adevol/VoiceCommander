from __future__ import annotations

import base64
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .settings import APP_DIR, POSTPROCESS_STYLES, Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
WHISPER_DIR = APP_DIR / "whisper.cpp"
WHISPER_MODEL = "ggml-base-q5_1.bin"
WHISPER_MODEL_URL = (
    f"https://huggingface.co/ggerganov/whisper.cpp/resolve/"
    f"f281eb45af861ab5e5297d23694b7d46e090c02c/{WHISPER_MODEL}"
)
WHISPER_MODEL_SHA256 = "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898"
WHISPER_RUNTIME_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
)
WHISPER_RUNTIME_SHA256 = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"
logger = logging.getLogger(__name__)
LoadedModel = tuple[str, Any, Any]


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
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Could not download {destination.name}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_whisper() -> tuple[Path, Path]:
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

    model = WHISPER_DIR / WHISPER_MODEL
    model_marker = WHISPER_DIR / f".model-{WHISPER_MODEL_SHA256}"
    if not (model_marker.exists() and model.exists()):
        _download(WHISPER_MODEL_URL, model, WHISPER_MODEL_SHA256)
        model_marker.touch()
    return executable, model


def load_local_model(local_asr_model: str = "whisper") -> LoadedModel:
    if local_asr_model == "whisper":
        executable, model = _ensure_whisper()
        logger.info("Loaded multilingual Whisper model %s", model)
        return "whisper", executable, model
    if local_asr_model != "nemotron":
        raise RuntimeError(f"Unsupported local ASR model: {local_asr_model}")
    try:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
    except ImportError as error:
        raise RuntimeError("Nemotron requires the 'nvidia' optional dependencies") from error

    logger.info("Loading local ASR model %s", NEMOTRON_MODEL)
    processor = AutoProcessor.from_pretrained(NEMOTRON_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForRNNT.from_pretrained(NEMOTRON_MODEL, dtype=dtype)
    model.to(device)
    model.eval()
    logger.info("Local ASR model loaded on %s with %s", device, dtype)
    return "nemotron", processor, model


def transcribe_local(path: Path, settings: Settings, loaded_model: LoadedModel) -> str:
    kind, first, second = loaded_model
    if kind == "whisper":
        output_base = path.with_suffix(path.suffix + ".whisper")
        output_path = Path(str(output_base) + ".txt")
        # ponytail: reloads 57 MiB per dictation; use the DLL if profiling shows startup lag.
        try:
            result = subprocess.run(
                [
                    str(first),
                    "-m",
                    str(second),
                    "-f",
                    str(path),
                    "-l",
                    settings.language.split("-", 1)[0].lower(),
                    "-otxt",
                    "-of",
                    str(output_base),
                    "-np",
                ],
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

    if kind != "nemotron":
        raise RuntimeError(f"Unsupported loaded ASR model: {kind}")
    import torch
    from transformers.audio_utils import load_audio

    processor, model = first, second
    sample_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(str(path), sampling_rate=sample_rate)
    inputs = processor(audio, sampling_rate=sample_rate, language=settings.language, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True)
    decoded = processor.decode(output.sequences, skip_special_tokens=True)
    text = (" ".join(decoded) if isinstance(decoded, list) else decoded).strip()
    if not text:
        raise RuntimeError("Local ASR returned an empty transcript")
    return text


def _openrouter(path: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("OpenRouter is selected but no API key is configured")
    request = Request(
        f"{OPENROUTER_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(200).decode("utf-8", errors="replace").strip()
        message = f"OpenRouter returned HTTP {error.code}{f': {detail}' if detail else ''}"
        logger.error(message)
        raise RuntimeError(message) from error
    except URLError as error:
        logger.error("OpenRouter request failed: %s", error.reason)
        raise RuntimeError("OpenRouter could not be reached") from error


def transcribe_openrouter(path: Path, settings: Settings, api_key: str) -> str:
    payload = {
        "model": settings.openrouter_asr_model,
        "language": settings.language.split("-", 1)[0],
        "input_audio": {"data": base64.b64encode(path.read_bytes()).decode("ascii"), "format": "wav"},
        "provider": {"data_collection": "deny", "zdr": True},
    }
    text = str(_openrouter("/audio/transcriptions", api_key, payload).get("text", "")).strip()
    if not text:
        raise RuntimeError("OpenRouter returned an empty transcript")
    return text


def transcribe(path: Path, settings: Settings, loaded_model: LoadedModel | None, api_key: str) -> str:
    if settings.asr_provider == "local":
        if loaded_model is None:
            raise RuntimeError("Local ASR model is not loaded")
        return transcribe_local(path, settings, loaded_model)
    return transcribe_openrouter(path, settings, api_key)


def _strength_instruction(strength: int) -> str:
    # Models follow these categories far more reliably than a numeric dial.
    if strength <= 25:
        return "Only fix punctuation and obvious mis-hearings; keep the wording exactly as spoken."
    if strength <= 75:
        return "Lightly clean up the text, keeping the speaker's wording and sentence order."
    return "Rewrite freely for clarity while preserving the meaning."


def postprocess_openrouter(text: str, settings: Settings, api_key: str) -> str:
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
    prompt = "\n".join(rules)
    payload = {
        "model": settings.postprocess_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "provider": {"data_collection": "deny", "zdr": True},
    }
    response = _openrouter("/chat/completions", api_key, payload)
    try:
        result = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as error:
        raise RuntimeError("OpenRouter returned an invalid post-processing response") from error
    if not result:
        raise RuntimeError("OpenRouter returned empty post-processed text")
    return result


def run_pipeline(
    path: Path,
    settings: Settings,
    loaded_model: LoadedModel | None,
    api_key: str,
) -> str:
    raw = transcribe(path, settings, loaded_model, api_key)
    if settings.postprocess_strength == 0:
        return raw
    try:
        return postprocess_openrouter(raw, settings, api_key)
    except Exception:
        logger.exception("Post-processing failed; using raw transcript")
        return raw
