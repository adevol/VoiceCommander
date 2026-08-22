"""Local speech recognition behind one small, warm-model interface."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import wave
import zipfile
from array import array
from functools import partial
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

from .settings import (
    APP_DIR,
    COHERE_MODELS,
    ONNX_PARAKEET_MODELS,
    PARAKEET_CPP_MODELS,
    WHISPER_MODELS,
    WHISPER_REVISION,
    Settings,
)

WHISPER_DIR = APP_DIR / "whisper.cpp"
WHISPER_RUNTIME_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
)
WHISPER_RUNTIME_SHA256 = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"
PARAKEET_DIR = APP_DIR / "parakeet"
PARAKEET_SNAPSHOTS = {
    "parakeet-tdt-v3": (
        "nemo-parakeet-tdt-0.6b-v3",
        "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    ),
    "parakeet-tdt-v2": (
        "nemo-parakeet-tdt-0.6b-v2",
        "istupakov/parakeet-tdt-0.6b-v2-onnx",
        "b733f6167f9653fef3cbcffbf101683a5d68b1dd",
    ),
}
PARAKEET_FILES = (
    "config.json",
    "vocab.txt",
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
)
PARAKEET_CPP_DIR = APP_DIR / "parakeet.cpp"
PARAKEET_CPP_RUNTIME_URL = (
    "https://github.com/mudler/parakeet.cpp/releases/download/v0.5.0/"
    "parakeet-v0.5.0-bin-win-vulkan-x64.zip"
)
PARAKEET_CPP_RUNTIME_SHA256 = (
    "717c416fab299755e8140137e3a0115121ce1acb6379d13c60f2f0613f6c13a3"
)
PARAKEET_CPP_REVISION = "bf0af9f425fa01809cadec671b3cb672709d13e9"
PARAKEET_CPP_FILES = {
    "parakeet-flash": (
        "realtime_eou_120m-v1-q4_k.gguf",
        "ac9109d0e422bd8aafa899c0f58e1938f4a2846838797a29c04f6a8729033c3c",
    ),
    "nemotron-3.5": (
        "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
        "5ad85eb3f3014c1a300d67b7ccbd23c38c4c952405cbe33a861e19fb2775e84b",
    ),
}
COHERE_DIR = APP_DIR / "cohere"
COHERE_FILENAME = "cohere-transcribe-03-2026-Q5_K_M.gguf"
COHERE_MODEL_URL = (
    "https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/"
    f"resolve/main/{COHERE_FILENAME}"
)
COHERE_MODEL_SHA256 = "14d02f1ad6dd77b3a60f82639879012c3adb4fe25c50a5a47a2c4c661daf1558"
logger = logging.getLogger(__name__)
LocalModel = Callable[[Path, Settings], str]


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


def _ensure_file(path: Path, url: str, sha256: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _download(url, path, sha256)
    return path


def _ensure_runtime(
    directory: Path,
    executable_name: str,
    url: str,
    sha256: str,
    required: tuple[str, ...] = (),
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / executable_name
    marker = directory / f".runtime-{sha256}"
    if marker.exists() and executable.exists():
        return executable

    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "runtime.zip"
        _download(url, archive_path, sha256)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = Path(member.filename).name
                if not member.is_dir() and (
                    name == executable_name or name.lower().endswith(".dll")
                ):
                    with archive.open(member) as source, (directory / name).open(
                        "wb"
                    ) as output:
                        shutil.copyfileobj(source, output)

    if not all((directory / name).exists() for name in (executable_name, *required)):
        raise RuntimeError("The local ASR runtime archive was incomplete")
    marker.touch()
    return executable


def _ensure_whisper(local_asr_model: str) -> tuple[Path, Path]:
    """Return the whisper.cpp runtime and model, downloading either if absent."""
    executable = _ensure_runtime(
        WHISPER_DIR,
        "whisper-cli.exe",
        WHISPER_RUNTIME_URL,
        WHISPER_RUNTIME_SHA256,
        ("whisper.dll", "ggml.dll"),
    )
    filename, expected_sha256 = WHISPER_MODELS[local_asr_model]
    model = _ensure_file(
        WHISPER_DIR / filename,
        f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_REVISION}/{filename}",
        expected_sha256,
    )
    return executable, model


def _ensure_parakeet_cpp(local_asr_model: str) -> tuple[Path, Path]:
    """Return the native parakeet.cpp runtime and selected verified model."""
    executable = _ensure_runtime(
        PARAKEET_CPP_DIR,
        "parakeet-cli.exe",
        PARAKEET_CPP_RUNTIME_URL,
        PARAKEET_CPP_RUNTIME_SHA256,
    )
    filename, expected_sha256 = PARAKEET_CPP_FILES[local_asr_model]
    model = _ensure_file(
        PARAKEET_CPP_DIR / filename,
        (
            "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/"
            f"{PARAKEET_CPP_REVISION}/{filename}"
        ),
        expected_sha256,
    )
    return executable, model


def _run(
    command: list[str], runtime: str, max_seconds: int
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(300, max_seconds * 4),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        detail = result.stderr.strip()[-500:]
        raise RuntimeError(f"{runtime} failed{f': {detail}' if detail else ''}")
    return result


def _transcript(text: str) -> str:
    text = text.strip()
    if not text:
        raise RuntimeError("Local ASR returned an empty transcript")
    return text


def _transcribe_whisper(
    executable: Path, model: Path, path: Path, settings: Settings
) -> str:
    output_base = path.with_suffix(path.suffix + ".whisper")
    output_path = Path(str(output_base) + ".txt")
    language = (
        settings.language
        if settings.language == "auto"
        else settings.language.split("-", 1)[0].lower()
    )
    command = [
        str(executable),
        "-m",
        str(model),
        "-f",
        str(path),
        "-l",
        language,
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
        _run(command, "whisper.cpp", settings.max_seconds)
        text = output_path.read_text(encoding="utf-8")
    finally:
        output_path.unlink(missing_ok=True)
    return _transcript(text)


def _transcribe_parakeet(
    recognizer: Any, english_only: bool, path: Path, settings: Settings
) -> str:
    if english_only and settings.language not in {"auto", "en-US"}:
        raise RuntimeError("Parakeet TDT v2 supports English only")
    try:
        text = recognizer.recognize(str(path))
    except Exception as error:
        raise RuntimeError(f"Parakeet transcription failed: {error}") from error
    return _transcript(text)


def _transcribe_parakeet_cpp(
    executable: Path,
    model: Path,
    english_only: bool,
    path: Path,
    settings: Settings,
) -> str:
    if english_only and settings.language not in {"auto", "en-US"}:
        raise RuntimeError("Parakeet Flash supports English only")
    command = [
        str(executable),
        "transcribe",
        "--model",
        str(model),
        "--input",
        str(path),
        "--json",
    ]
    if not english_only:
        command += ["--lang", settings.language]
    result = _run(command, "parakeet.cpp", settings.max_seconds)
    try:
        text = str(json.loads(result.stdout)["text"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("parakeet.cpp returned an invalid transcript") from error
    return _transcript(text)


def _read_float32_wav(path: Path) -> array[float]:
    try:
        with wave.open(str(path), "rb") as recording:
            if (
                recording.getnchannels() != 1
                or recording.getframerate() != 16_000
                or recording.getsampwidth() != 2
            ):
                raise RuntimeError("Cohere requires a 16 kHz mono 16-bit WAV recording")
            frames = recording.readframes(recording.getnframes())
    except (OSError, wave.Error) as error:
        raise RuntimeError(f"Could not read recording: {error}") from error
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return array("f", (sample / 32768.0 for sample in samples))


def _transcribe_cohere(model: Any, path: Path, settings: Settings) -> str:
    pcm = _read_float32_wav(path)
    language = (
        None
        if settings.language == "auto"
        else settings.language.split("-", 1)[0].lower()
    )
    try:
        with model.session() as session:
            result = session.run(pcm, language=language, timestamps="none")
    except Exception as error:
        raise RuntimeError(f"Cohere transcription failed: {error}") from error
    return _transcript(result.text)


def _load_parakeet(local_asr_model: str) -> LocalModel:
    try:
        import onnx_asr
        import onnxruntime
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Parakeet support is not installed") from error

    model_name, repo_id, revision = PARAKEET_SNAPSHOTS[local_asr_model]
    model_dir = PARAKEET_DIR / local_asr_model
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=model_dir,
            allow_patterns=PARAKEET_FILES,
        )
    except Exception as error:
        raise RuntimeError(f"Could not load {local_asr_model}: {error}") from error

    provider_choices = [["CPUExecutionProvider"]]
    if "DmlExecutionProvider" in onnxruntime.get_available_providers():
        provider_choices.insert(0, ["DmlExecutionProvider", "CPUExecutionProvider"])
    for providers in provider_choices:
        try:
            recognizer = onnx_asr.load_model(
                model_name,
                model_dir,
                quantization="int8",
                providers=providers,
            )
            break
        except Exception as error:
            if providers[0] == "CPUExecutionProvider":
                raise RuntimeError(f"Could not load {local_asr_model}: {error}") from error
            logger.warning(
                "DirectML could not load %s; retrying on CPU: %s",
                local_asr_model,
                error,
            )
    logger.info(
        "Loaded %s from pinned ONNX snapshot %s using %s",
        local_asr_model,
        revision,
        providers[0],
    )
    return partial(
        _transcribe_parakeet,
        recognizer,
        local_asr_model == "parakeet-tdt-v2",
    )


def _load_parakeet_cpp(local_asr_model: str) -> LocalModel:
    executable, model = _ensure_parakeet_cpp(local_asr_model)
    logger.info("Loaded %s with parakeet.cpp", local_asr_model)
    return partial(
        _transcribe_parakeet_cpp,
        executable,
        model,
        local_asr_model == "parakeet-flash",
    )


def _load_cohere() -> LocalModel:
    try:
        import transcribe_cpp
    except ImportError as error:
        raise RuntimeError("Cohere support is not installed") from error
    model_path = _ensure_file(
        COHERE_DIR / COHERE_FILENAME,
        COHERE_MODEL_URL,
        COHERE_MODEL_SHA256,
    )
    try:
        model = transcribe_cpp.Model(str(model_path), backend="auto")
    except Exception as error:
        raise RuntimeError(f"Could not load cohere-transcribe: {error}") from error
    logger.info("Loaded Cohere Transcribe with transcribe.cpp")
    return partial(_transcribe_cohere, model)


def load_local_model(local_asr_model: str = "base") -> LocalModel:
    if local_asr_model in WHISPER_MODELS:
        executable, model = _ensure_whisper(local_asr_model)
        logger.info("Loaded multilingual Whisper model %s", model)
        return partial(_transcribe_whisper, executable, model)
    if local_asr_model in ONNX_PARAKEET_MODELS:
        return _load_parakeet(local_asr_model)
    if local_asr_model in PARAKEET_CPP_MODELS:
        return _load_parakeet_cpp(local_asr_model)
    if local_asr_model in COHERE_MODELS:
        return _load_cohere()
    raise RuntimeError(f"Unsupported local ASR model: {local_asr_model}")
