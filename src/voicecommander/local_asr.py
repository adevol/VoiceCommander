"""Local speech recognition behind one small, warm-model interface."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.request import urlopen

from .settings import APP_DIR, WHISPER_MODELS, WHISPER_REVISION, Settings

WHISPER_DIR = APP_DIR / "whisper.cpp"
WHISPER_RUNTIME_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
)
WHISPER_RUNTIME_SHA256 = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    stable: str = ""
    tentative: str = ""


@dataclass(frozen=True, slots=True)
class FinalTranscript:
    text: str


class AsrSession(Protocol):
    def feed(self, pcm16: bytes) -> TranscriptUpdate | None: ...

    def finish(self, path: Path) -> FinalTranscript: ...


class LocalAsrEngine(Protocol):
    def start(self, settings: Settings) -> AsrSession: ...


@dataclass(frozen=True, slots=True)
class _WhisperSession:
    executable: Path
    model: Path
    settings: Settings

    def feed(self, pcm16: bytes) -> None:
        return None

    def finish(self, path: Path) -> FinalTranscript:
        return FinalTranscript(
            _transcribe_whisper(self.executable, self.model, path, self.settings)
        )


@dataclass(frozen=True, slots=True)
class _WhisperEngine:
    executable: Path
    model: Path

    def start(self, settings: Settings) -> AsrSession:
        return _WhisperSession(self.executable, self.model, settings)


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


def load_local_model(local_asr_model: str = "base") -> LocalAsrEngine:
    if local_asr_model in WHISPER_MODELS:
        executable, model = _ensure_whisper(local_asr_model)
        logger.info("Loaded multilingual Whisper model %s", model)
        return _WhisperEngine(executable, model)
    raise RuntimeError(f"Unsupported local ASR model: {local_asr_model}")
