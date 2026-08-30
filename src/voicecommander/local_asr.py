"""Local speech recognition behind one small, warm-model interface."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
import wave
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from .settings import APP_DIR, WHISPER_MODELS, WHISPER_REVISION, Settings

WHISPER_DIR = APP_DIR / "whisper.cpp"
WHISPER_RUNTIME_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
)
WHISPER_RUNTIME_SHA256 = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _WhisperSession:
    engine: LocalAsrEngine
    settings: Settings

    def feed(self, pcm16: bytes) -> str | None:
        """Decode the latest complete 16 kHz mono PCM snapshot.

        A concurrent request is dropped rather than queued. The final WAV pass
        remains authoritative.
        """
        return self.engine._preview(pcm16, self.settings)

    def finish(self, path: Path) -> str:
        return _transcribe_whisper(
            self.engine.executable,
            self.engine.model,
            path,
            self.settings,
        )


@dataclass(slots=True)
class LocalAsrEngine:
    executable: Path
    server_executable: Path
    model: Path
    _server: _WhisperServer | None = field(default=None, init=False, repr=False)
    _preview_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def start(self, settings: Settings) -> _WhisperSession:
        return _WhisperSession(self, settings)

    def _preview(self, pcm16: bytes, settings: Settings) -> str | None:
        if not self._preview_lock.acquire(blocking=False):
            return None
        try:
            if self._server is None or self._server.process.poll() is not None:
                self._server = _start_whisper_server(self.server_executable, self.model)
            return self._server.transcribe(pcm16, settings)
        finally:
            self._preview_lock.release()

    def close(self) -> None:
        with self._preview_lock:
            if self._server is not None:
                self._server.close()
                self._server = None


@dataclass(frozen=True, slots=True)
class _WhisperServer:
    process: subprocess.Popen
    url: str

    def transcribe(self, pcm16: bytes, settings: Settings) -> str:
        boundary, body = _preview_request(pcm16, settings)
        request = Request(
            self.url + "/inference",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read(200).decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Whisper preview returned HTTP {error.code}{f': {detail}' if detail else ''}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Whisper preview failed: {error}") from error
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise RuntimeError("Whisper preview returned an invalid response")
        return result["text"].strip()

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)


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
    wanted = (executable_name, *required)
    if marker.exists() and all((directory / name).exists() for name in wanted):
        return executable

    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "runtime.zip"
        _download(url, archive_path, sha256)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = Path(member.filename).name
                if not member.is_dir() and (
                    name in wanted or name.lower().endswith(".dll")
                ):
                    with archive.open(member) as source, (directory / name).open(
                        "wb"
                    ) as output:
                        shutil.copyfileobj(source, output)

    if not all((directory / name).exists() for name in wanted):
        raise RuntimeError("The local ASR runtime archive was incomplete")
    marker.touch()
    return executable


def _ensure_whisper(local_asr_model: str) -> tuple[Path, Path, Path]:
    """Return the whisper.cpp runtime and model, downloading either if absent."""
    executable = _ensure_runtime(
        WHISPER_DIR,
        "whisper-cli.exe",
        WHISPER_RUNTIME_URL,
        WHISPER_RUNTIME_SHA256,
        ("whisper-server.exe", "whisper.dll", "ggml.dll"),
    )
    filename, expected_sha256 = WHISPER_MODELS[local_asr_model]
    model = WHISPER_DIR / filename
    if not model.exists():
        _download(
            f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{WHISPER_REVISION}/{filename}",
            model,
            expected_sha256,
        )
    return executable, WHISPER_DIR / "whisper-server.exe", model


def _preview_request(pcm16: bytes, settings: Settings) -> tuple[str, bytes]:
    wav = BytesIO()
    with wave.open(wav, "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm16)

    boundary = f"voicecommander-{os.urandom(12).hex()}"
    fields = {
        "response_format": "json",
        "language": _whisper_language(settings.language),
        "temperature": "0",
        "token_timestamps": "false",
    }
    if settings.vocabulary:
        fields |= {"prompt": settings.vocabulary, "carry_initial_prompt": "true"}
    body = bytearray()
    for name, value in fields.items():
        body.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    body.extend(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        'filename="preview.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    )
    body.extend(wav.getvalue())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    return boundary, bytes(body)


def _start_whisper_server(executable: Path, model: Path) -> _WhisperServer:
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            str(executable),
            "-m",
            str(model),
            "-t",
            str(min(8, os.cpu_count() or 4)),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-timestamps",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    server = _WhisperServer(process, url)
    try:
        for _ in range(150):
            if process.poll() is not None:
                raise RuntimeError("Whisper preview server stopped during startup")
            try:
                with urlopen(url + "/", timeout=0.2):
                    return server
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("Whisper preview server did not become ready")
    except Exception:
        server.close()
        raise


def _transcript(text: str) -> str:
    text = text.strip()
    if not text:
        raise RuntimeError("Local ASR returned an empty transcript")
    return text


def _whisper_language(language: str) -> str:
    return language if language == "auto" else language.split("-", 1)[0].lower()


def _transcribe_whisper(
    executable: Path, model: Path, path: Path, settings: Settings
) -> str:
    output_base = path.with_suffix(path.suffix + ".whisper")
    output_path = Path(str(output_base) + ".txt")
    command = [
        str(executable),
        "-m",
        str(model),
        "-f",
        str(path),
        "-l",
        _whisper_language(settings.language),
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
        text = output_path.read_text(encoding="utf-8")
    finally:
        output_path.unlink(missing_ok=True)
    return _transcript(text)


def load_local_model(local_asr_model: str = "base") -> LocalAsrEngine:
    if local_asr_model in WHISPER_MODELS:
        executable, server, model = _ensure_whisper(local_asr_model)
        logger.info("Loaded multilingual Whisper model %s", model)
        return LocalAsrEngine(executable, server, model)
    raise RuntimeError(f"Unsupported local ASR model: {local_asr_model}")
