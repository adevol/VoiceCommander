"""Local speech recognition behind one small, warm-model interface."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import string
import subprocess
import tempfile
import time
import wave
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from .settings import APP_DIR, WHISPER_MODELS, WHISPER_REVISION, Settings

if TYPE_CHECKING:
    from .preview import PreviewText

WHISPER_DIR = APP_DIR / "whisper.cpp"
WHISPER_RUNTIME_URL = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
)
WHISPER_RUNTIME_SHA256 = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"
logger = logging.getLogger(__name__)
FINAL_OVERLAP_SECONDS = 2.0
MIN_SAVED_PREFIX_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class PreviewSegment:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class PreviewResult:
    text: str
    segments: tuple[PreviewSegment, ...]
    duration: float
    language: str | None = None


@dataclass(slots=True)
class LocalAsrEngine:
    executable: Path
    server_executable: Path
    model: Path
    _server: _WhisperServer | None = field(default=None, init=False, repr=False)
    _preview_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _preview_cancel: Event = field(default_factory=Event, init=False, repr=False)

    def preview(self, pcm16: bytes, settings: Settings) -> PreviewResult | None:
        """Decode a snapshot, dropping rather than queuing a concurrent request."""
        return self._decode_pcm(pcm16, settings, blocking=False)

    def _decode_pcm(
        self, pcm16: bytes, settings: Settings, *, blocking: bool
    ) -> PreviewResult | None:
        if not self._preview_lock.acquire(blocking=blocking):
            return None
        try:
            self._preview_cancel.clear()
            if self._server is None or self._server.process.poll() is not None:
                self._server = _start_whisper_server(
                    self.server_executable, self.model, self._preview_cancel
                )
            return self._server.transcribe(pcm16, settings)
        finally:
            self._preview_lock.release()

    def transcribe(
        self,
        path: Path,
        settings: Settings,
        preview: PreviewText | None = None,
    ) -> str:
        if (
            preview is not None
            and preview.stable
            and preview.stable_end - FINAL_OVERLAP_SECONDS
            >= MIN_SAVED_PREFIX_SECONDS
        ):
            try:
                tail_start = preview.stable_end - FINAL_OVERLAP_SECONDS
                with wave.open(str(path), "rb") as source:
                    source.setpos(min(source.getnframes(), round(tail_start * SAMPLE_RATE)))
                    tail = source.readframes(source.getnframes())
                result = self._decode_pcm(tail, settings, blocking=True)
                if (
                    result.segments
                    and result.segments[0].start <= FINAL_OVERLAP_SECONDS
                ):
                    before = preview.stable.split()
                    after = result.text.split()
                    before_keys = [
                        word.strip(string.punctuation).casefold() for word in before
                    ]
                    after_keys = [
                        word.strip(string.punctuation).casefold() for word in after
                    ]
                    for count in range(min(len(before), len(after)), 2, -1):
                        if before_keys[-count:] == after_keys[:count]:
                            return " ".join(before + after[count:])
            except (OSError, EOFError, RuntimeError, wave.Error):
                logger.warning(
                    "Could not finalize from the committed preview; decoding the full file",
                    exc_info=True,
                )
        return _transcribe_whisper(self.executable, self.model, path, settings)

    def cancel_preview(self) -> None:
        self._preview_cancel.set()
        if self._server is not None:
            self._server.close()

    def close(self) -> None:
        self.cancel_preview()
        with self._preview_lock:
            if self._server is not None:
                self._server = None


@dataclass(frozen=True, slots=True)
class _WhisperServer:
    process: subprocess.Popen
    url: str

    def transcribe(self, pcm16: bytes, settings: Settings) -> PreviewResult:
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
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("text"), str)
            or not isinstance(result.get("duration"), (int, float))
            or not isinstance(result.get("segments"), list)
        ):
            raise RuntimeError("Whisper preview returned an invalid response")
        segments = []
        for segment in result["segments"]:
            if (
                not isinstance(segment, dict)
                or not isinstance(segment.get("text"), str)
                or not isinstance(segment.get("start"), (int, float))
                or not isinstance(segment.get("end"), (int, float))
            ):
                raise RuntimeError("Whisper preview returned an invalid response")
            segments.append(
                PreviewSegment(
                    segment["text"], float(segment["start"]), float(segment["end"])
                )
            )
        probabilities = result.get("language_probabilities", {})
        language = None
        if isinstance(probabilities, dict):
            candidates = [
                (code, probability)
                for code, probability in probabilities.items()
                if isinstance(code, str) and isinstance(probability, (int, float))
            ]
            if candidates:
                language = max(candidates, key=lambda candidate: candidate[1])[0]
        return PreviewResult(
            result["text"].strip(),
            tuple(segments),
            float(result["duration"]),
            language,
        )

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
        "response_format": "verbose_json",
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


def _start_whisper_server(executable: Path, model: Path, cancel: Event) -> _WhisperServer:
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
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    server = _WhisperServer(process, url)
    try:
        for _ in range(150):
            if cancel.is_set():
                raise RuntimeError("Whisper preview cancelled")
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
    text = text.strip()
    if not text:
        raise RuntimeError("Local ASR returned an empty transcript")
    return text


def load_local_model(local_asr_model: str = "base") -> LocalAsrEngine:
    if local_asr_model in WHISPER_MODELS:
        executable, server, model = _ensure_whisper(local_asr_model)
        logger.info("Loaded multilingual Whisper model %s", model)
        return LocalAsrEngine(executable, server, model)
    raise RuntimeError(f"Unsupported local ASR model: {local_asr_model}")
