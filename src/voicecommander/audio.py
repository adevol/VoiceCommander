from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path
from threading import Lock

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2

logger = logging.getLogger(__name__)


def write_wav(data: bytes) -> Path:
    directory = Path(tempfile.gettempdir()) / "VoiceCommander"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="recording-", suffix=".wav", dir=directory, delete=False) as temporary:
        path = Path(temporary.name)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(data)
    return path


class Recorder:
    """Records the microphone to a WAV file between `start` and `stop`.

    Attributes:
        input_device: An "index: name" entry from `settings._microphones`, of which
            only the leading index is used, or empty for the system default.
    """

    def __init__(self, input_device: str = "") -> None:
        self.input_device = input_device
        self._chunks = bytearray()
        self._stream = None
        self._lock = Lock()

    def start(self) -> None:
        import sounddevice

        device = None
        if self.input_device:
            try:
                device = int(self.input_device.split(":", 1)[0])
            except ValueError:
                device = self.input_device

        with self._lock:
            if self._stream is not None:
                raise RuntimeError("Recording is already active")
            self._chunks.clear()
            self._stream = sounddevice.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                device=device,
                callback=self._capture,
            )
            self._stream.start()

    def _capture(self, data, frames, time, status) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)
        with self._lock:
            self._chunks.extend(data)

    def snapshot(self, start: int = 0, length: int | None = None) -> bytes:
        with self._lock:
            end = start + length if length is not None else None
            return bytes(self._chunks[start:end])

    def stop(self) -> Path:
        with self._lock:
            if self._stream is None:
                raise RuntimeError("Recording is not active")
            stream, self._stream = self._stream, None
        stream.stop()
        stream.close()
        with self._lock:
            if not self._chunks:
                raise RuntimeError("No audio was recorded")
            data = bytes(self._chunks)
            self._chunks.clear()
        path = write_wav(data)
        return path
