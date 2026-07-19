from __future__ import annotations

import logging
import os
import tempfile
import wave
from pathlib import Path
from threading import Lock

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, input_device: str = "") -> None:
        self.input_device = input_device
        self._chunks: list[bytes] = []
        self._stream = None
        self._lock = Lock()

    def start(self) -> None:
        import sounddevice

        device = None
        if self.input_device:
            try:
                # _microphones formats selections as "index: name".
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
        self._chunks.append(bytes(data))

    def stop(self) -> Path:
        with self._lock:
            if self._stream is None:
                raise RuntimeError("Recording is not active")
            stream, self._stream = self._stream, None
        stream.stop()
        stream.close()
        if not self._chunks:
            raise RuntimeError("No audio was recorded")

        directory = Path(tempfile.gettempdir()) / "VoiceCommander"
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix="recording-", suffix=".wav", dir=directory)
        os.close(descriptor)
        path = Path(filename)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(CHANNELS)
            output.setsampwidth(SAMPLE_WIDTH)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(b"".join(self._chunks))
        self._chunks.clear()
        return path
