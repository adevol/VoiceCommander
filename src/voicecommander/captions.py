"""Live captions for whatever the machine is playing, via speaker loopback.

Attributes:
    WINDOW_SECONDS: Seconds of audio in each rolling transcription window.
    STEP_SECONDS: Seconds between rolling-window updates.
    BLOCK_SECONDS: Seconds of audio per read from the loopback device.
    SILENCE_PEAK: Peak amplitude below which a chunk counts as silence.
"""

from __future__ import annotations

import logging
import queue
import threading
from contextlib import suppress
from dataclasses import dataclass, field

from .audio import SAMPLE_RATE
from .local_asr import LocalAsrEngine, PreviewResult, load_local_model
from .settings import Settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 4
STEP_SECONDS = 1
BLOCK_SECONDS = 0.25
CORRECTION_HORIZON = 1.0
SILENCE_PEAK = 0.005
CAPTION_HISTORY_LIMIT = 200


@dataclass(frozen=True, slots=True)
class CaptionWindow:
    starts_at: float
    pcm16: bytes
    peak: float


@dataclass(frozen=True, slots=True)
class CaptionSegment:
    text: str
    start: float
    end: float
    final: bool


@dataclass(frozen=True, slots=True)
class CaptionUpdate:
    segments: tuple[CaptionSegment, ...]

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments).strip()


@dataclass(slots=True)
class CaptionSession:
    model: LocalAsrEngine
    settings: Settings
    _segments: tuple[CaptionSegment, ...] = field(default=(), init=False, repr=False)

    def process(self, window: CaptionWindow) -> CaptionUpdate | None:
        """Transcribe one rolling window and return the reconciled caption history."""
        if window.peak < SILENCE_PEAK:
            return None
        result = self.model.preview(window.pcm16, self.settings)
        if result is None:
            return None
        current = _caption_segments(result, window.starts_at)
        retained = tuple(
            CaptionSegment(segment.text, segment.start, segment.end, True)
            for segment in self._segments
            if segment.end <= window.starts_at
        )
        current = _drop_repeated_segments(retained, current)
        self._segments = (retained + current)[-CAPTION_HISTORY_LIMIT:]
        update = CaptionUpdate(self._segments)
        return update if update.text else None

    def close(self) -> None:
        self.model.close()


def _caption_segments(
    result: PreviewResult, starts_at: float
) -> tuple[CaptionSegment, ...]:
    finalized_at = starts_at + max(0.0, result.duration - CORRECTION_HORIZON)
    previous_end = starts_at
    segments = []
    for segment in result.segments:
        end = starts_at + min(result.duration, max(0.0, segment.end))
        if end < previous_end:
            continue
        if is_speech(segment.text):
            segments.append(
                CaptionSegment(
                    segment.text,
                    previous_end,
                    end,
                    end <= finalized_at,
                )
            )
        previous_end = end
    return tuple(segments)


def _drop_repeated_segments(
    retained: tuple[CaptionSegment, ...], current: tuple[CaptionSegment, ...]
) -> tuple[CaptionSegment, ...]:
    """Remove ASR segments repeated across a rolling-window boundary."""
    limit = min(len(retained), len(current))
    for count in range(limit, 0, -1):
        previous = tuple(_normalized_text(segment.text) for segment in retained[-count:])
        repeated = tuple(_normalized_text(segment.text) for segment in current[:count])
        if previous == repeated:
            return current[count:]
    return current


def _normalized_text(text: str) -> str:
    return text.strip().casefold().strip(".,!?;:")


def _put_latest(items: queue.Queue, item: object) -> None:
    """Replace the contents of a one-slot queue, dropping any backlog.

    Args:
        items: The one-slot queue to overwrite.
        item: The value to leave in it.
    """
    with suppress(queue.Empty):
        items.get_nowait()
    items.put_nowait(item)


def is_speech(text: str) -> bool:
    """Report whether a transcript is speech rather than a non-speech label.

    Args:
        text: A transcript from Whisper.

    Returns:
        True if the text looks like speech, and False for the bracketed or
        parenthesised labels Whisper gives non-speech audio.
    """
    return bool(text) and not (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
    )


def _capture(chunks: queue.Queue, lines: queue.Queue, stop: threading.Event) -> None:
    """Record speaker loopback into `chunks` until `stop` is set.

    Args:
        chunks: Receives the latest rolling mono audio window, replacing any the
            transcriber has not collected yet.
        lines: Receives a message for the caption window if capture fails.
        stop: Ends the loop when set, and is set here on failure.
    """
    try:
        import numpy
        import soundcard

        speaker = soundcard.default_speaker()
        microphone = soundcard.get_microphone(str(speaker.name), include_loopback=True)
        block = int(SAMPLE_RATE * BLOCK_SECONDS)
        buffered: list = []
        window_blocks = int(WINDOW_SECONDS / BLOCK_SECONDS)
        step_blocks = int(STEP_SECONDS / BLOCK_SECONDS)
        blocks_since_update = 0
        blocks_recorded = 0
        with microphone.recorder(samplerate=SAMPLE_RATE) as recorder:
            while not stop.is_set():
                buffered.append(recorder.record(numframes=block))
                buffered = buffered[-window_blocks:]
                blocks_since_update += 1
                blocks_recorded += 1
                if len(buffered) == window_blocks and blocks_since_update >= step_blocks:
                    samples = numpy.concatenate(buffered).mean(axis=1)
                    pcm16 = (
                        numpy.clip(samples, -1, 1) * 32767
                    ).astype(numpy.int16).tobytes()
                    _put_latest(
                        chunks,
                        CaptionWindow(
                            (blocks_recorded - window_blocks) * BLOCK_SECONDS,
                            pcm16,
                            float(numpy.abs(samples).max()),
                        ),
                    )
                    blocks_since_update = 0
    except Exception as error:
        logger.exception("System audio capture failed")
        lines.put(f"Captions stopped, audio capture failed: {error}")
        stop.set()


def _transcribe(
    chunks: queue.Queue, lines: queue.Queue, settings: Settings, stop: threading.Event
) -> None:
    """Transcribe chunks into `lines` until `stop` is set.

    Args:
        chunks: Supplies mono audio blocks from `_capture`.
        lines: Receives each caption line, and any failure message.
        settings: Supplies the language.
        stop: Ends the loop when set, and is set here if the model cannot load.
    """
    try:
        session = CaptionSession(load_local_model(settings.caption_asr_model), settings)
    except Exception as error:
        logger.exception("Could not load Whisper for captions")
        lines.put(f"Captions unavailable: {error}")
        stop.set()
        return
    if not stop.is_set():
        lines.put("(listening)")
    try:
        while not stop.is_set():
            try:
                window = chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                update = session.process(window)
            except Exception as error:
                logger.warning("Caption chunk failed: %s", error)
                continue
            if update is not None:
                lines.put(update)
    finally:
        session.close()


def run_captions(settings: Settings) -> int:
    import tkinter as tk

    chunks: queue.Queue = queue.Queue(maxsize=1)
    lines: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=_capture, args=(chunks, lines, stop), daemon=True).start()
    threading.Thread(target=_transcribe, args=(chunks, lines, settings, stop), daemon=True).start()

    root = tk.Tk()
    root.title("VoiceCommander Captions")
    root.attributes("-topmost", True)
    text = tk.Text(root, wrap="word", height=10, width=70, font=("Segoe UI", 13), state="disabled")
    text.pack(fill="both", expand=True)

    def poll() -> None:
        while True:
            try:
                update = lines.get_nowait()
            except queue.Empty:
                break
            text.configure(state="normal")
            if isinstance(update, CaptionUpdate):
                text.delete("1.0", "end")
                text.insert("end", update.text)
            else:
                text.insert("end", str(update) + "\n")
            text.see("end")
            text.configure(state="disabled")
        root.after(250, poll)

    poll()
    root.mainloop()
    stop.set()
    return 0
