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
from collections import deque
from contextlib import suppress

from .audio import SAMPLE_RATE
from .local_asr import is_speech, load_local_model, merge_overlapping_text
from .settings import Settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 4
STEP_SECONDS = 1
BLOCK_SECONDS = 0.25
SILENCE_PEAK = 0.005
CAPTION_WORD_LIMIT = 200


def _merge_caption_text(previous: str, current: str) -> str:
    """Append only words not repeated by the overlapping audio window."""
    merged = merge_overlapping_text(previous, current)
    return " ".join((merged or f"{previous} {current}").split()[-CAPTION_WORD_LIMIT:])


def _put_latest(items: queue.Queue, item: object) -> None:
    """Replace the contents of a one-slot queue, dropping any backlog.

    Args:
        items: The one-slot queue to overwrite.
        item: The value to leave in it.
    """
    with suppress(queue.Empty):
        items.get_nowait()
    items.put_nowait(item)


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
        window_blocks = int(WINDOW_SECONDS / BLOCK_SECONDS)
        step_blocks = int(STEP_SECONDS / BLOCK_SECONDS)
        buffered = deque(maxlen=window_blocks)
        blocks_recorded = 0
        with microphone.recorder(samplerate=SAMPLE_RATE) as recorder:
            while not stop.is_set():
                buffered.append(recorder.record(numframes=block))
                blocks_recorded += 1
                if len(buffered) == window_blocks and blocks_recorded % step_blocks == 0:
                    samples = numpy.concatenate(buffered).mean(axis=1)
                    pcm16 = (
                        numpy.clip(samples, -1, 1) * 32767
                    ).astype(numpy.int16).tobytes()
                    _put_latest(
                        chunks,
                        (pcm16, float(numpy.abs(samples).max())),
                    )
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
        model = load_local_model(settings.caption_asr_model)
    except Exception as error:
        logger.exception("Could not load Whisper for captions")
        lines.put(f"Captions unavailable: {error}")
        stop.set()
        return
    if not stop.is_set():
        lines.put("(listening)")
    history = ""
    try:
        while not stop.is_set():
            try:
                pcm16, peak = chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            if peak < SILENCE_PEAK:
                continue
            try:
                result = model.preview(pcm16, settings)
            except Exception as error:
                logger.warning("Caption chunk failed: %s", error)
                continue
            if result is not None and is_speech(result.text):
                history = _merge_caption_text(history, result.text)
                lines.put(history)
    finally:
        model.close()


def run_captions(settings: Settings) -> int:
    import tkinter as tk

    chunks: queue.Queue = queue.Queue(maxsize=1)
    lines: queue.Queue = queue.Queue()
    stop = threading.Event()
    capture = threading.Thread(target=_capture, args=(chunks, lines, stop))
    transcriber = threading.Thread(target=_transcribe, args=(chunks, lines, settings, stop))

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
            text.delete("1.0", "end")
            text.insert("end", update)
            text.see("end")
            text.configure(state="disabled")
        root.after(250, poll)

    capture.start()
    transcriber.start()
    try:
        poll()
        root.mainloop()
    finally:
        stop.set()
        capture.join()
        transcriber.join()
    return 0
