"""Live captions for whatever the machine is playing, via speaker loopback.

Attributes:
    CHUNK_SECONDS: Seconds of audio per transcribed block.
    BLOCK_SECONDS: Seconds of audio per read from the loopback device.
    SILENCE_PEAK: Peak amplitude below which a chunk counts as silence.
"""

from __future__ import annotations

import logging
import queue
import threading
from contextlib import suppress

from .audio import SAMPLE_RATE, write_wav
from .local_asr import load_local_model
from .settings import Settings

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 8
BLOCK_SECONDS = 0.25
SILENCE_PEAK = 0.005


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
        chunks: Receives one mono block of about `CHUNK_SECONDS`, replacing any the
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
        with microphone.recorder(samplerate=SAMPLE_RATE) as recorder:
            while not stop.is_set():
                buffered.append(recorder.record(numframes=block))
                if len(buffered) * BLOCK_SECONDS >= CHUNK_SECONDS:
                    _put_latest(chunks, numpy.concatenate(buffered).mean(axis=1))
                    buffered = []
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
    import numpy

    try:
        model = load_local_model("base")
    except Exception as error:
        logger.exception("Could not load Whisper for captions")
        lines.put(f"Captions unavailable: {error}")
        stop.set()
        return
    if not stop.is_set():
        lines.put("(listening)")
    while not stop.is_set():
        try:
            chunk = chunks.get(timeout=0.5)
        except queue.Empty:
            continue
        if numpy.abs(chunk).max() < SILENCE_PEAK:
            continue
        path = write_wav((numpy.clip(chunk, -1, 1) * 32767).astype(numpy.int16).tobytes())
        try:
            text = model.start(settings).finish(path)
        except Exception as error:
            logger.warning("Caption chunk failed: %s", error)
            continue
        finally:
            path.unlink(missing_ok=True)
        if is_speech(text):
            lines.put(text)


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
                line = lines.get_nowait()
            except queue.Empty:
                break
            text.configure(state="normal")
            text.insert("end", line + "\n")
            text.see("end")
            text.configure(state="disabled")
        root.after(250, poll)

    poll()
    root.mainloop()
    stop.set()
    return 0
