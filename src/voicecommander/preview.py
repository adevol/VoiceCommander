"""Live transcription worker and its small on-screen overlay."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import replace

from .audio import Recorder
from .local_asr import LocalAsrEngine, PreviewSegment, PreviewText
from .settings import Settings

PREVIEW_INTERVAL = 1.0
CORRECTION_HORIZON = 2.0
PREVIEW_MODELS = {"tiny", "base"}


def transcribe_live(
    recorder: Recorder,
    model: LocalAsrEngine,
    settings: Settings,
    stop: threading.Event,
    languages: queue.SimpleQueue[str],
) -> Iterator[PreviewText]:
    previous: tuple[PreviewSegment, ...] | None = None
    stable: tuple[PreviewSegment, ...] = ()
    settings_for_recording = settings
    while not stop.wait(PREVIEW_INTERVAL):
        pcm16 = recorder.snapshot()
        if not pcm16:
            continue
        if stop.is_set():
            break
        result = model.preview(pcm16, settings_for_recording)
        if result is None or stop.is_set():
            continue
        if settings_for_recording.language == "auto" and result.language and result.text:
            noise = (
                result.text.startswith("[")
                and result.text.endswith("]")
                or result.text.startswith("(")
                and result.text.endswith(")")
            )
            if not noise:
                settings_for_recording = replace(settings_for_recording, language=result.language)
                languages.put(result.language)
        eligible = tuple(
            segment
            for segment in result.segments
            if segment.end <= result.duration - CORRECTION_HORIZON
        )
        if previous is not None:
            confirmed = []
            for before, current in zip(previous, eligible):
                if before.text != current.text:
                    break
                confirmed.append(current)
            if tuple(segment.text for segment in confirmed[: len(stable)]) == tuple(
                segment.text for segment in stable
            ):
                stable = tuple(confirmed)
        previous = eligible
        stable_text = "".join(segment.text for segment in stable).strip()
        tentative = result.text
        if stable_text and tentative.startswith(stable_text):
            tentative = tentative[len(stable_text) :]
        update = PreviewText(stable_text, tentative, stable[-1].end if stable else 0.0)
        if update.text:
            yield update


class PreviewOverlay:
    def __init__(self) -> None:
        import tkinter as tk

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(background="#17191f")
        self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)
        self.text = tk.Label(
            self.root,
            background="#17191f",
            foreground="#f4f4f5",
            font=("Segoe UI", 13),
            justify="left",
            wraplength=720,
        )
        self.text.pack(padx=20, pady=14)

    def pump(self, updates: queue.SimpleQueue[PreviewText | str | None]) -> None:
        while True:
            try:
                text = updates.get_nowait()
            except queue.Empty:
                break
            if text is None:
                self.root.withdraw()
                continue
            if isinstance(text, PreviewText):
                text = text.text
            self.text.configure(text=text)
            self.root.update_idletasks()
            width = min(760, max(260, self.text.winfo_reqwidth() + 40))
            height = self.text.winfo_reqheight() + 28
            x = (self.root.winfo_screenwidth() - width) // 2
            self.root.geometry(f"{width}x{height}+{x}+28")
            self.root.deiconify()
            self.root.lift()

    def close(self) -> None:
        self.root.destroy()
