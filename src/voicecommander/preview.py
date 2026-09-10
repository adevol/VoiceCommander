"""Live transcription worker and its small on-screen overlay."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from .audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, Recorder
from .local_asr import (
    LocalAsrEngine,
    PreviewSegment,
    PreviewText,
    is_speech,
    merge_overlapping_text,
)
from .settings import WHISPER_DEFINITIONS, Settings

PREVIEW_INTERVAL = 2.0
PREVIEW_WINDOW_SECONDS = 8.0
CORRECTION_HORIZON = 2.0
PREVIEW_MODELS = frozenset(name for name, model in WHISPER_DEFINITIONS.items() if model.preview)
PREVIEW_WINDOW_BYTES = round(PREVIEW_WINDOW_SECONDS * SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)


def transcribe_live(
    recorder: Recorder,
    model: LocalAsrEngine,
    settings: Settings,
    stop: threading.Event,
) -> Iterator[PreviewText]:
    previous: tuple[PreviewSegment, ...] | None = None
    previous_start = 0.0
    stable_text = ""
    stable_end = 0.0
    language = settings.language
    while not stop.wait(PREVIEW_INTERVAL):
        window_start = max(0.0, stable_end - CORRECTION_HORIZON)
        if window_start != previous_start:
            previous = None
            previous_start = window_start
        first_byte = round(window_start * SAMPLE_RATE) * CHANNELS * SAMPLE_WIDTH
        pcm16 = recorder.snapshot(first_byte, PREVIEW_WINDOW_BYTES)
        if not pcm16:
            continue
        if stop.is_set():
            break
        result = model.preview(
            pcm16,
            language=language,
            vocabulary=settings.vocabulary,
            stop=stop,
        )
        if result is None or stop.is_set():
            continue
        if language == "auto" and result.language and result.text:
            if is_speech(result.text):
                language = result.language
        eligible = tuple(
            PreviewSegment(segment.text, segment.end + window_start)
            for segment in result.segments
            if segment.end <= result.duration - CORRECTION_HORIZON
        )
        if previous is not None:
            confirmed = []
            for before, current in zip(previous, eligible):
                if before.text != current.text:
                    break
                confirmed.append(current)
            confirmed_text = "".join(segment.text for segment in confirmed).strip()
            joined = (
                merge_overlapping_text(stable_text, confirmed_text)
                if stable_text
                else confirmed_text
            )
            # Timestamps wobble between decodes; repeated text is the stable signal.
            if joined and joined != stable_text:
                stable_text = joined
                stable_end = confirmed[-1].end
        previous = eligible
        tentative = result.text
        if stable_text:
            combined = merge_overlapping_text(stable_text, tentative)
            if combined is not None:
                tentative = combined[len(stable_text) :]
        update = PreviewText(
            stable_text,
            tentative,
            stable_end,
            language if language != "auto" else None,
        )
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
        self.text = tk.Text(
            self.root,
            background="#17191f",
            foreground="#f4f4f5",
            font=("Segoe UI", 13),
            wrap="word",
            width=72,
            height=1,
            borderwidth=0,
            highlightthickness=0,
            takefocus=False,
            state="disabled",
        )
        self.text.tag_configure("tentative", foreground="#a1a1aa")
        self.text.pack(padx=20, pady=14)
        self.text.bind("<Configure>", lambda event: self._resize())

    def pump(self, updates: queue.SimpleQueue[PreviewText | str | None]) -> None:
        latest: PreviewText | str | None = None
        received = False
        while True:
            try:
                latest = updates.get_nowait()
                received = True
            except queue.Empty:
                break
        if not received:
            return
        if latest is None:
            self.root.withdraw()
            return
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if isinstance(latest, PreviewText):
            self.text.insert("end", latest.stable)
            self.text.insert("end", latest.tentative, "tentative")
        else:
            self.text.insert("end", latest)
        self.text.configure(state="disabled")
        self.root.deiconify()
        self._resize()
        self.root.lift()

    def _resize(self) -> None:
        # Keep long dictations compact and the changing phrase in view.
        # The first Configure event supplies the width needed for word wrapping.
        lines = (
            self.text.count("1.0", "end-1c", "update", "displaylines")
            if self.text.winfo_width() > 1
            else 0
        )
        self.text.configure(height=min(8, (lines or 0) + 1))
        self.root.update_idletasks()
        width = self.text.winfo_reqwidth() + 40
        height = self.text.winfo_reqheight() + 28
        x = (self.root.winfo_screenwidth() - width) // 2
        self.root.geometry(f"{width}x{height}+{x}+28")
        self.text.see("end")

    def close(self) -> None:
        self.root.destroy()
