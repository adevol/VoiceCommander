"""Live transcription worker and its small on-screen overlay."""

from __future__ import annotations

import queue
import threading

from .audio import Recorder
from .local_asr import LocalAsrEngine
from .settings import Settings

PREVIEW_INTERVAL = 1.0
PREVIEW_MODELS = {"tiny", "base"}


def transcribe_live(
    recorder: Recorder,
    model: LocalAsrEngine,
    settings: Settings,
    stop: threading.Event,
    updates: queue.SimpleQueue[str | None],
) -> None:
    session = model.start(settings)
    while not stop.wait(PREVIEW_INTERVAL):
        pcm16 = recorder.snapshot()
        if not pcm16:
            continue
        text = session.feed(pcm16)
        if text and not stop.is_set():
            updates.put(text)


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

    def pump(self, updates: queue.SimpleQueue[str | None]) -> None:
        while True:
            try:
                text = updates.get_nowait()
            except queue.Empty:
                break
            if text is None:
                self.root.withdraw()
                continue
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
