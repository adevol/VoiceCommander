"""One recording's capture, preview worker, and stop handoff."""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import CancelledError, Future
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .audio import Recorder
from .local_asr import LocalAsrEngine, PreviewText
from .pipeline import refine_transcript, transcribe_openrouter
from .preview import PREVIEW_MODELS, transcribe_live
from .settings import Settings

logger = logging.getLogger(__name__)


class RecordingSession:
    def __init__(
        self,
        recorder: Recorder,
        settings: Settings,
        model: Future[LocalAsrEngine] | None,
        markdown: bool,
        updates: queue.SimpleQueue[PreviewText | str | None],
    ) -> None:
        self.settings = settings
        self._model = model
        self.markdown = markdown
        self.path: Path | None = None
        self._recorder = recorder
        self._updates = updates
        self._stop = threading.Event()
        self._publish_lock = threading.Lock()
        self._drain_lock = threading.Lock()
        self._languages: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._preview: PreviewText | None = None
        self._worker: threading.Thread | None = None
        self._timer: threading.Timer | None = None
        self._recording = False
        self._closed = False

    def start(self, on_limit: Callable[[RecordingSession], None]) -> None:
        self._recorder.start()
        self._recording = True
        try:
            self._timer = threading.Timer(self.settings.max_seconds, lambda: on_limit(self))
            self._timer.daemon = True
            self._timer.start()
            if (
                self.settings.live_preview
                and self._model is not None
                and self.settings.local_asr_model in PREVIEW_MODELS
            ):
                self._updates.put("Listening")
                worker = threading.Thread(target=self._run_preview, daemon=True)
                worker.start()
                self._worker = worker
        except Exception:
            self.close()
            raise

    def _run_preview(self) -> None:
        try:
            # A download can outlive a recording. Never wait for it after stop.
            while not self._model.done():
                if self._stop.wait(0.05):
                    return
            if self._stop.is_set():
                return
            for update in transcribe_live(
                self._recorder, self._model.result(), self.settings, self._stop, self._languages
            ):
                with self._publish_lock:
                    if self._stop.is_set():
                        return
                    self._preview = update
                    self._updates.put(update)
        except Exception as error:
            with self._publish_lock:
                if not self._stop.is_set():
                    logger.exception("Live preview failed; recording continues")
                    self._updates.put(f"Preview unavailable: {error}")

    def stop(self) -> Path:
        """Stop capture immediately; drain preview in the processing worker."""
        with self._publish_lock:
            self._stop.set()
            if self._worker is not None:
                self._updates.put("Finalizing")
        if self._timer is not None:
            self._timer.cancel()
        if self._recording:
            self._recording = False
            self.path = self._recorder.stop()
        if self.path is None:
            raise RuntimeError("No recording is available")
        return self.path

    def _wait_preview(self) -> tuple[Settings, PreviewText | None]:
        """Return a fixed snapshot only after the worker can no longer change it."""
        with self._drain_lock:
            if self._worker is not None:
                self._worker.join(timeout=0.5)
                if self._worker.is_alive():
                    if self._model is not None and self._model.done() and not self._model.cancelled():
                        if self._model.exception() is None:
                            self._model.result().cancel_preview()
                    self._worker.join()
                self._worker = None
            settings = self.settings
            try:
                settings = replace(settings, language=self._languages.get_nowait())
            except queue.Empty:
                pass
            self.settings = settings
            return settings, self._preview

    def finish(
        self, api_key: str, on_refine: Callable[[str], None] | None = None
    ) -> str:
        """Produce the final text, keeping preview reuse private to this recording."""
        if self._recording or self.path is None:
            raise RuntimeError("Stop recording before finalizing")
        settings, preview = self._wait_preview()
        if self._closed:
            raise CancelledError("Recording session closed")
        if settings.asr_provider == "local":
            model = self._model.result() if self._model is not None else None
            if self._closed:
                raise CancelledError("Recording session closed")
            if model is None:
                raise RuntimeError("Local ASR model is not loaded")
            raw = model.transcribe(self.path, settings, preview)
        else:
            raw = transcribe_openrouter(self.path, settings, api_key)
        if self._closed:
            raise CancelledError("Recording session closed")
        return refine_transcript(raw, settings, api_key, self.markdown, on_refine)

    def close(self) -> None:
        """Stop background work and preserve any unfinished recording."""
        self._closed = True
        try:
            if self._recording:
                self.stop()
            else:
                self._stop.set()
                if self._timer is not None:
                    self._timer.cancel()
        finally:
            self._wait_preview()
