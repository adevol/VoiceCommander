from __future__ import annotations

import ctypes
import logging
import os
import queue
import sys
import threading
import time
import winsound
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

from .audio import Recorder
from .local_asr import LocalAsrEngine, PreviewText, load_local_model
from .pipeline import run_pipeline
from .preview import PreviewOverlay
from .session import RecordingSession
from .settings import APP_DIR, Settings, get_api_key, show_settings

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


def _close_model(model: Future[LocalAsrEngine] | None) -> None:
    if model is None:
        return
    if not model.done():
        if not model.cancel():
            model.add_done_callback(_close_model)
    elif not model.cancelled() and model.exception() is None:
        model.result().close()


def configure_logging() -> Path:
    log_dir = Path(os.environ.get("LOCALAPPDATA", APP_DIR)) / "VoiceCommander"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "voicecommander.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [file_handler]
    if sys.stderr is not None:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        handlers.append(console)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    return log_path


def deliver_text(text: str) -> None:
    import keyboard
    import pyperclip

    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    logger.info("Delivered transcript with %d characters", len(text))


def complete_recording(
    path: Path,
    settings: Settings,
    loaded_model: LocalAsrEngine | None,
    api_key: str,
    markdown: bool = False,
    on_refine: Callable[[str], None] | None = None,
    preview: PreviewText | None = None,
    deliver: Callable[[str], None] | None = None,
) -> str:
    text = run_pipeline(
        path,
        settings,
        loaded_model,
        api_key,
        markdown=markdown,
        on_refine=on_refine,
        preview=preview,
    )
    (deliver or deliver_text)(text)
    try:
        path.unlink()
    except OSError:
        logger.warning("Could not delete completed recording %s", path, exc_info=True)
    return text


def _notify(title: str, message: str, error: bool = False) -> None:
    logger.log(logging.ERROR if error else logging.INFO, "%s: %s", title, message)
    ctypes.windll.user32.MessageBoxW(None, message, title, 0x10 if error else 0x40)


def _beep(frequency: int) -> None:
    winsound.Beep(frequency, 90)


def _ctrl_pressed() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)


class VoiceCommander:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.recorder = Recorder(settings.input_device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voicecommander")
        self.state = State.IDLE
        self._lock = threading.Lock()
        self._preview_updates: queue.SimpleQueue[PreviewText | str | None] = queue.SimpleQueue()
        self._session: RecordingSession | None = None
        self._closing = False
        self._menu_open = False
        self._settings_requested = threading.Event()
        self._local_model = self._load_model(settings)

    def _load_model(self, settings: Settings) -> Future[LocalAsrEngine] | None:
        if settings.asr_provider != "local":
            return None
        return self.executor.submit(load_local_model, settings.local_asr_model)

    def on_hotkey(self, markdown: bool = False) -> None:
        with self._lock:
            if self._menu_open or self._closing:
                return
            if self.state is State.PROCESSING:
                logger.info("Ignoring hotkey while processing")
                return
            if self.state is State.IDLE:
                self._start_recording(markdown)
            else:
                self._stop_recording()

    def request_settings(self) -> None:
        with self._lock:
            if not self._menu_open and self.state is State.IDLE:
                self._settings_requested.set()

    def _register_hotkeys(self, keyboard: object) -> None:
        keyboard.add_hotkey(
            self.settings.hotkey,
            lambda: None if _ctrl_pressed() else self.on_hotkey(),
        )
        keyboard.add_hotkey(
            self.settings.markdown_hotkey,
            lambda: None if _ctrl_pressed() else self.on_hotkey(markdown=True),
        )
        keyboard.add_hotkey(f"ctrl+{self.settings.hotkey}", self.request_settings)

    def _show_settings(self, parent: object) -> None:
        with self._lock:
            if self._menu_open or self.state is not State.IDLE:
                return
            self._menu_open = True
        try:
            previous = self.settings
            updated = show_settings(previous, parent)
            if updated is not None:
                with self._lock:
                    self.settings = updated

                if updated.input_device != previous.input_device:
                    self.recorder = Recorder(updated.input_device)

                model_changed = (
                    updated.asr_provider != previous.asr_provider
                    or updated.local_asr_model != previous.local_asr_model
                )
                if model_changed:
                    _close_model(self._local_model)
                    self._local_model = self._load_model(updated)

                if (
                    updated.hotkey != previous.hotkey
                    or updated.markdown_hotkey != previous.markdown_hotkey
                ):
                    import keyboard

                    keyboard.unhook_all_hotkeys()
                    self._register_hotkeys(keyboard)

                logger.info(
                    "Settings updated: hotkey=%s asr=%s model=%s",
                    updated.hotkey,
                    updated.asr_provider,
                    updated.local_asr_model,
                )
        finally:
            with self._lock:
                self._menu_open = False

    def _start_recording(self, markdown: bool = False) -> None:
        session = RecordingSession(
            self.recorder, self.settings, self._local_model, markdown, self._preview_updates
        )
        try:
            session.start(self._recording_limit)
        except Exception as error:
            self._preview_updates.put(None)
            logger.exception("Could not start recording")
            _notify("VoiceCommander", str(error), error=True)
            return
        self.state = State.RECORDING
        self._session = session
        logger.info("Recording started in %s mode", "Markdown" if markdown else "text")
        _beep(900)

    def _stop_recording(self) -> None:
        self.state = State.PROCESSING
        session = self._session
        try:
            path = session.stop()
            self.executor.submit(self._finish, session)
        except Exception as error:
            session.close()
            self._preview_updates.put(None)
            self.state = State.IDLE
            self._session = None
            logger.exception("Could not stop recording")
            _notify("VoiceCommander", str(error), error=True)
            return
        logger.info("Recording stopped; processing %s", path.name)
        _beep(650)

    def _recording_limit(self, session: RecordingSession) -> None:
        with self._lock:
            if not self._closing and self.state is State.RECORDING and self._session is session:
                logger.info("Recording limit reached")
                self._stop_recording()

    def _finish(self, session: RecordingSession) -> None:
        path = session.path
        try:
            settings, preview = session.wait_preview()
            if self._closing:
                return
            model = session.model.result() if session.model else None
            key = get_api_key() if settings.uses_openrouter or session.markdown else ""
            complete_recording(
                path,
                settings,
                model,
                key,
                markdown=session.markdown,
                on_refine=lambda text: self._preview_updates.put(text or "Refining"),
                preview=preview,
                deliver=self._deliver,
            )
            if not self._closing:
                _beep(1100)
        except Exception as error:
            logger.exception("Pipeline failed; recording preserved at %s", path)
            if not self._closing:
                _notify("VoiceCommander failed", f"{error}\n\nRecording preserved at:\n{path}", error=True)
        finally:
            with self._lock:
                if self._session is session:
                    self._preview_updates.put(None)
                    self.state = State.IDLE
                    self._session = None

    def _deliver(self, text: str) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("Application closed before transcript delivery")
            deliver_text(text)

    def run(self) -> int:
        import keyboard

        overlay = PreviewOverlay()
        logger.info(
            "Starting VoiceCommander: hotkey=%s markdown_hotkey=%s asr=%s postprocess_strength=%s",
            self.settings.hotkey,
            self.settings.markdown_hotkey,
            self.settings.asr_provider,
            self.settings.postprocess_strength,
        )
        self._register_hotkeys(keyboard)
        logger.info("VoiceCommander ready")
        _beep(1000)

        def poll() -> None:
            if self._settings_requested.is_set():
                self._settings_requested.clear()
                self._show_settings(overlay.root)
            overlay.pump(self._preview_updates)
            overlay.root.after(100, poll)

        try:
            overlay.root.after(0, poll)
            overlay.root.mainloop()
        except KeyboardInterrupt:
            logger.info("VoiceCommander stopped")
        finally:
            with self._lock:
                self._closing = True
                session = self._session
            keyboard.unhook_all_hotkeys()
            overlay.close()
            try:
                if session is not None:
                    session.close()
            finally:
                self.executor.shutdown(wait=True, cancel_futures=True)
                _close_model(self._local_model)
        return 0
