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

from .audio import Recorder
from .local_asr import LocalAsrEngine, load_local_model
from .pipeline import run_pipeline
from .preview import PREVIEW_MODELS, PreviewOverlay, PreviewText, transcribe_live
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
        model.cancel()
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
) -> str:
    text = run_pipeline(path, settings, loaded_model, api_key, markdown=markdown)
    deliver_text(text)
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
        self._timer: threading.Timer | None = None
        self._preview_stop = threading.Event()
        self._preview_stop.set()
        self._preview_updates: queue.SimpleQueue[PreviewText | str | None] = queue.SimpleQueue()
        self._menu_open = False
        self._markdown = False
        self._settings_requested = threading.Event()
        self._local_model = self._load_model(settings)

    def _load_model(self, settings: Settings) -> Future[LocalAsrEngine] | None:
        if settings.asr_provider != "local":
            return None
        return self.executor.submit(load_local_model, settings.local_asr_model)

    def on_hotkey(self, markdown: bool = False) -> None:
        with self._lock:
            if self._menu_open:
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
        try:
            self.recorder.start()
        except Exception as error:
            logger.exception("Could not start recording")
            _notify("VoiceCommander", str(error), error=True)
            return
        self.state = State.RECORDING
        self._markdown = markdown
        self._timer = threading.Timer(self.settings.max_seconds, self._recording_limit)
        self._timer.daemon = True
        self._timer.start()
        if (
            self.settings.live_preview
            and self._local_model is not None
            and self.settings.local_asr_model in PREVIEW_MODELS
        ):
            stop = threading.Event()
            self._preview_stop = stop
            self._preview_updates.put("Listening")
            threading.Thread(target=self._preview, args=(stop,), daemon=True).start()
        logger.info("Recording started in %s mode", "Markdown" if markdown else "text")
        _beep(900)

    def _preview(self, stop: threading.Event) -> None:
        try:
            transcribe_live(
                self.recorder,
                self._local_model.result(),
                self.settings,
                stop,
                self._preview_updates,
            )
        except Exception as error:
            logger.exception("Live preview failed; recording continues")
            if not stop.is_set():
                self._preview_updates.put(f"Preview unavailable: {error}")

    def _stop_recording(self) -> None:
        self.state = State.PROCESSING
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if not self._preview_stop.is_set():
            self._preview_stop.set()
            self._preview_updates.put("Finalizing")
            model = self._local_model
            if (
                model is not None
                and model.done()
                and not model.cancelled()
                and model.exception() is None
            ):
                model.result().cancel_preview()
        try:
            path = self.recorder.stop()
            self.executor.submit(self._finish, path, self._markdown)
        except Exception as error:
            self._preview_stop.set()
            self._preview_updates.put(None)
            self.state = State.IDLE
            logger.exception("Could not stop recording")
            _notify("VoiceCommander", str(error), error=True)
            return
        logger.info("Recording stopped; processing %s", path.name)
        _beep(650)

    def _recording_limit(self) -> None:
        with self._lock:
            if self.state is State.RECORDING:
                logger.info("Recording limit reached")
                self._stop_recording()

    def _finish(self, path: Path, markdown: bool = False) -> None:
        try:
            model = self._local_model.result() if self._local_model else None
            key = get_api_key() if self.settings.uses_openrouter or markdown else ""
            complete_recording(path, self.settings, model, key, markdown=markdown)
            _beep(1100)
        except Exception as error:
            logger.exception("Pipeline failed; recording preserved at %s", path)
            _notify("VoiceCommander failed", f"{error}\n\nRecording preserved at:\n{path}", error=True)
        finally:
            self._preview_stop.set()
            self._preview_updates.put(None)
            with self._lock:
                self.state = State.IDLE
                self._markdown = False

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
            self._preview_stop.set()
            keyboard.unhook_all_hotkeys()
            overlay.close()
            try:
                _close_model(self._local_model)
            finally:
                self.executor.shutdown(wait=False, cancel_futures=True)
        return 0
