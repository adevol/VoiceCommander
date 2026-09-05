from concurrent.futures import Future
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from voicecommander.app import State, VoiceCommander, _close_model
from voicecommander.audio import Recorder
from voicecommander.local_asr import LocalAsrEngine
from voicecommander.session import RecordingSession
from voicecommander.settings import Settings


class LifecycleTests(unittest.TestCase):
    def test_failed_audio_start_closes_stream_and_allows_retry(self):
        device = Mock()
        first, second = Mock(), Mock()
        first.start.side_effect = RuntimeError("device unavailable")
        device.RawInputStream.side_effect = [first, second]
        recorder = Recorder()
        with patch.dict("sys.modules", {"sounddevice": device}):
            with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                recorder.start()
            first.close.assert_called_once_with()
            recorder.start()
        second.start.assert_called_once_with()

    def test_failed_preview_worker_start_stops_capture_and_timer(self):
        recorder = Mock()
        future = Future()
        future.set_result(Mock())
        session = RecordingSession(recorder, Settings(), future, False, queue.SimpleQueue())
        with (
            patch("voicecommander.session.threading.Timer") as timer,
            patch("voicecommander.session.threading.Thread") as worker,
        ):
            worker.return_value.start.side_effect = RuntimeError("thread unavailable")
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                session.start(Mock())
        recorder.stop.assert_called_once_with()
        timer.return_value.cancel.assert_called_once_with()

    def test_obsolete_model_is_closed_when_loading_finishes(self):
        future = Future()
        future.set_running_or_notify_cancel()
        engine = Mock()

        _close_model(future)
        engine.close.assert_not_called()
        future.set_result(engine)

        engine.close.assert_called_once_with()

    def test_shutdown_during_finalization_preserves_recording_without_paste(self):
        settings = Settings(asr_provider="openrouter")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("voicecommander.app.Recorder"),
            patch("voicecommander.app.ThreadPoolExecutor"),
            patch("voicecommander.session.transcribe_openrouter") as pipeline,
            patch("voicecommander.app.deliver_text") as deliver,
            patch("voicecommander.app._notify") as notify,
        ):
            app = VoiceCommander(settings)
            session = RecordingSession(app.recorder, settings, None, False, queue.SimpleQueue())
            session.path = Path(directory) / "recording.wav"
            session.path.touch()
            app._session = session
            app.state = State.PROCESSING

            def finish_after_shutdown(*args, **kwargs):
                with app._lock:
                    app._closing = True
                return "finished text"

            pipeline.side_effect = finish_after_shutdown
            with self.assertLogs("voicecommander.app", level="ERROR"):
                app._finish(session)

            self.assertTrue(session.path.exists())
            deliver.assert_not_called()
            notify.assert_not_called()
            app.on_hotkey()
            app.recorder.start.assert_not_called()

    def test_engine_close_during_startup_cannot_leave_a_server_running(self):
        started = threading.Event()
        release = threading.Event()
        engine = LocalAsrEngine(Path("cli"), Path("server"), Path("model"))
        server = Mock()
        errors = []

        def start(*args):
            started.set()
            self.assertTrue(release.wait(2))
            return server

        def preview():
            try:
                engine.preview(b"pcm", Settings())
            except RuntimeError as error:
                errors.append(str(error))

        with patch("voicecommander.local_asr._start_whisper_server", side_effect=start):
            worker = threading.Thread(target=preview)
            worker.start()
            self.assertTrue(started.wait(2))
            closer = threading.Thread(target=engine.close)
            closer.start()
            self.assertTrue(engine._preview_cancel.wait(2))
            release.set()
            worker.join(2)
            closer.join(2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, ["Whisper preview cancelled"])
        server.close.assert_called()
        server.transcribe.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            engine.preview(b"pcm", Settings())

    def test_stopped_preview_does_not_start_a_server(self):
        stop = threading.Event()
        stop.set()
        engine = LocalAsrEngine(Path("cli"), Path("server"), Path("model"))
        with patch("voicecommander.local_asr._start_whisper_server") as start:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                engine.preview(b"pcm", Settings(), stop)
        start.assert_not_called()
