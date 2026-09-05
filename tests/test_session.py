from __future__ import annotations

import queue
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

from voicecommander.local_asr import PreviewText
from voicecommander.session import RecordingSession
from voicecommander.settings import Settings


class RecordingSessionTests(unittest.TestCase):
    def make_session(
        self,
        *,
        settings: Settings | None = None,
        model: Future | None = None,
    ) -> tuple[RecordingSession, Mock, queue.SimpleQueue]:
        recorder = Mock()
        recorder.stop.return_value = Path("recording.wav")
        updates: queue.SimpleQueue = queue.SimpleQueue()
        session = RecordingSession(
            recorder,
            settings or Settings(local_asr_model="base"),
            model,
            False,
            updates,
        )
        return session, recorder, updates

    def test_preview_uses_the_model_and_settings_captured_by_the_session(self) -> None:
        model = Mock()
        model_future: Future = Future()
        model_future.set_result(model)
        settings = Settings(local_asr_model="base", language="de-DE")
        session, recorder, _ = self.make_session(settings=settings, model=model_future)
        observed: list[tuple[object, object, object]] = []

        def transcribe(rec, loaded_model, used_settings, stop, languages):
            observed.append((rec, loaded_model, used_settings))
            yield PreviewText("Fest", "", 3.0)

        with patch("voicecommander.session.transcribe_live", side_effect=transcribe):
            session.start(Mock())
            session._worker.join(timeout=1)
            session.stop()
            final_settings, preview = session.wait_preview()

        self.assertEqual(observed, [(recorder, model, settings)])
        self.assertIs(final_settings, settings)
        self.assertEqual(preview, PreviewText("Fest", "", 3.0))

    def test_late_preview_result_is_ignored_after_stop(self) -> None:
        model_future: Future = Future()
        model_future.set_result(Mock())
        session, _, updates = self.make_session(model=model_future)
        entered = threading.Event()
        release = threading.Event()
        late = PreviewText("too late", "", 4.0)

        def transcribe(*args):
            entered.set()
            release.wait(timeout=2)
            yield late

        with patch("voicecommander.session.transcribe_live", side_effect=transcribe):
            session.start(Mock())
            self.assertTrue(entered.wait(timeout=1))
            session.stop()
            release.set()
            _, preview = session.wait_preview()

        self.assertIsNone(preview)
        published = []
        while not updates.empty():
            published.append(updates.get_nowait())
        self.assertNotIn(late, published)

    def test_close_does_not_wait_for_an_unresolved_model_load(self) -> None:
        unresolved: Future = Future()
        session, recorder, _ = self.make_session(model=unresolved)
        session.start(Mock())

        started = time.monotonic()
        session.close()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.4)
        recorder.stop.assert_called_once_with()
        self.assertFalse(session._worker and session._worker.is_alive())

    def test_wait_preview_cancels_and_drains_an_in_flight_decode(self) -> None:
        release = threading.Event()
        entered = threading.Event()
        model = Mock()
        model.cancel_preview.side_effect = release.set
        model_future: Future = Future()
        model_future.set_result(model)
        session, _, _ = self.make_session(model=model_future)

        def transcribe(*args):
            entered.set()
            release.wait(timeout=3)
            return
            yield  # pragma: no cover - make this a generator

        with patch("voicecommander.session.transcribe_live", side_effect=transcribe):
            session.start(Mock())
            self.assertTrue(entered.wait(timeout=1))
            session.stop()
            started = time.monotonic()
            session.wait_preview()
            elapsed = time.monotonic() - started

        model.cancel_preview.assert_called_once_with()
        self.assertGreaterEqual(elapsed, 0.45)
        self.assertLess(elapsed, 2.0)
        self.assertFalse(session._worker and session._worker.is_alive())

    def test_preview_state_is_not_carried_into_another_recording(self) -> None:
        model_future: Future = Future()
        model_future.set_result(Mock())
        first, _, _ = self.make_session(model=model_future)
        second, _, _ = self.make_session(model=model_future)
        preview = PreviewText("first recording", "", 5.0)

        with patch(
            "voicecommander.session.transcribe_live",
            side_effect=[iter((preview,)), iter(())],
        ):
            first.start(Mock())
            first._worker.join(timeout=1)
            first.stop()
            _, first_preview = first.wait_preview()
            second.start(Mock())
            second._worker.join(timeout=1)
            second.stop()
            second_settings, second_preview = second.wait_preview()

        self.assertEqual(first_preview, preview)
        self.assertIsNone(second_preview)
        self.assertEqual(second_settings.language, "auto")


if __name__ == "__main__":
    unittest.main()
