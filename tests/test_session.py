from __future__ import annotations

import queue
import threading
import unittest
from concurrent.futures import CancelledError, Future
from pathlib import Path
from unittest.mock import Mock, patch

from voicecommander.local_asr import PreviewText
from voicecommander.session import RecordingSession
from voicecommander.settings import Settings


class RecordingSessionTests(unittest.TestCase):
    def test_recording_limit_sets_the_final_transcription_timeout(self):
        for seconds, expected in ((10, 300), (300, 1200), (600, 2400)):
            with self.subTest(seconds=seconds):
                future, model = self.loaded_model()
                session, _, _ = self.make_session(
                    settings=Settings(live_preview=False, max_seconds=seconds), model=future
                )
                session.start(Mock())
                path = session.stop()
                session.finish("")
                model.transcribe.assert_called_once_with(
                    path,
                    language="auto",
                    vocabulary="",
                    timeout=expected,
                    preview=None,
                )

    def make_session(self, *, settings=None, model=None):
        recorder = Mock()
        recorder.stop.return_value = Path("recording.wav")
        updates = queue.SimpleQueue()
        session = RecordingSession(
            recorder,
            settings or Settings(local_asr_model="base"),
            model,
            False,
            updates,
        )
        return session, recorder, updates

    @staticmethod
    def loaded_model():
        model = Mock()
        model.transcribe.return_value = "final text"
        future = Future()
        future.set_result(model)
        return future, model

    def test_finish_uses_captured_model_settings_and_preview(self):
        model_future, model = self.loaded_model()
        settings = Settings(local_asr_model="base", language="de-DE")
        session, recorder, _ = self.make_session(settings=settings, model=model_future)
        preview = PreviewText("Fest", "", 3.0)
        preview_done = threading.Event()

        def transcribe(rec, loaded_model, used_settings, stop):
            self.assertIs(rec, recorder)
            self.assertIs(loaded_model, model)
            self.assertIs(used_settings, settings)
            yield preview
            preview_done.set()

        with patch("voicecommander.session.transcribe_live", side_effect=transcribe):
            session.start(Mock())
            self.assertTrue(preview_done.wait(1))
            path = session.stop()
            result = session.finish("")

        self.assertEqual(result, "final text")
        model.transcribe.assert_called_once_with(
            path,
            language=settings.language,
            vocabulary=settings.vocabulary,
            timeout=1200,
            preview=preview,
        )

    def test_finish_passes_detected_language_to_final_transcription(self):
        model_future, model = self.loaded_model()
        session, _, _ = self.make_session(model=model_future)
        preview = PreviewText("Hallo", "", 3.0, "de")
        preview_done = threading.Event()

        def transcribe(recorder, loaded_model, used_settings, stop):
            yield preview
            preview_done.set()

        with patch("voicecommander.session.transcribe_live", side_effect=transcribe):
            session.start(Mock())
            self.assertTrue(preview_done.wait(1))
            path = session.stop()
            session.finish("")

        model.transcribe.assert_called_once_with(
            path,
            language="de",
            vocabulary="",
            timeout=1200,
            preview=preview,
        )
        self.assertEqual(session.settings.language, "auto")

    def test_late_preview_result_is_ignored_after_stop(self):
        model_future, model = self.loaded_model()
        session, _, updates = self.make_session(model=model_future)
        entered = threading.Event()
        release = threading.Event()
        late = PreviewText("too late", "", 4.0, "fr")

        def transcribe(*args):
            entered.set()
            release.wait(2)
            yield late

        with patch("voicecommander.session.transcribe_live", side_effect=transcribe):
            session.start(Mock())
            self.assertTrue(entered.wait(1))
            path = session.stop()
            release.set()
            session.finish("")

        model.transcribe.assert_called_once_with(
            path,
            language="auto",
            vocabulary="",
            timeout=1200,
            preview=None,
        )
        published = []
        while not updates.empty():
            published.append(updates.get_nowait())
        self.assertNotIn(late, published)

    def test_close_does_not_wait_for_an_unresolved_model_load(self):
        unresolved = Future()
        session, recorder, _ = self.make_session(model=unresolved)
        session.start(Mock())

        session.close()

        recorder.stop.assert_called_once_with()
        self.assertFalse(session._worker and session._worker.is_alive())
        with self.assertRaises(CancelledError):
            session.finish("")

    def test_shutdown_during_transcription_does_not_start_refinement(self):
        future, model = self.loaded_model()
        session, _, _ = self.make_session(
            settings=Settings(live_preview=False, postprocess_strength=50), model=future
        )
        session.start(Mock())
        session.stop()

        def transcribe(*args, **kwargs):
            session.close()
            return "raw text"

        model.transcribe.side_effect = transcribe
        with patch("voicecommander.session.refine_transcript") as refine:
            with self.assertRaises(CancelledError):
                session.finish("secret")
        refine.assert_not_called()

    def test_finish_cancels_and_drains_an_in_flight_decode(self):
        release = threading.Event()
        entered = threading.Event()
        model_future, model = self.loaded_model()
        model.cancel_preview.side_effect = release.set
        session, _, _ = self.make_session(model=model_future)

        def transcribe(*args):
            entered.set()
            release.wait(3)
            return
            yield  # pragma: no cover - make this a generator

        with patch("voicecommander.session.transcribe_live", side_effect=transcribe):
            session.start(Mock())
            self.assertTrue(entered.wait(1))
            session.stop()
            session.finish("")

        model.cancel_preview.assert_called_once_with()
        self.assertFalse(session._worker and session._worker.is_alive())

    def test_preview_state_is_not_carried_into_another_recording(self):
        model_future, model = self.loaded_model()
        first, _, _ = self.make_session(model=model_future)
        second, _, _ = self.make_session(model=model_future)
        preview = PreviewText("first recording", "", 5.0)
        first_done = threading.Event()
        second_done = threading.Event()

        def first_preview(*args):
            yield preview
            first_done.set()

        def second_preview(*args):
            second_done.set()
            return
            yield

        with patch(
            "voicecommander.session.transcribe_live",
            side_effect=[first_preview(), second_preview()],
        ):
            first.start(Mock())
            self.assertTrue(first_done.wait(1))
            first_path = first.stop()
            first.finish("")
            second.start(Mock())
            self.assertTrue(second_done.wait(1))
            second_path = second.stop()
            second.finish("")

        self.assertEqual(model.transcribe.call_args_list[0].args, (first_path,))
        self.assertEqual(model.transcribe.call_args_list[0].kwargs["preview"], preview)
        self.assertEqual(model.transcribe.call_args_list[1].args, (second_path,))
        self.assertIsNone(model.transcribe.call_args_list[1].kwargs["preview"])
        self.assertEqual(model.transcribe.call_args_list[1].kwargs["language"], "auto")
        self.assertEqual(second.settings.language, "auto")


if __name__ == "__main__":
    unittest.main()
