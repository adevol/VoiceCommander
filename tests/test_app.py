from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from sys import modules
from unittest.mock import Mock, call, patch

from voicecommander.app import State, VoiceCommander, complete_recording
from voicecommander.session import RecordingSession
from voicecommander.local_asr import load_local_model
from voicecommander.settings import Settings


def run_first_tk_poll(root: Mock) -> None:
    def mainloop() -> None:
        root.after.call_args.args[1]()
        raise KeyboardInterrupt

    root.mainloop.side_effect = mainloop


class AppTests(unittest.TestCase):
    @patch("voicecommander.app._ctrl_pressed", return_value=False)
    @patch("voicecommander.app.threading.Timer")
    @patch("voicecommander.app.ThreadPoolExecutor")
    @patch("voicecommander.app.Recorder")
    def test_plain_hotkey_still_works_after_paste(
        self, recorder_type: Mock, executor_type: Mock, timer: Mock, ctrl_pressed: Mock
    ) -> None:
        keyboard = Mock()
        keyboard.is_pressed.return_value = True
        app = VoiceCommander(Settings(asr_provider="openrouter"))
        app._register_hotkeys(keyboard)
        callback = keyboard.add_hotkey.call_args_list[0].args[1]

        callback()
        app.state = State.IDLE
        callback()

        self.assertEqual(recorder_type.return_value.start.call_count, 2)
        ctrl_pressed.assert_called()
        keyboard.is_pressed.assert_not_called()

    @patch("voicecommander.app.deliver_text")
    def test_recording_is_deleted_only_after_success(self, deliver: Mock) -> None:
        session = Mock(spec=RecordingSession)
        with tempfile.TemporaryDirectory() as directory:
            session.path = Path(directory) / "success.wav"
            session.path.touch()
            session.finish.return_value = "done"
            complete_recording(session, "")
            self.assertFalse(session.path.exists())
            deliver.assert_called_once_with("done")

            session.path = Path(directory) / "failure.wav"
            session.path.touch()
            session.finish.side_effect = RuntimeError("failed")
            with self.assertRaises(RuntimeError):
                complete_recording(session, "")
            self.assertTrue(session.path.exists())

            session.finish.side_effect = None
            deliver.side_effect = RuntimeError("paste failed")
            with self.assertRaisesRegex(RuntimeError, "paste failed"):
                complete_recording(session, "")
            self.assertTrue(session.path.exists())

    def test_streaming_refinement_updates_the_overlay_queue(self) -> None:
        settings = Settings(asr_provider="openrouter", postprocess_strength=50)
        with (
            patch("voicecommander.app.Recorder"),
            patch("voicecommander.app.ThreadPoolExecutor"),
            patch("voicecommander.app.get_api_key", return_value="secret"),
            patch("voicecommander.app.complete_recording") as complete,
            patch("voicecommander.app._beep"),
        ):
            complete.side_effect = lambda *args, **kwargs: (
                kwargs["on_refine"](""),
                kwargs["on_refine"]("Edited text"),
            )
            app = VoiceCommander(settings)
            app.state = State.PROCESSING

            session = RecordingSession(app.recorder, settings, None, False, app._preview_updates)
            session.path = Path("recording.wav")
            app._session = session
            app._finish(session)

        self.assertEqual(app._preview_updates.get_nowait(), "Refining")
        self.assertEqual(app._preview_updates.get_nowait(), "Edited text")
        self.assertIsNone(app._preview_updates.get_nowait())
        self.assertEqual(app.state, State.IDLE)

    @patch("voicecommander.app._beep")
    @patch("voicecommander.app.threading.Timer")
    @patch("voicecommander.app.ThreadPoolExecutor")
    @patch("voicecommander.app.Recorder")
    def test_old_timer_cannot_stop_a_new_recording(
        self, recorder_type: Mock, executor_type: Mock, timer: Mock, beep: Mock
    ) -> None:
        recorder_type.return_value.stop.return_value = Path("recording.wav")
        app = VoiceCommander(Settings(asr_provider="openrouter"))
        app.on_hotkey()
        first = app._session
        old_timer = timer.call_args.args[1]
        app.on_hotkey()
        app.state = State.IDLE
        app.on_hotkey()
        second = app._session

        old_timer()

        self.assertIsNot(first, second)
        self.assertIs(app._session, second)
        self.assertEqual(app.state, State.RECORDING)
        recorder_type.return_value.stop.assert_called_once()

    @patch("voicecommander.app.threading.Timer")
    @patch("voicecommander.app.ThreadPoolExecutor")
    @patch("voicecommander.app.Recorder")
    def test_hotkey_is_ignored_during_processing(
        self, recorder_type: Mock, executor_type: Mock, timer: Mock
    ) -> None:
        recorder = recorder_type.return_value
        recorder.stop.return_value = Path("recording.wav")
        executor = executor_type.return_value
        app = VoiceCommander(Settings(asr_provider="openrouter"))

        app.on_hotkey()
        self.assertEqual(app.state, State.RECORDING)
        stale_timer = timer.call_args.args[1]
        app.on_hotkey()
        self.assertEqual(app.state, State.PROCESSING)
        app.on_hotkey()
        app.state = State.IDLE
        stale_timer()

        recorder.start.assert_called_once()
        recorder.stop.assert_called_once()
        executor.submit.assert_called_once()
        timer.return_value.start.assert_called_once()

    @patch("voicecommander.app._ctrl_pressed", return_value=False)
    @patch("voicecommander.app.threading.Timer")
    @patch("voicecommander.app.ThreadPoolExecutor")
    @patch("voicecommander.app.Recorder")
    def test_markdown_hotkey_marks_the_recording_for_markdown(
        self, recorder_type: Mock, executor_type: Mock, timer: Mock, ctrl_pressed: Mock
    ) -> None:
        keyboard = Mock()
        recorder_type.return_value.stop.return_value = Path("recording.wav")
        app = VoiceCommander(Settings(asr_provider="openrouter"))
        app._register_hotkeys(keyboard)
        markdown_callback = keyboard.add_hotkey.call_args_list[1].args[1]

        markdown_callback()
        markdown_callback()

        executor_type.return_value.submit.assert_called_once_with(app._finish, app._session)
        self.assertTrue(app._session.markdown)
        self.assertEqual(app._session.path, Path("recording.wav"))
        ctrl_pressed.assert_called()

    def test_settings_hotkey_applies_runtime_changes(self) -> None:
        keyboard = Mock()
        with (
            patch("voicecommander.app.Recorder") as recorder_type,
            patch("voicecommander.app.ThreadPoolExecutor") as executor_type,
            patch("voicecommander.app.threading.Event") as event_type,
            patch("voicecommander.app._beep"),
            patch("voicecommander.app.show_settings") as show_settings,
            patch("voicecommander.app.PreviewOverlay") as preview_type,
            patch.dict(modules, {"keyboard": keyboard}),
        ):
            event_type.return_value.is_set.return_value = True
            run_first_tk_poll(preview_type.return_value.root)
            initial = Settings(asr_provider="openrouter", hotkey="f9")
            updated = Settings(
                asr_provider="openrouter",
                hotkey="f10",
                markdown_hotkey="f6",
                input_device="3: Microphone",
            )
            show_settings.return_value = updated
            app = VoiceCommander(initial)
            app.run()

        self.assertEqual(
            [call.args[0] for call in keyboard.add_hotkey.call_args_list],
            ["f9", "f7", "ctrl+f9", "f10", "f6", "ctrl+f10"],
        )
        show_settings.assert_called_once_with(initial, preview_type.return_value.root)
        recorder_type.assert_called_with("3: Microphone")
        self.assertEqual(app.settings, updated)
        executor_type.return_value.shutdown.assert_called_once()

    def test_settings_window_returns_control_to_recording(self) -> None:
        keyboard = Mock()
        tkinter = Mock()
        with (
            patch("voicecommander.app.Recorder") as recorder_type,
            patch("voicecommander.app.ThreadPoolExecutor"),
            patch("voicecommander.app.threading.Event") as event_type,
            patch("voicecommander.app.threading.Timer"),
            patch("voicecommander.app._beep"),
            patch.dict(modules, {"keyboard": keyboard, "tkinter": tkinter}),
        ):
            event_type.return_value.is_set.return_value = True
            run_first_tk_poll(tkinter.Tk.return_value)
            app = VoiceCommander(Settings(asr_provider="openrouter"))
            app.request_settings()
            def interact() -> None:
                tkinter.Tk.return_value.after.call_args.args[1]()
                app.on_hotkey()
                raise KeyboardInterrupt
            tkinter.Tk.return_value.mainloop.side_effect = interact
            app.run()

        tkinter.Tk.assert_called_once_with()
        tkinter.Toplevel.assert_called_once_with(tkinter.Tk.return_value)
        tkinter.Toplevel.return_value.wait_window.assert_called_once_with()
        tkinter.Toplevel.return_value.mainloop.assert_not_called()
        recorder_type.return_value.start.assert_called_once_with()

    def test_switching_to_local_transcription_loads_the_model(self) -> None:
        with (
            patch("voicecommander.app.Recorder"),
            patch("voicecommander.app.ThreadPoolExecutor") as executor_type,
            patch("voicecommander.app.threading.Event") as event_type,
            patch("voicecommander.app._beep"),
            patch("voicecommander.app.show_settings") as show_settings,
            patch("voicecommander.app.PreviewOverlay") as preview_type,
            patch.dict(modules, {"keyboard": Mock()}),
        ):
            event_type.return_value.is_set.return_value = True
            run_first_tk_poll(preview_type.return_value.root)
            show_settings.return_value = Settings(asr_provider="local")
            app = VoiceCommander(Settings(asr_provider="openrouter"))
            self.assertIsNone(app._local_model)
            app.run()

        executor_type.return_value.submit.assert_called_once_with(load_local_model, "base")
        self.assertIsNotNone(app._local_model)


if __name__ == "__main__":
    unittest.main()
