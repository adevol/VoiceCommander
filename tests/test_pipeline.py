from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from sys import modules
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from voicecommander.app import State, VoiceCommander, complete_recording
from voicecommander.captions import _put_latest, _transcribe, is_speech
from voicecommander.pipeline import (
    WHISPER_RUNTIME_SHA256,
    _ensure_whisper,
    _openrouter,
    load_local_model,
    postprocess_openrouter,
    run_pipeline,
    transcribe_local,
    transcribe_openrouter,
)
from voicecommander.settings import (
    WHISPER_MODELS,
    Settings,
    load_settings,
    save_settings,
    validate,
)


class EssentialTests(unittest.TestCase):
    def test_unknown_local_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unsupported local ASR model: nemotron"):
            load_local_model("nemotron")

    @patch("voicecommander.pipeline._download")
    def test_each_whisper_size_downloads_its_own_verified_file(self, download: Mock) -> None:
        for name, (filename, expected_sha256) in WHISPER_MODELS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                whisper_dir = Path(directory)
                # Pretend the runtime is already unpacked so only the model download runs.
                (whisper_dir / "whisper-cli.exe").touch()
                (whisper_dir / f".runtime-{WHISPER_RUNTIME_SHA256}").touch()
                download.reset_mock()

                with patch("voicecommander.pipeline.WHISPER_DIR", whisper_dir):
                    _, model = _ensure_whisper(name)

                url, destination, sha256 = download.call_args.args
                self.assertEqual(model.name, filename)
                self.assertEqual(destination, whisper_dir / filename)
                self.assertEqual(sha256, expected_sha256)
                self.assertIn(f"/{filename}", url)

    @patch("voicecommander.pipeline.subprocess.run")
    def test_multilingual_whisper_uses_selected_language(self, run: Mock) -> None:
        run.return_value = Mock(returncode=0, stderr="")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            path.touch()
            Path(str(path) + ".whisper.txt").write_text("Guten Tag", encoding="utf-8")
            result = transcribe_local(
                path,
                Settings(language="de-DE"),
                (Path("whisper-cli.exe"), Path("ggml-base-q5_1.bin")),
            )

        self.assertEqual(result, "Guten Tag")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-l") + 1], "de")

    def test_settings_round_trip_without_secrets(self) -> None:
        settings = Settings(
            hotkey="f9",
            input_device='2: Mic "Main"',
            max_seconds=42,
            local_asr_model="small",
            openrouter_asr_model="google/chirp-3",
            postprocess_style="professional",
            postprocess_strength=75,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            save_settings(settings, path)
            self.assertEqual(load_settings(path), settings)
            self.assertNotIn("api_key", path.read_text(encoding="utf-8"))
            with path.open("a", encoding="utf-8") as config:
                config.write('hotkeys = "f9"\n')
            with self.assertRaisesRegex(ValueError, "Unknown settings: hotkeys"):
                load_settings(path)

    def test_settings_reject_unsupported_asr_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid language"):
            validate(Settings(language="zh-CN"))
        with self.assertRaisesRegex(ValueError, "Invalid OpenRouter transcription model"):
            validate(Settings(openrouter_asr_model="custom/transcriber"))
        with self.assertRaisesRegex(ValueError, "Invalid local ASR model"):
            validate(Settings(local_asr_model="tiny"))

    def test_caption_noise_labels_are_dropped(self) -> None:
        self.assertTrue(is_speech("Hello there"))
        for noise in ("", "[BLANK_AUDIO]", "(music)", "[Musik]"):
            self.assertFalse(is_speech(noise))

    def test_captions_drop_backlog_and_stop_after_model_failure(self) -> None:
        chunks = queue.Queue(maxsize=1)
        _put_latest(chunks, "old")
        _put_latest(chunks, "new")
        self.assertEqual(chunks.get_nowait(), "new")

        lines = queue.Queue()
        stop = threading.Event()
        with (
            patch(
                "voicecommander.captions.load_local_model",
                side_effect=RuntimeError("model failed"),
            ),
            self.assertLogs("voicecommander.captions", level="ERROR"),
        ):
            _transcribe(chunks, lines, Settings(), stop)
        self.assertTrue(stop.is_set())
        self.assertIn("model failed", lines.get_nowait())

    def test_postprocessing_failure_falls_back_to_raw_transcript(self) -> None:
        settings = Settings(asr_provider="openrouter", postprocess_strength=50)

        with (
            patch("voicecommander.pipeline.transcribe", return_value="raw transcript"),
            patch(
                "voicecommander.pipeline.postprocess_openrouter",
                side_effect=RuntimeError("cloud unavailable"),
            ),
            self.assertLogs("voicecommander.pipeline", level="ERROR"),
        ):
            result = run_pipeline(Path("recording.wav"), settings, None, "secret")
        self.assertEqual(result, "raw transcript")

    @patch("voicecommander.pipeline._openrouter")
    def test_postprocessing_controls_are_sent_in_the_prompt(self, openrouter: Mock) -> None:
        openrouter.return_value = {"choices": [{"message": {"content": "Edited"}}]}
        settings = Settings(
            postprocess_style="professional",
            postprocess_strength=75,
        )

        self.assertEqual(postprocess_openrouter("Raw", settings, "secret"), "Edited")
        prompt = openrouter.call_args.args[2]["messages"][0]["content"]
        self.assertIn("keeping the speaker's wording and sentence order", prompt)
        self.assertIn("professional prose", prompt)
        self.assertIn("scratch that", prompt)
        self.assertIn("language of the dictated text", prompt)

    @patch("voicecommander.app._ctrl_pressed", return_value=False)
    @patch("voicecommander.app.threading.Timer")
    @patch("voicecommander.app.ThreadPoolExecutor")
    @patch("voicecommander.app.Recorder")
    def test_plain_hotkey_still_works_after_paste(
        self, recorder_type: Mock, executor_type: Mock, timer: Mock, ctrl_pressed: Mock
    ) -> None:
        keyboard = Mock()
        keyboard.is_pressed.return_value = True  # The library's cached state may be stale.
        app = VoiceCommander(Settings(asr_provider="openrouter"))
        app._register_hotkeys(keyboard)
        callback = keyboard.add_hotkey.call_args_list[0].args[1]

        callback()
        app.state = State.IDLE
        callback()

        self.assertEqual(recorder_type.return_value.start.call_count, 2)
        ctrl_pressed.assert_called()
        keyboard.is_pressed.assert_not_called()

    @patch("voicecommander.pipeline._openrouter")
    def test_selected_openrouter_transcription_model_is_sent(self, openrouter: Mock) -> None:
        openrouter.return_value = {"text": "Transcript"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            path.write_bytes(b"RIFF")
            result = transcribe_openrouter(
                path,
                Settings(asr_provider="openrouter", openrouter_asr_model="google/chirp-3"),
                "secret",
            )

        self.assertEqual(result, "Transcript")
        payload = openrouter.call_args.args[2]
        self.assertEqual(payload["model"], "google/chirp-3")
        self.assertEqual(payload["language"], "en")

    def test_zero_editing_strength_keeps_raw_transcript(self) -> None:
        with (
            patch("voicecommander.pipeline.transcribe", return_value="raw transcript"),
            patch("voicecommander.pipeline.postprocess_openrouter") as postprocess,
        ):
            result = run_pipeline(
                Path("recording.wav"),
                Settings(postprocess_strength=0),
                None,
                "secret",
            )
        self.assertEqual(result, "raw transcript")
        postprocess.assert_not_called()

    @patch("voicecommander.app.deliver_text")
    @patch("voicecommander.app.run_pipeline")
    def test_recording_is_deleted_only_after_success(
        self, process: Mock, deliver: Mock
    ) -> None:
        settings = Settings(asr_provider="openrouter")
        with tempfile.TemporaryDirectory() as directory:
            success = Path(directory) / "success.wav"
            success.touch()
            process.return_value = "done"
            complete_recording(success, settings, None, "")
            self.assertFalse(success.exists())
            deliver.assert_called_once_with("done")

            failure = Path(directory) / "failure.wav"
            failure.touch()
            process.side_effect = RuntimeError("failed")
            with self.assertRaises(RuntimeError):
                complete_recording(failure, settings, None, "")
            self.assertTrue(failure.exists())

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

    def test_settings_hotkey_applies_runtime_changes(self) -> None:
        keyboard = Mock()
        with (
            patch("voicecommander.app.Recorder") as recorder_type,
            patch("voicecommander.app.ThreadPoolExecutor") as executor_type,
            patch("voicecommander.app.threading.Event") as event_type,
            patch("voicecommander.app._beep"),
            patch("voicecommander.app.show_settings") as show_settings,
            patch.dict(modules, {"keyboard": keyboard}),
        ):
            event_type.return_value.wait.side_effect = [True, KeyboardInterrupt]
            initial = Settings(asr_provider="openrouter", hotkey="f9")
            updated = Settings(
                asr_provider="openrouter",
                hotkey="f10",
                input_device="3: Microphone",
            )
            show_settings.return_value = updated
            app = VoiceCommander(initial)
            app.run()

        self.assertEqual(
            [call.args[0] for call in keyboard.add_hotkey.call_args_list],
            ["f9", "ctrl+f9", "f10", "ctrl+f10"],
        )
        show_settings.assert_called_once_with(initial)
        recorder_type.assert_called_with("3: Microphone")
        self.assertEqual(app.settings, updated)
        executor_type.return_value.shutdown.assert_called_once()

    def test_switching_to_local_transcription_loads_the_model(self) -> None:
        with (
            patch("voicecommander.app.Recorder"),
            patch("voicecommander.app.ThreadPoolExecutor") as executor_type,
            patch("voicecommander.app.threading.Event") as event_type,
            patch("voicecommander.app._beep"),
            patch("voicecommander.app.show_settings") as show_settings,
            patch.dict(modules, {"keyboard": Mock()}),
        ):
            event_type.return_value.wait.side_effect = [True, KeyboardInterrupt]
            show_settings.return_value = Settings(asr_provider="local")
            app = VoiceCommander(Settings(asr_provider="openrouter"))
            self.assertIsNone(app._local_model)
            app.run()

        executor_type.return_value.submit.assert_called_once_with(load_local_model, "base")
        self.assertIsNotNone(app._local_model)

    @patch("voicecommander.pipeline.urlopen")
    def test_openrouter_error_includes_response_detail(self, urlopen: Mock) -> None:
        urlopen.side_effect = HTTPError(
            "https://openrouter.ai", 400, "Bad Request", {}, BytesIO(b"invalid model")
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 400: invalid model"):
            _openrouter("/test", "secret", {})


if __name__ == "__main__":
    unittest.main()
