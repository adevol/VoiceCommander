from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from functools import partial
from io import BytesIO
from pathlib import Path
from sys import modules
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from voicecommander.app import State, VoiceCommander, complete_recording
from voicecommander.captions import _put_latest, _transcribe, is_speech
from voicecommander.local_asr import (
    FinalTranscript,
    WHISPER_RUNTIME_SHA256,
    _ensure_whisper,
    _transcribe_whisper,
    load_local_model,
)
from voicecommander.pipeline import (
    _openrouter,
    postprocess_openrouter,
    run_pipeline,
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

    @patch("voicecommander.local_asr._transcribe_whisper", return_value="Hello")
    @patch("voicecommander.local_asr._ensure_whisper")
    def test_whisper_starts_a_final_only_session(self, ensure: Mock, transcribe: Mock) -> None:
        ensure.return_value = Path("whisper-cli.exe"), Path("model.bin")
        settings = Settings(language="de-DE")

        session = load_local_model("base").start(settings)

        self.assertIsNone(session.feed(b"partial audio"))
        self.assertEqual(session.finish(Path("recording.wav")), FinalTranscript("Hello"))
        transcribe.assert_called_once_with(
            Path("whisper-cli.exe"), Path("model.bin"), Path("recording.wav"), settings
        )

    def test_local_pipeline_refines_only_the_final_transcript(self) -> None:
        settings = Settings(postprocess_strength=50)
        engine = Mock()
        engine.start.return_value.finish.return_value = FinalTranscript("raw transcript")

        with patch(
            "voicecommander.pipeline.postprocess_openrouter", return_value="edited transcript"
        ) as postprocess:
            result = run_pipeline(Path("recording.wav"), settings, engine, "secret")

        self.assertEqual(result, "edited transcript")
        engine.start.assert_called_once_with(settings)
        engine.start.return_value.finish.assert_called_once_with(Path("recording.wav"))
        postprocess.assert_called_once_with("raw transcript", settings, "secret")

    @patch("voicecommander.local_asr._download")
    def test_each_whisper_size_downloads_its_own_verified_file(self, download: Mock) -> None:
        for name, (filename, expected_sha256) in WHISPER_MODELS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                whisper_dir = Path(directory)
                (whisper_dir / "whisper-cli.exe").touch()
                (whisper_dir / f".runtime-{WHISPER_RUNTIME_SHA256}").touch()
                download.reset_mock()

                with patch("voicecommander.local_asr.WHISPER_DIR", whisper_dir):
                    _, model = _ensure_whisper(name)

                url, destination, sha256 = download.call_args.args
                self.assertEqual(model.name, filename)
                self.assertEqual(destination, whisper_dir / filename)
                self.assertEqual(sha256, expected_sha256)
                self.assertIn(f"/{filename}", url)
        self.assertEqual(WHISPER_MODELS["tiny"][0], "ggml-tiny-q5_1.bin")

    @patch("voicecommander.local_asr.subprocess.run")
    def test_whisper_passes_selected_or_automatic_language(self, run: Mock) -> None:
        run.return_value = Mock(returncode=0, stderr="")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            path.touch()
            transcript = Path(str(path) + ".whisper.txt")
            model = partial(
                _transcribe_whisper,
                Path("whisper-cli.exe"),
                Path("ggml-base-q5_1.bin"),
            )
            for language, expected, text in (
                ("de-DE", "de", "Guten Tag"),
                ("auto", "auto", "Bonjour"),
            ):
                with self.subTest(language):
                    transcript.write_text(text, encoding="utf-8")
                    self.assertEqual(model(path, Settings(language=language)), text)
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("-l") + 1], expected)

    @patch("voicecommander.pipeline._openrouter")
    @patch("voicecommander.local_asr.subprocess.run")
    def test_custom_vocabulary_reaches_supported_models(self, run: Mock, openrouter: Mock) -> None:
        run.return_value = Mock(returncode=0, stderr="")
        openrouter.return_value = {"choices": [{"message": {"content": "Text"}}]}
        settings = Settings(vocabulary="Kubernetes, VoiceCommander", postprocess_strength=50)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            path.write_bytes(b"RIFF")
            transcript = Path(str(path) + ".whisper.txt")
            transcript.write_text("Kubernetes", encoding="utf-8")
            model = partial(_transcribe_whisper, Path("w.exe"), Path("m.bin"))
            model(path, settings)
            transcribe_openrouter(path, settings, "secret")
            postprocess_openrouter("Raw", settings, "secret")

            command = run.call_args.args[0]
            self.assertNotIn("-p", command, "whisper.cpp reads -p as --processors")
            self.assertEqual(command[command.index("--prompt") + 1], "Kubernetes, VoiceCommander")
            self.assertIn(
                "--carry-initial-prompt",
                command,
                "without it the prompt biases only the first 30 s of a recording",
            )
            for call in openrouter.call_args_list:
                self.assertIn("Kubernetes, VoiceCommander", str(call.args[2]["messages"]))

            transcript.write_text("Kubernetes", encoding="utf-8")
            model(path, Settings())
        self.assertNotIn("--prompt", run.call_args.args[0])

    def test_settings_round_trip_without_secrets(self) -> None:
        settings = Settings(
            hotkey="f9",
            markdown_hotkey="f6",
            language="de-DE",
            input_device='2: Mic "Main"',
            max_seconds=42,
            local_asr_model="small",
            openrouter_asr_model="xiaomi/mimo-v2.5",
            postprocess_style="professional",
            postprocess_strength=75,
        )
        self.assertEqual(Settings().language, "auto")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            save_settings(settings, path)
            self.assertEqual(load_settings(path), settings)
            self.assertNotIn("api_key", path.read_text(encoding="utf-8"))
            with path.open("a", encoding="utf-8") as config:
                config.write('hotkeys = "f9"\n')
            with self.assertRaisesRegex(ValueError, "Unknown settings: hotkeys"):
                load_settings(path)

    def test_removed_local_models_migrate_to_whisper(self) -> None:
        migrations = {
            "large-v3-turbo": "small",
            "parakeet-tdt-v3": "base",
            "parakeet-tdt-v2": "base",
            "parakeet-flash": "base",
            "nemotron-3.5": "base",
            "cohere-transcribe": "base",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            for old, new in migrations.items():
                with self.subTest(old):
                    path.write_text(
                        f'local_asr_model = "{old}"\n', encoding="utf-8"
                    )
                    self.assertEqual(load_settings(path).local_asr_model, new)

    def test_settings_reject_unsupported_asr_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid language"):
            validate(Settings(language="zh-CN"))
        with self.assertRaisesRegex(ValueError, "Invalid OpenRouter transcription model"):
            validate(Settings(openrouter_asr_model="custom/transcriber"))
        validate(Settings(local_asr_model="tiny"))
        for name in (
            "parakeet-tdt-v3",
            "parakeet-tdt-v2",
            "parakeet-flash",
            "nemotron-3.5",
            "cohere-transcribe",
        ):
            with self.subTest(name), self.assertRaisesRegex(
                ValueError, "Invalid local ASR model"
            ):
                validate(Settings(local_asr_model=name))
        with self.assertRaisesRegex(ValueError, "at most 500 characters"):
            validate(Settings(vocabulary="x" * 501))
        with self.assertRaisesRegex(ValueError, "hotkeys must be different"):
            validate(Settings(hotkey="F8", markdown_hotkey="f8"))

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
            patch("voicecommander.pipeline.transcribe", return_value=FinalTranscript("raw transcript")),
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
        self.assertNotIn("auto locale", prompt)

    @patch("voicecommander.pipeline._openrouter")
    def test_markdown_mode_requests_structure_and_latex(self, openrouter: Mock) -> None:
        openrouter.return_value = {"choices": [{"message": {"content": "## Result"}}]}

        result = postprocess_openrouter("Describe a heading", Settings(), "secret", markdown=True)

        self.assertEqual(result, "## Result")
        prompt = openrouter.call_args.args[2]["messages"][0]["content"]
        self.assertIn("finished Markdown", prompt)
        self.assertIn("$...$", prompt)
        self.assertIn("$$...$$", prompt)
        self.assertIn("without commentary", prompt)

    def test_markdown_mode_postprocesses_even_when_editing_is_off(self) -> None:
        with (
            patch("voicecommander.pipeline.transcribe", return_value=FinalTranscript("raw description")),
            patch(
                "voicecommander.pipeline.postprocess_openrouter",
                return_value="# Finished",
            ) as postprocess,
        ):
            result = run_pipeline(
                Path("recording.wav"),
                Settings(postprocess_strength=0),
                None,
                "secret",
                markdown=True,
            )
        self.assertEqual(result, "# Finished")
        postprocess.assert_called_once_with(
            "raw description", Settings(postprocess_strength=0), "secret", markdown=True
        )

    def test_markdown_mode_does_not_paste_raw_instructions_on_failure(self) -> None:
        with (
            patch("voicecommander.pipeline.transcribe", return_value=FinalTranscript("raw description")),
            patch(
                "voicecommander.pipeline.postprocess_openrouter",
                side_effect=RuntimeError("cloud unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "cloud unavailable"),
        ):
            run_pipeline(
                Path("recording.wav"),
                Settings(postprocess_strength=0),
                None,
                "secret",
                markdown=True,
            )

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

    @patch("voicecommander.pipeline._openrouter")
    def test_openrouter_transcription_sends_audio_to_a_chat_model(self, openrouter: Mock) -> None:
        openrouter.return_value = {"choices": [{"message": {"content": "Transcript"}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            path.write_bytes(b"RIFF")
            result = transcribe_openrouter(
                path,
                Settings(
                    asr_provider="openrouter",
                    language="de-DE",
                    openrouter_asr_model="google/gemini-2.5-flash",
                ),
                "secret",
            )

        self.assertEqual(result, "Transcript")
        self.assertEqual(
            openrouter.call_args.args[0],
            "/chat/completions",
            "OpenRouter has no /audio/transcriptions route",
        )
        payload = openrouter.call_args.args[2]
        self.assertEqual(payload["model"], "google/gemini-2.5-flash")
        self.assertEqual(payload["provider"], {"data_collection": "deny"})
        parts = payload["messages"][0]["content"]
        self.assertIn("de-DE", parts[0]["text"])
        self.assertEqual(parts[1]["input_audio"], {"data": "UklGRg==", "format": "wav"})

    @patch("voicecommander.pipeline._openrouter")
    def test_openrouter_transcription_can_detect_language_automatically(
        self, openrouter: Mock
    ) -> None:
        openrouter.return_value = {"choices": [{"message": {"content": "Bonjour"}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            path.write_bytes(b"RIFF")
            result = transcribe_openrouter(path, Settings(), "secret")

        self.assertEqual(result, "Bonjour")
        instruction = openrouter.call_args.args[2]["messages"][0]["content"][0]["text"]
        self.assertIn("Detect the spoken language", instruction)
        self.assertNotIn("auto", instruction.casefold())

    @patch("voicecommander.pipeline.urlopen")
    def test_connection_reset_is_retried_then_reported_cleanly(self, urlopen: Mock) -> None:
        urlopen.side_effect = ConnectionResetError(10054, "forcibly closed")
        with (
            self.assertRaisesRegex(RuntimeError, "OpenRouter could not be reached"),
            self.assertLogs("voicecommander.pipeline", level="WARNING"),
        ):
            _openrouter("/chat/completions", "secret", {})
        self.assertEqual(urlopen.call_count, 2)

    def test_zero_editing_strength_keeps_raw_transcript(self) -> None:
        with (
            patch("voicecommander.pipeline.transcribe", return_value=FinalTranscript("raw transcript")),
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

        executor_type.return_value.submit.assert_called_once_with(
            app._finish, Path("recording.wav"), True
        )
        ctrl_pressed.assert_called()

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
