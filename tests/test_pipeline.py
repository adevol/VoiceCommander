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
    NEMOTRON_MODEL,
    _openrouter,
    load_local_model,
    postprocess_openrouter,
    run_pipeline,
    transcribe_local,
    transcribe_openrouter,
)
from voicecommander.settings import Settings, load_settings, save_settings


class EssentialTests(unittest.TestCase):
    def test_local_model_uses_half_precision_on_cuda(self) -> None:
        torch = Mock(float16="float16", float32="float32")
        torch.cuda.is_available.return_value = True
        model = Mock()
        model_type = Mock()
        model_type.from_pretrained.return_value = model
        transformers = Mock(
            AutoModelForRNNT=model_type,
            AutoProcessor=Mock(),
        )

        with patch.dict(modules, {"torch": torch, "transformers": transformers}):
            load_local_model("nemotron")

        model_type.from_pretrained.assert_called_once_with(
            NEMOTRON_MODEL, dtype="float16"
        )
        model.to.assert_called_once_with("cuda")

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
                ("whisper", Path("whisper-cli.exe"), Path("ggml-base-q5_1.bin")),
            )

        self.assertEqual(result, "Guten Tag")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-l") + 1], "de")

    def test_settings_round_trip_without_secrets(self) -> None:
        settings = Settings(
            hotkey="f9",
            input_device='2: Mic "Main"',
            max_seconds=42,
            local_asr_model="nemotron",
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
                config.write('asr_model = "obsolete"\n')
            self.assertEqual(load_settings(path), settings)
            with path.open("a", encoding="utf-8") as config:
                config.write('hotkeys = "f9"\n')
            with self.assertRaisesRegex(ValueError, "Unknown settings: hotkeys"):
                load_settings(path)

    def test_existing_local_config_keeps_nemotron(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('asr_provider = "local"\n', encoding="utf-8")
            self.assertEqual(load_settings(path).local_asr_model, "nemotron")

    def test_removed_postprocess_provider_setting_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                'postprocess_provider = "none"\npostprocess_strength = 60\n', encoding="utf-8"
            )
            self.assertEqual(load_settings(path).postprocess_strength, 0)
            path.write_text(
                'postprocess_provider = "openrouter"\npostprocess_strength = 60\n', encoding="utf-8"
            )
            self.assertEqual(load_settings(path).postprocess_strength, 60)

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
        self.assertEqual(openrouter.call_args.args[2]["model"], "google/chirp-3")

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

    @patch("voicecommander.pipeline.urlopen")
    def test_openrouter_error_includes_response_detail(self, urlopen: Mock) -> None:
        urlopen.side_effect = HTTPError(
            "https://openrouter.ai", 400, "Bad Request", {}, BytesIO(b"invalid model")
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 400: invalid model"):
            _openrouter("/test", "secret", {})


if __name__ == "__main__":
    unittest.main()
