from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from functools import partial
from io import BytesIO
from pathlib import Path
from sys import modules
from unittest.mock import Mock, call, patch
from urllib.error import HTTPError

from voicecommander.app import State, VoiceCommander, complete_recording
from voicecommander.audio import Recorder, write_wav
from voicecommander.captions import (
    _merge_caption_text,
    _put_latest,
    _transcribe,
    is_speech,
    run_captions,
)
from voicecommander.local_asr import (
    LocalAsrEngine,
    PreviewResult,
    PreviewSegment,
    PreviewText,
    WHISPER_RUNTIME_SHA256,
    _WhisperServer,
    _ensure_whisper,
    _transcribe_whisper,
    load_local_model,
)
from voicecommander.pipeline import (
    _openrouter,
    postprocess_openrouter,
    postprocess_openrouter_stream,
    run_pipeline,
    transcribe_openrouter,
)
from voicecommander.preview import PreviewOverlay, transcribe_live
from voicecommander.settings import (
    WHISPER_MODELS,
    Settings,
    load_settings,
    save_settings,
    validate,
)


def run_first_tk_poll(root: Mock) -> None:
    def mainloop() -> None:
        root.after.call_args.args[1]()
        raise KeyboardInterrupt

    root.mainloop.side_effect = mainloop


class VoiceCommanderTests(unittest.TestCase):
    def test_recorder_snapshot_does_not_consume_final_audio(self) -> None:
        recorder = Recorder()
        recorder._chunks = [b"first", b"second"]

        self.assertEqual(recorder.snapshot(), b"firstsecond")
        self.assertEqual(recorder._chunks, [b"first", b"second"])

    def test_live_preview_publishes_the_latest_transcript(self) -> None:
        recorder = Mock()
        recorder.snapshot.return_value = b"pcm"
        model = Mock()
        model.preview.side_effect = [
            PreviewResult(
                "Hello brave",
                (
                    PreviewSegment("Hello", 0.0, 1.0),
                    PreviewSegment(" brave", 1.0, 3.5),
                ),
                4.0,
                "de",
            ),
            PreviewResult(
                "Hello brave world",
                (
                    PreviewSegment("Hello", 0.0, 1.0),
                    PreviewSegment(" brave", 1.0, 3.0),
                    PreviewSegment(" world", 3.0, 4.5),
                ),
                5.0,
            ),
        ]
        stop = Mock()
        stop.wait.side_effect = [False, False, True]
        stop.is_set.return_value = False
        languages = queue.SimpleQueue()

        updates = list(transcribe_live(recorder, model, Settings(), stop, languages))

        self.assertEqual(updates[0], PreviewText("", "Hello brave", 0.0))
        self.assertEqual(
            updates[1], PreviewText("Hello", " brave world", 1.0)
        )
        self.assertEqual(languages.get_nowait(), "de")
        self.assertEqual(
            [call.args[1].language for call in model.preview.call_args_list],
            ["auto", "de"],
        )

    def test_preview_overlay_consumes_worker_updates(self) -> None:
        tkinter = Mock()
        tkinter.Tk.return_value.winfo_screenwidth.return_value = 1920
        tkinter.Label.return_value.winfo_reqwidth.return_value = 300
        tkinter.Label.return_value.winfo_reqheight.return_value = 40
        updates = queue.SimpleQueue()
        updates.put(PreviewText("Hello", " world", 1.0))

        with patch.dict(modules, {"tkinter": tkinter}):
            overlay = PreviewOverlay()
            overlay.pump(updates)

        tkinter.Label.return_value.configure.assert_called_once_with(text="Hello world")
        tkinter.Tk.return_value.deiconify.assert_called_once()
        tkinter.Tk.return_value.update.assert_not_called()

    @patch("voicecommander.local_asr.urlopen")
    def test_preview_server_parses_timestamped_results(self, urlopen: Mock) -> None:
        urlopen.return_value = BytesIO(
            b'{"text":"Hello world","duration":3.0,"segments":'
            b'[{"text":"Hello","start":0.0,"end":1.0},'
            b'{"text":" world","start":1.0,"end":2.5}],'
            b'"language_probabilities":{"en":0.1,"de":0.9}}'
        )
        server = _WhisperServer(Mock(), "http://127.0.0.1:1")

        result = server.transcribe(b"pcm", Settings())

        self.assertEqual(
            result,
            PreviewResult(
                "Hello world",
                (
                    PreviewSegment("Hello", 0.0, 1.0),
                    PreviewSegment(" world", 1.0, 2.5),
                ),
                3.0,
                "de",
            ),
        )
        self.assertIn(b'verbose_json', urlopen.call_args.args[0].data)

    def test_unknown_local_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unsupported local ASR model: nemotron"):
            load_local_model("nemotron")

    @patch("voicecommander.local_asr._start_whisper_server")
    @patch("voicecommander.local_asr._transcribe_whisper", return_value="Hello")
    @patch("voicecommander.local_asr._ensure_whisper")
    def test_whisper_engine_previews_then_transcribes_authoritatively(
        self, ensure: Mock, transcribe: Mock, start_server: Mock
    ) -> None:
        ensure.return_value = (
            Path("whisper-cli.exe"),
            Path("whisper-server.exe"),
            Path("model.bin"),
        )
        server = start_server.return_value
        preview = PreviewResult("Hallo", (PreviewSegment("Hallo", 0.0, 1.0),), 1.0)
        server.transcribe.return_value = preview
        settings = Settings(language="de-DE")

        engine = load_local_model("base")
        start_server.assert_not_called()
        self.assertEqual(
            engine.preview(b"partial audio", settings),
            preview,
        )
        self.assertEqual(engine.transcribe(Path("recording.wav"), settings), "Hello")
        start_server.assert_called_once_with(
            Path("whisper-server.exe"), Path("model.bin"), engine._preview_cancel
        )
        server.transcribe.assert_called_once_with(b"partial audio", settings)
        transcribe.assert_called_once_with(
            Path("whisper-cli.exe"), Path("model.bin"), Path("recording.wav"), settings
        )
        engine.close()
        server.close.assert_called_once_with()

    @patch("voicecommander.local_asr._start_whisper_server")
    @patch("voicecommander.local_asr._transcribe_whisper", return_value="full fallback")
    def test_whisper_engine_appends_a_matching_preview_tail(
        self, transcribe: Mock, start_server: Mock
    ) -> None:
        server = Mock()
        server.process.poll.return_value = None
        server.transcribe.return_value = PreviewResult(
            "brave new bright world",
            (PreviewSegment("brave new bright world", 0.2, 3.0),),
            3.0,
        )
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        engine._server = server
        path = write_wav(b"\0\0" * 16_000 * 26)
        try:
            text = engine.transcribe(
                path, Settings(), PreviewText("Hello brave new bright", "", 24.0)
            )
        finally:
            path.unlink()

        self.assertEqual(text, "Hello brave new bright world")
        transcribe.assert_not_called()
        start_server.assert_not_called()
        tail = server.transcribe.call_args.args[0]
        self.assertEqual(len(tail), 16_000 * 2 * 4)

    @patch("voicecommander.local_asr._start_whisper_server")
    @patch("voicecommander.local_asr._transcribe_whisper", return_value="full fallback")
    def test_whisper_engine_falls_back_when_the_preview_tail_does_not_match(
        self, transcribe: Mock, start_server: Mock
    ) -> None:
        start_server.return_value.transcribe.return_value = PreviewResult(
            "different words", (), 3.0
        )
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        path = write_wav(b"\0\0" * 16_000 * 26)
        try:
            text = engine.transcribe(
                path, Settings(), PreviewText("Hello brave new", "", 24.0)
            )
        finally:
            path.unlink()

        self.assertEqual(text, "full fallback")
        transcribe.assert_called_once_with(
            Path("cli.exe"), Path("model.bin"), path, Settings()
        )

    @patch("voicecommander.local_asr._transcribe_whisper", return_value="full fallback")
    def test_whisper_engine_rejects_a_match_after_the_overlap(
        self, transcribe: Mock
    ) -> None:
        server = Mock()
        server.process.poll.return_value = None
        server.transcribe.return_value = PreviewResult(
            "and the actual tail",
            (PreviewSegment("and the actual tail", 2.1, 4.0),),
            4.0,
        )
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        engine._server = server
        path = write_wav(b"\0\0" * 16_000 * 26)
        try:
            text = engine.transcribe(
                path, Settings(), PreviewText("Earlier words and the", "", 24.0)
            )
        finally:
            path.unlink()

        self.assertEqual(text, "full fallback")
        transcribe.assert_called_once()

    @patch("voicecommander.local_asr._transcribe_whisper", return_value="full transcript")
    def test_short_preview_uses_the_full_decode(self, transcribe: Mock) -> None:
        server = Mock()
        server.process.poll.return_value = None
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        engine._server = server
        path = write_wav(b"\0\0" * 16_000 * 6)
        try:
            text = engine.transcribe(path, Settings(), PreviewText("Hello", "", 4.0))
        finally:
            path.unlink()

        self.assertEqual(text, "full transcript")
        server.transcribe.assert_not_called()
        transcribe.assert_called_once()

    @patch("voicecommander.local_asr._transcribe_whisper", return_value="full fallback")
    def test_tail_decoder_failure_falls_back_to_full_decode(
        self, transcribe: Mock
    ) -> None:
        server = Mock()
        server.process.poll.return_value = None
        server.transcribe.side_effect = RuntimeError("decoder stopped")
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        engine._server = server
        path = write_wav(b"\0\0" * 16_000 * 26)
        try:
            with self.assertLogs("voicecommander.local_asr", level="WARNING"):
                text = engine.transcribe(
                    path, Settings(), PreviewText("Long stable prefix", "", 24.0)
                )
        finally:
            path.unlink()

        self.assertEqual(text, "full fallback")
        transcribe.assert_called_once()

    @patch("voicecommander.local_asr._transcribe_whisper")
    def test_tail_programming_error_is_not_swallowed(self, transcribe: Mock) -> None:
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        path = write_wav(b"\0\0" * 16_000 * 26)
        try:
            with (
                patch.object(
                    LocalAsrEngine,
                    "_decode_pcm",
                    side_effect=ValueError("programming error"),
                ),
                self.assertRaisesRegex(ValueError, "programming error"),
            ):
                engine.transcribe(
                    path, Settings(), PreviewText("Long stable prefix", "", 24.0)
                )
        finally:
            path.unlink()

        transcribe.assert_not_called()

    @patch("voicecommander.local_asr._start_whisper_server")
    @patch("voicecommander.local_asr._ensure_whisper")
    def test_whisper_preview_drops_concurrent_requests(
        self, ensure: Mock, start_server: Mock
    ) -> None:
        ensure.return_value = Path("cli.exe"), Path("server.exe"), Path("model.bin")
        entered = threading.Event()
        release = threading.Event()

        preview = PreviewResult("first", (PreviewSegment("first", 0.0, 1.0),), 1.0)

        def transcribe(data: bytes, settings: Settings) -> PreviewResult:
            entered.set()
            release.wait(1)
            return preview

        start_server.return_value.transcribe.side_effect = transcribe
        engine = load_local_model("base")
        results = []
        worker = threading.Thread(
            target=lambda: results.append(engine.preview(b"first", Settings()))
        )
        worker.start()
        self.assertTrue(entered.wait(1))

        self.assertIsNone(engine.preview(b"newest", Settings()))

        release.set()
        worker.join(1)
        self.assertEqual(results, [preview])
        engine.close()

    def test_cancel_preview_stops_the_warm_server(self) -> None:
        server = Mock()
        model = LocalAsrEngine(Path("cli.exe"), Path("server.exe"), Path("model.bin"))
        model._server = server

        model.cancel_preview()

        self.assertTrue(model._preview_cancel.is_set())
        server.close.assert_called_once_with()

    def test_local_pipeline_refines_only_the_final_transcript(self) -> None:
        settings = Settings(postprocess_strength=50)
        engine = Mock()
        engine.transcribe.return_value = "raw transcript"
        preview = PreviewText("stable words", " tentative", 3.0)

        with patch(
            "voicecommander.pipeline.postprocess_openrouter", return_value="edited transcript"
        ) as postprocess:
            result = run_pipeline(
                Path("recording.wav"), settings, engine, "secret", preview=preview
            )

        self.assertEqual(result, "edited transcript")
        engine.transcribe.assert_called_once_with(Path("recording.wav"), settings, preview)
        postprocess.assert_called_once_with(
            "raw transcript", settings, "secret", markdown=False
        )

    @patch("voicecommander.local_asr._download")
    def test_each_whisper_size_downloads_its_own_verified_file(self, download: Mock) -> None:
        for name, (filename, expected_sha256) in WHISPER_MODELS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                whisper_dir = Path(directory)
                for runtime_file in (
                    "whisper-cli.exe",
                    "whisper-server.exe",
                    "whisper.dll",
                    "ggml.dll",
                ):
                    (whisper_dir / runtime_file).touch()
                (whisper_dir / f".runtime-{WHISPER_RUNTIME_SHA256}").touch()
                download.reset_mock()

                with patch("voicecommander.local_asr.WHISPER_DIR", whisper_dir):
                    _, server, model = _ensure_whisper(name)

                url, destination, sha256 = download.call_args.args
                self.assertEqual(server.name, "whisper-server.exe")
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
            caption_asr_model="tiny",
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
        with self.assertRaisesRegex(ValueError, "Invalid caption ASR model"):
            validate(Settings(caption_asr_model="nemotron-3.5"))
        with self.assertRaisesRegex(ValueError, "at most 500 characters"):
            validate(Settings(vocabulary="x" * 501))
        with self.assertRaisesRegex(ValueError, "hotkeys must be different"):
            validate(Settings(hotkey="F8", markdown_hotkey="f8"))

    def test_caption_noise_labels_are_dropped(self) -> None:
        self.assertTrue(is_speech("Hello there"))
        for noise in ("", "[BLANK_AUDIO]", "(music)", "[Musik]"):
            self.assertFalse(is_speech(noise))

    def test_caption_text_merges_overlapping_words(self) -> None:
        cases = (
            ("", "Hello world", "Hello world"),
            ("Hello world", "hello world", "Hello world"),
            ("Hello, world!", "WORLD again", "Hello, world! again"),
            ("one two three", "two three four", "one two three four"),
            ("hello", "there", "hello there"),
        )
        for previous, current, expected in cases:
            with self.subTest(previous=previous, current=current):
                self.assertEqual(_merge_caption_text(previous, current), expected)

    def test_captions_drop_backlog(self) -> None:
        chunks = queue.Queue(maxsize=1)
        _put_latest(chunks, "old")
        _put_latest(chunks, "new")
        self.assertEqual(chunks.get_nowait(), "new")

    def test_caption_worker_skips_silence_and_closes_its_model(self) -> None:
        chunks = queue.Queue()
        chunks.put((b"silence", 0.001))
        chunks.put((b"speech", 0.5))
        lines = queue.Queue()
        stop = threading.Event()
        model = Mock()

        def finish(*_args) -> PreviewResult:
            stop.set()
            return PreviewResult("Hello", (PreviewSegment("Hello", 0.0, 1.0),), 1.0)

        model.preview.side_effect = finish
        with patch("voicecommander.captions.load_local_model", return_value=model) as load:
            _transcribe(chunks, lines, Settings(caption_asr_model="tiny"), stop)

        load.assert_called_once_with("tiny")
        model.preview.assert_called_once_with(
            b"speech", Settings(caption_asr_model="tiny")
        )
        model.close.assert_called_once_with()
        self.assertEqual(lines.get_nowait(), "(listening)")
        self.assertEqual(lines.get_nowait(), "Hello")

    def test_caption_window_joins_workers_on_shutdown(self) -> None:
        tkinter = Mock()
        run_first_tk_poll(tkinter.Tk.return_value)
        workers = [Mock(), Mock()]
        with (
            patch.dict(modules, {"tkinter": tkinter}),
            patch("voicecommander.captions.threading.Thread", side_effect=workers) as thread,
            self.assertRaises(KeyboardInterrupt),
        ):
            run_captions(Settings())

        self.assertEqual(thread.call_count, 2)
        for worker in workers:
            worker.start.assert_called_once_with()
            worker.join.assert_called_once_with()
        stop = thread.call_args_list[0].kwargs["args"][2]
        self.assertTrue(stop.is_set())

    def test_caption_worker_stops_after_model_load_failure(self) -> None:
        lines = queue.Queue()
        stop = threading.Event()
        with (
            patch(
                "voicecommander.captions.load_local_model",
                side_effect=RuntimeError("model failed"),
            ),
            self.assertLogs("voicecommander.captions", level="ERROR"),
        ):
            _transcribe(queue.Queue(), lines, Settings(), stop)
        self.assertTrue(stop.is_set())
        self.assertIn("model failed", lines.get_nowait())

    def test_postprocessing_failure_falls_back_to_raw_transcript(self) -> None:
        settings = Settings(asr_provider="openrouter", postprocess_strength=50)

        with (
            patch(
                "voicecommander.pipeline.transcribe_openrouter",
                return_value="raw transcript",
            ),
            patch(
                "voicecommander.pipeline.postprocess_openrouter",
                side_effect=RuntimeError("cloud unavailable"),
            ),
            self.assertLogs("voicecommander.pipeline", level="ERROR"),
        ):
            result = run_pipeline(Path("recording.wav"), settings, None, "secret")
        self.assertEqual(result, "raw transcript")

    @patch("voicecommander.pipeline.monotonic", side_effect=[0.0, 0.1, 0.2])
    @patch("voicecommander.pipeline.urlopen")
    def test_postprocessing_stream_accumulates_text(
        self, urlopen: Mock, monotonic: Mock
    ) -> None:
        urlopen.return_value = BytesIO(
            b": OPENROUTER PROCESSING\n\n"
            b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        updates = []

        result = postprocess_openrouter_stream(
            "Raw", Settings(), "secret", updates.append
        )

        self.assertEqual(result, "Hello world")
        self.assertEqual(updates, ["Hello ", "Hello world"])
        request = urlopen.call_args.args[0]
        self.assertIn(b'"stream": true', request.data)

    @patch("voicecommander.pipeline.postprocess_openrouter", return_value="Edited")
    @patch(
        "voicecommander.pipeline.postprocess_openrouter_stream",
        side_effect=RuntimeError("stream failed"),
    )
    def test_stream_failure_retries_without_streaming(
        self, stream: Mock, postprocess: Mock
    ) -> None:
        engine = Mock()
        engine.transcribe.return_value = "Raw"
        updates = Mock()
        settings = Settings(postprocess_strength=50)

        with self.assertLogs("voicecommander.pipeline", level="ERROR"):
            result = run_pipeline(
                Path("recording.wav"),
                settings,
                engine,
                "secret",
                on_refine=updates,
            )

        self.assertEqual(result, "Edited")
        self.assertEqual(updates.call_args_list, [call(""), call("Edited")])
        stream.assert_called_once_with("Raw", settings, "secret", updates, markdown=False)
        postprocess.assert_called_once_with("Raw", settings, "secret", markdown=False)

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
        settings = Settings(asr_provider="openrouter", postprocess_strength=0)
        with (
            patch(
                "voicecommander.pipeline.transcribe_openrouter",
                return_value="raw description",
            ),
            patch(
                "voicecommander.pipeline.postprocess_openrouter",
                return_value="# Finished",
            ) as postprocess,
        ):
            result = run_pipeline(
                Path("recording.wav"),
                settings,
                None,
                "secret",
                markdown=True,
            )
        self.assertEqual(result, "# Finished")
        postprocess.assert_called_once_with(
            "raw description", settings, "secret", markdown=True
        )

    def test_markdown_mode_does_not_paste_raw_instructions_on_failure(self) -> None:
        settings = Settings(asr_provider="openrouter", postprocess_strength=0)
        with (
            patch(
                "voicecommander.pipeline.transcribe_openrouter",
                return_value="raw description",
            ),
            patch(
                "voicecommander.pipeline.postprocess_openrouter",
                side_effect=RuntimeError("cloud unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "cloud unavailable"),
        ):
            run_pipeline(
                Path("recording.wav"),
                settings,
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
        settings = Settings(asr_provider="openrouter", postprocess_strength=0)
        with (
            patch(
                "voicecommander.pipeline.transcribe_openrouter",
                return_value="raw transcript",
            ),
            patch("voicecommander.pipeline.postprocess_openrouter") as postprocess,
        ):
            result = run_pipeline(
                Path("recording.wav"),
                settings,
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

            app._finish(Path("recording.wav"), settings=settings)

        self.assertEqual(app._preview_updates.get_nowait(), "Refining")
        self.assertEqual(app._preview_updates.get_nowait(), "Edited text")
        self.assertIsNone(app._preview_updates.get_nowait())
        self.assertEqual(app.state, State.IDLE)

    @patch("voicecommander.app.threading.Timer")
    @patch("voicecommander.app.ThreadPoolExecutor")
    @patch("voicecommander.app.Recorder")
    @patch("voicecommander.app.threading.Thread")
    def test_each_preview_worker_keeps_its_own_stop_event(
        self,
        thread_type: Mock,
        recorder_type: Mock,
        executor_type: Mock,
        timer: Mock,
    ) -> None:
        recorder_type.return_value.stop.return_value = Path("recording.wav")
        model_future = executor_type.return_value.submit.return_value
        model_future.done.return_value = True
        model_future.cancelled.return_value = False
        model_future.exception.return_value = None
        app = VoiceCommander(Settings())

        app.on_hotkey()

        first_stop = app._preview_stop
        first_languages = app._preview_language
        self.assertEqual(app._preview_updates.get_nowait(), "Listening")
        thread_type.assert_called_once_with(
            target=app._preview, args=(first_stop, first_languages), daemon=True
        )
        thread_type.return_value.start.assert_called_once()

        first_languages.put("de")
        committed = PreviewText("Hallo", " Welt", 2.5)
        app._preview_commit = committed
        app.on_hotkey()

        self.assertTrue(first_stop.is_set())
        self.assertEqual(app._preview_updates.get_nowait(), "Finalizing")
        model_future.result.return_value.cancel_preview.assert_not_called()
        finish_args = executor_type.return_value.submit.call_args.args
        self.assertEqual(finish_args[-2].language, "de")
        self.assertEqual(finish_args[-1], committed)

        app.state = State.IDLE
        app.on_hotkey()

        second_stop = app._preview_stop
        second_languages = app._preview_language
        self.assertIsNot(first_stop, second_stop)
        self.assertIsNot(first_languages, second_languages)
        self.assertTrue(first_stop.is_set())
        self.assertFalse(second_stop.is_set())

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
            app._finish,
            Path("recording.wav"),
            True,
            Settings(asr_provider="openrouter"),
            None,
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
            app.run()
            app.on_hotkey()

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

    @patch("voicecommander.pipeline.urlopen")
    def test_openrouter_error_includes_response_detail(self, urlopen: Mock) -> None:
        urlopen.side_effect = HTTPError(
            "https://openrouter.ai", 400, "Bad Request", {}, BytesIO(b"invalid model")
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 400: invalid model"):
            _openrouter("/test", "secret", {})


if __name__ == "__main__":
    unittest.main()
