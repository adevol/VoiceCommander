from __future__ import annotations

import os
import tempfile
import threading
import unittest
import wave
from functools import partial
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, call, patch

from voicecommander.audio import write_wav
from voicecommander.local_asr import (
    LocalAsrEngine,
    PreviewResult,
    PreviewSegment,
    PreviewText,
    WHISPER_DIR,
    WHISPER_RUNTIME_SHA256,
    _WhisperServer,
    _ensure_whisper,
    _transcribe_whisper,
    load_local_model,
)
from voicecommander.pipeline import postprocess_openrouter, transcribe_openrouter
from voicecommander.settings import WHISPER_MODELS, Settings


class LocalAsrTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("VOICECOMMANDER_INTEGRATION_AUDIO"),
        "set VOICECOMMANDER_INTEGRATION_AUDIO to run the real Whisper probe",
    )
    def test_real_whisper_server_reuses_preview_for_finalization(self) -> None:
        audio = Path(os.environ["VOICECOMMANDER_INTEGRATION_AUDIO"])
        model_name = os.environ.get("VOICECOMMANDER_INTEGRATION_MODEL", "base")
        executable = WHISPER_DIR / "whisper-cli.exe"
        server_executable = WHISPER_DIR / "whisper-server.exe"
        model_path = WHISPER_DIR / WHISPER_MODELS[model_name][0]
        if not all(path.exists() for path in (audio, executable, server_executable, model_path)):
            self.skipTest("integration audio or installed Whisper files are missing")
        with wave.open(str(audio), "rb") as source:
            seconds = source.getnframes() / source.getframerate()
            pcm16 = source.readframes(source.getnframes())
        prefix_seconds = 22
        silence = b"\0" * 16_000 * 2 * prefix_seconds
        final_path = write_wav(silence + pcm16 * 2)
        settings = Settings(local_asr_model=model_name, language="en-US")
        engine = LocalAsrEngine(executable, server_executable, model_path)
        try:
            preview = engine.preview(pcm16, settings)
            self.assertIsNotNone(preview)
            self.assertTrue(preview.segments)
            process = engine._server.process
            with patch(
                "voicecommander.local_asr._transcribe_whisper",
                side_effect=AssertionError("warm tail finalization fell back"),
            ):
                final = engine.transcribe(
                    final_path,
                    settings,
                    PreviewText(preview.text, "", prefix_seconds + seconds),
                )
            self.assertIs(engine._server.process, process)
            self.assertGreater(len(final.split()), len(preview.text.split()))
        finally:
            engine.close()
            final_path.unlink()
        self.assertIsNotNone(process.poll())

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
                    PreviewSegment("Hello", 1.0),
                    PreviewSegment(" world", 2.5),
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
        preview = PreviewResult("Hallo", (PreviewSegment("Hallo", 1.0),), 1.0)
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
        prefix_words = [f"word{number}" for number in range(60)]
        prefix = " ".join(prefix_words)
        server = Mock()
        server.process.poll.return_value = None
        server.transcribe.return_value = PreviewResult(
            "word57 word58 word59 ending",
            (PreviewSegment("word57 word58 word59 ending", 3.0),),
            3.0,
        )
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        engine._server = server
        settings = Settings(vocabulary="VoiceCommander")
        path = write_wav(b"\0\0" * 16_000 * 26)
        try:
            text = engine.transcribe(
                path, settings, PreviewText(prefix, "", 24.0)
            )
        finally:
            path.unlink()

        self.assertEqual(text, f"{prefix} ending")
        transcribe.assert_not_called()
        start_server.assert_not_called()
        tail = server.transcribe.call_args.args[0]
        self.assertEqual(len(tail), 16_000 * 2 * 4)
        self.assertEqual(
            server.transcribe.call_args.args[1].vocabulary,
            f"VoiceCommander {' '.join(prefix_words[-50:])}",
        )

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
        start_server.assert_not_called()

    @patch("voicecommander.local_asr._transcribe_whisper", return_value="full fallback")
    def test_whisper_engine_rejects_a_tail_with_no_new_words(
        self, transcribe: Mock
    ) -> None:
        server = Mock()
        server.process.poll.return_value = None
        server.transcribe.return_value = PreviewResult(
            "Hello brave new",
            (PreviewSegment("Hello brave new", 2.0),),
            2.0,
        )
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        engine._server = server
        path = write_wav(b"\0\0" * 16_000 * 26)
        try:
            text = engine.transcribe(
                path, Settings(), PreviewText("Hello brave new", "", 24.0)
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
        server = Mock()
        server.process.poll.return_value = None
        engine = LocalAsrEngine(
            Path("cli.exe"), Path("server.exe"), Path("model.bin")
        )
        engine._server = server
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

        preview = PreviewResult("first", (PreviewSegment("first", 1.0),), 1.0)

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

    def test_close_stops_the_warm_server(self) -> None:
        server = Mock()
        model = LocalAsrEngine(Path("cli.exe"), Path("server.exe"), Path("model.bin"))
        model._server = server

        model.close()

        self.assertTrue(model._preview_cancel.is_set())
        server.close.assert_called_once_with()

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
                self.assertIn("Kubernetes, VoiceCommander", str(call.args[1]["messages"]))

            transcript.write_text("Kubernetes", encoding="utf-8")
            model(path, Settings())
        self.assertNotIn("--prompt", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
