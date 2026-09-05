from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, call, patch
from urllib.error import HTTPError

from voicecommander.pipeline import (
    _openrouter,
    postprocess_openrouter,
    postprocess_openrouter_stream,
    refine_transcript,
    transcribe_openrouter,
)
from voicecommander.settings import Settings


class PipelineTests(unittest.TestCase):
    def test_refinement_receives_only_the_final_transcript(self) -> None:
        settings = Settings(postprocess_strength=50)
        with patch(
            "voicecommander.pipeline.postprocess_openrouter", return_value="edited transcript"
        ) as postprocess:
            result = refine_transcript("raw transcript", settings, "secret")
        self.assertEqual(result, "edited transcript")
        postprocess.assert_called_once_with(
            "raw transcript", settings, "secret", markdown=False
        )

    def test_postprocessing_failure_falls_back_to_raw_transcript(self) -> None:
        settings = Settings(asr_provider="openrouter", postprocess_strength=50)

        with (
            patch(
                "voicecommander.pipeline.postprocess_openrouter",
                side_effect=RuntimeError("cloud unavailable"),
            ),
            self.assertLogs("voicecommander.pipeline", level="ERROR"),
        ):
            result = refine_transcript("raw transcript", settings, "secret")
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
        updates = Mock()
        settings = Settings(postprocess_strength=50)

        with self.assertLogs("voicecommander.pipeline", level="ERROR"):
            result = refine_transcript(
                "Raw",
                settings,
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
        prompt = openrouter.call_args.args[1]["messages"][0]["content"]
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
        prompt = openrouter.call_args.args[1]["messages"][0]["content"]
        self.assertIn("finished Markdown", prompt)
        self.assertIn("$...$", prompt)
        self.assertIn("$$...$$", prompt)
        self.assertIn("without commentary", prompt)

    def test_markdown_mode_postprocesses_even_when_editing_is_off(self) -> None:
        settings = Settings(asr_provider="openrouter", postprocess_strength=0)
        with (
            patch(
                "voicecommander.pipeline.postprocess_openrouter",
                return_value="# Finished",
            ) as postprocess,
        ):
            result = refine_transcript(
                "raw description",
                settings,
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
                "voicecommander.pipeline.postprocess_openrouter",
                side_effect=RuntimeError("cloud unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "cloud unavailable"),
        ):
            refine_transcript(
                "raw description",
                settings,
                "secret",
                markdown=True,
            )

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
        payload = openrouter.call_args.args[1]
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
        instruction = openrouter.call_args.args[1]["messages"][0]["content"][0]["text"]
        self.assertIn("Detect the spoken language", instruction)
        self.assertNotIn("auto", instruction.casefold())

    @patch("voicecommander.pipeline.urlopen")
    def test_connection_reset_is_retried_then_reported_cleanly(self, urlopen: Mock) -> None:
        urlopen.side_effect = ConnectionResetError(10054, "forcibly closed")
        with (
            self.assertRaisesRegex(RuntimeError, "OpenRouter could not be reached"),
            self.assertLogs("voicecommander.pipeline", level="WARNING"),
        ):
            _openrouter("secret", {})
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(
            urlopen.call_args.args[0].full_url,
            "https://openrouter.ai/api/v1/chat/completions",
        )

    def test_zero_editing_strength_keeps_raw_transcript(self) -> None:
        settings = Settings(asr_provider="openrouter", postprocess_strength=0)
        with (
            patch("voicecommander.pipeline.postprocess_openrouter") as postprocess,
        ):
            result = refine_transcript(
                "raw transcript",
                settings,
                "secret",
            )
        self.assertEqual(result, "raw transcript")
        postprocess.assert_not_called()

    @patch("voicecommander.pipeline.urlopen")
    def test_openrouter_error_includes_response_detail(self, urlopen: Mock) -> None:
        urlopen.side_effect = HTTPError(
            "https://openrouter.ai", 400, "Bad Request", {}, BytesIO(b"invalid model")
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 400: invalid model"):
            _openrouter("secret", {})


if __name__ == "__main__":
    unittest.main()
