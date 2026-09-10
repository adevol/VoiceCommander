from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, call, patch

import httpx
from openrouter import OpenRouter

from voicecommander.pipeline import (
    _openrouter,
    postprocess_openrouter,
    postprocess_openrouter_stream,
    refine_transcript,
    transcribe_openrouter,
)
from voicecommander.settings import Settings


class PipelineTests(unittest.TestCase):
    @contextmanager
    def cloud(self, handler):
        requests = []

        def respond(request):
            requests.append(request)
            return handler(request)

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with patch(
                "voicecommander.pipeline.OpenRouter",
                side_effect=lambda **kwargs: OpenRouter(client=client, **kwargs),
            ):
                yield requests

    @staticmethod
    def stream_event(content=None, finish_reason=None, **extra):
        chunk = {
            "id": "test",
            "model": "test-model",
            "created": 0,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content} if content is not None else {},
                    "finish_reason": finish_reason,
                }
            ],
        } | extra
        return f"data: {json.dumps(chunk)}\n\n".encode()

    def test_refinement_receives_only_the_final_transcript(self) -> None:
        settings = Settings(postprocess_strength=50)
        with patch(
            "voicecommander.pipeline.postprocess_openrouter", return_value="edited transcript"
        ) as postprocess:
            result = refine_transcript("raw transcript", settings, "secret")
        self.assertEqual(result, "edited transcript")
        postprocess.assert_called_once_with("raw transcript", settings, "secret", markdown=False)

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
    def test_postprocessing_stream_accumulates_text(self, monotonic: Mock) -> None:
        response = httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b": OPENROUTER PROCESSING\n\n"
            + self.stream_event("Hello ")
            + self.stream_event("world")
            + self.stream_event(finish_reason="stop")
            + b"data: [DONE]\n\n",
        )
        updates = []

        with self.cloud(lambda request: response) as requests:
            result = postprocess_openrouter_stream("Raw", Settings(), "secret", updates.append)

        self.assertEqual(result, "Hello world")
        self.assertEqual(updates, ["Hello ", "Hello world"])
        payload = json.loads(requests[0].content)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["provider"], {"data_collection": "deny"})
        self.assertTrue(response.is_closed)

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
        openrouter.return_value = Mock(choices=[Mock(message=Mock(content="Edited"))])
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
        openrouter.return_value = Mock(choices=[Mock(message=Mock(content="## Result"))])

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
        postprocess.assert_called_once_with("raw description", settings, "secret", markdown=True)

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
        openrouter.return_value = Mock(choices=[Mock(message=Mock(content="Transcript"))])
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
        openrouter.return_value = Mock(choices=[Mock(message=Mock(content="Bonjour"))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            path.write_bytes(b"RIFF")
            result = transcribe_openrouter(path, Settings(), "secret")

        self.assertEqual(result, "Bonjour")
        instruction = openrouter.call_args.args[1]["messages"][0]["content"][0]["text"]
        self.assertIn("Detect the spoken language", instruction)
        self.assertNotIn("auto", instruction.casefold())

    def test_connection_reset_is_retried_then_reported_cleanly(self) -> None:
        def reset(request):
            raise httpx.ReadError("forcibly closed", request=request)

        with (
            self.cloud(reset) as requests,
            self.assertRaisesRegex(RuntimeError, "OpenRouter could not be reached"),
            self.assertLogs("voicecommander.pipeline", level="WARNING"),
        ):
            _openrouter("secret", {"messages": []})
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            str(requests[0].url),
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

    def test_openrouter_error_includes_response_detail(self) -> None:
        for status in (400, 500):
            with (
                self.subTest(status=status),
                self.cloud(
                    lambda request: httpx.Response(status, text="invalid model")
                ) as requests,
                self.assertRaisesRegex(RuntimeError, f"HTTP {status}: invalid model"),
            ):
                _openrouter("secret", {"messages": []})
            self.assertEqual(len(requests), 1)

    def test_sdk_preserves_audio_model_and_privacy_and_retries_connection_failure(self) -> None:
        def respond(request):
            if len(requests) == 1:
                raise httpx.ConnectError("connection failed", request=request)
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "model": "google/test-model",
                    "created": 0,
                    "object": "chat.completion",
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": " Bonjour "},
                        }
                    ],
                },
            )

        with tempfile.TemporaryDirectory() as directory, self.cloud(respond) as requests:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"RIFF")
            with self.assertLogs("voicecommander.pipeline", level="WARNING"):
                result = transcribe_openrouter(
                    path, Settings(openrouter_asr_model="google/test-model"), "secret"
                )
        self.assertEqual(result, "Bonjour")
        self.assertEqual(len(requests), 2)
        payload = json.loads(requests[-1].content)
        self.assertEqual(payload["model"], "google/test-model")
        self.assertEqual(payload["provider"], {"data_collection": "deny"})
        self.assertEqual(
            payload["messages"][0]["content"][1]["input_audio"],
            {"data": "UklGRg==", "format": "wav"},
        )
        self.assertEqual(requests[-1].headers["Authorization"], "Bearer secret")
        self.assertEqual(requests[-1].extensions["timeout"]["read"], 300)

    def test_incomplete_invalid_and_failed_streams_are_rejected(self) -> None:
        for body, message in (
            (self.stream_event("Partial"), "before completion"),
            (self.stream_event("Partial") + b"data: [DONE]\n\n", "before completion"),
            (self.stream_event(finish_reason="stop"), "empty"),
            (self.stream_event("Partial", finish_reason="error"), "stream failed"),
            (b'data: {"choices": "invalid"}\n\n', "invalid response"),
            (
                self.stream_event(error={"code": 500, "message": "provider failed"}),
                "provider failed",
            ),
        ):
            response = httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=body
            )
            with (
                self.subTest(body=body),
                self.cloud(lambda request: response),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                postprocess_openrouter_stream("Raw", Settings(), "secret", Mock())
            self.assertTrue(response.is_closed)

    def test_missing_api_key_does_not_make_a_request(self) -> None:
        with patch("voicecommander.pipeline.OpenRouter") as client:
            with self.assertRaisesRegex(RuntimeError, "no API key"):
                _openrouter("", {"messages": []})
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
