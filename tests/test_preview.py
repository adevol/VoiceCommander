from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

from voicecommander.local_asr import PreviewResult, PreviewSegment, PreviewText
from voicecommander.preview import transcribe_live
from voicecommander.settings import Settings


class PreviewTests(unittest.TestCase):
    def test_noise_does_not_lock_language_and_forced_language_is_preserved(self):
        for language, expected in (("auto", [None, "de", "de"]), ("en-US", ["en-US"] * 3)):
            with self.subTest(language=language):
                recorder = Mock()
                recorder.snapshot.return_value = b"pcm"
                model = Mock()
                model.preview.side_effect = [
                    PreviewResult("[Music]", (), 4.0, "fr"),
                    PreviewResult("Hallo", (), 4.0, "de"),
                    PreviewResult("Bonjour", (), 4.0, "fr"),
                ]
                stop = Mock()
                stop.wait.side_effect = [False, False, False, True]
                stop.is_set.return_value = False

                updates = list(transcribe_live(recorder, model, Settings(language=language), stop))

                self.assertEqual([update.language for update in updates], expected)

    @patch("voicecommander.preview.monotonic")
    def test_live_preview_publishes_the_latest_transcript_without_extra_decode_delay(
        self, monotonic: Mock
    ) -> None:
        now = 0.0
        waits = []
        monotonic.side_effect = lambda: now
        recorder = Mock()
        recorder.snapshot.return_value = b"pcm"
        model = Mock()
        results = [
            PreviewResult(
                "Hello brave",
                (
                    PreviewSegment("Hello", 1.0),
                    PreviewSegment(" brave", 3.5),
                ),
                4.0,
                "de",
            ),
            PreviewResult(
                "Hello brave world",
                (
                    PreviewSegment("Hello", 1.0),
                    PreviewSegment(" brave", 3.0),
                    PreviewSegment(" world", 4.5),
                ),
                5.0,
            ),
        ]

        def decode(*args, **kwargs):
            nonlocal now
            now += (0.4, 1.6)[model.preview.call_count - 1]
            return results[model.preview.call_count - 1]

        def wait(timeout):
            nonlocal now
            waits.append(timeout)
            now += timeout
            return len(waits) > 2

        model.preview.side_effect = decode
        stop = Mock()
        stop.wait.side_effect = wait
        stop.is_set.return_value = False

        updates = list(transcribe_live(recorder, model, Settings(), stop))

        self.assertEqual(updates[0], PreviewText("", "Hello brave", 0.0, "de"))
        self.assertEqual(updates[1], PreviewText("Hello", " brave world", 1.0, "de"))
        self.assertEqual(len(waits), 3)
        for actual, expected in zip(waits, (1.0, 0.6, 0.0)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            [call.kwargs["language"] for call in model.preview.call_args_list],
            ["auto", "de"],
        )

    @patch("voicecommander.preview.monotonic", return_value=0)
    def test_live_preview_decodes_bounded_windows_with_absolute_commit_times(
        self, monotonic: Mock
    ) -> None:
        bytes_per_second = 16_000 * 2
        recorder = Mock()
        recording = b"\0" * bytes_per_second * 20
        recorder.snapshot.side_effect = lambda start=0, length=None: recording[
            start : start + length if length is not None else None
        ]
        first_window = PreviewResult(
            "zero one two three",
            (
                PreviewSegment("zero", 2.0),
                PreviewSegment(" one", 4.0),
                PreviewSegment(" two", 7.0),
            ),
            8.0,
        )
        second_window = PreviewResult(
            "one two three four",
            (
                PreviewSegment("one", 2.0),
                PreviewSegment(" two", 4.0),
                PreviewSegment(" three", 6.0),
                PreviewSegment(" four", 7.5),
            ),
            8.0,
        )
        model = Mock()
        model.preview.side_effect = [
            first_window,
            first_window,
            second_window,
            second_window,
        ]
        stop = Mock()
        stop.wait.side_effect = [False, False, False, False, True]
        stop.is_set.return_value = False

        updates = list(transcribe_live(recorder, model, Settings(), stop))

        self.assertTrue(
            all(
                len(preview_call.args[0]) == bytes_per_second * 8
                for preview_call in model.preview.call_args_list
            )
        )
        self.assertEqual(
            [snapshot_call.args for snapshot_call in recorder.snapshot.call_args_list],
            [
                (0, bytes_per_second * 8),
                (0, bytes_per_second * 8),
                (bytes_per_second * 2, bytes_per_second * 8),
                (bytes_per_second * 2, bytes_per_second * 8),
            ],
        )
        self.assertEqual(updates[1].stable_end, 4.0)
        self.assertEqual(updates[3].stable, "zero one two three")
        self.assertEqual(updates[3].stable_end, 8.0)
        stop.wait.assert_has_calls([call(1.0)] * 5)


if __name__ == "__main__":
    unittest.main()
