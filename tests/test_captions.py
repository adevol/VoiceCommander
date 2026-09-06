from __future__ import annotations

import queue
import threading
import unittest
from sys import modules
from unittest.mock import Mock, patch

from voicecommander.captions import (
    _merge_caption_text,
    _put_latest,
    _transcribe,
    is_speech,
    run_captions,
)
from voicecommander.local_asr import PreviewResult, PreviewSegment
from voicecommander.settings import Settings


def run_first_tk_poll(root: Mock) -> None:
    def mainloop() -> None:
        root.after.call_args.args[1]()
        raise KeyboardInterrupt

    root.mainloop.side_effect = mainloop


class CaptionsTests(unittest.TestCase):
    def test_caption_noise_labels_are_dropped(self) -> None:
        self.assertTrue(is_speech("Hello there"))
        for noise in ("", "[BLANK_AUDIO]", "(music)", "[Musik]"):
            self.assertFalse(is_speech(noise))

    def test_caption_text_merges_overlapping_words(self) -> None:
        cases = (
            ("", "Hello world", "Hello world"),
            ("Hello world", "hello world", "Hello world"),
            ("Hello, world!", "WORLD again", "Hello, world! again"),
            ("你好， 世界！", "世界？ again", "你好， 世界！ again"),
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

        def finish(*_args, **_kwargs) -> PreviewResult:
            stop.set()
            return PreviewResult("Hello", (PreviewSegment("Hello", 1.0),), 1.0)

        model.preview.side_effect = finish
        with patch("voicecommander.captions.load_local_model", return_value=model) as load:
            _transcribe(chunks, lines, Settings(caption_asr_model="tiny"), stop)

        load.assert_called_once_with("tiny")
        model.preview.assert_called_once_with(
            b"speech", language="auto", vocabulary="", stop=stop
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


if __name__ == "__main__":
    unittest.main()
