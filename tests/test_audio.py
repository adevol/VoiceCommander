from __future__ import annotations

import unittest

from voicecommander.audio import Recorder


class AudioTests(unittest.TestCase):
    def test_recorder_snapshot_does_not_consume_final_audio(self) -> None:
        recorder = Recorder()
        recorder._chunks.extend(b"firstsecond")

        self.assertEqual(recorder.snapshot(), b"firstsecond")
        self.assertEqual(recorder._chunks, bytearray(b"firstsecond"))


if __name__ == "__main__":
    unittest.main()
