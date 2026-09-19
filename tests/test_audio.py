from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from voicecommander.audio import Recorder, _resolve_input_device


class AudioTests(unittest.TestCase):
    def test_missing_microphone_does_not_open_an_unrelated_device(self) -> None:
        sounddevice = Mock()
        sounddevice.query_devices.return_value = [
            {"name": "Microphone (Web Camera)", "max_input_channels": 2},
        ]
        with patch.dict("sys.modules", {"sounddevice": sounddevice}):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                Recorder("0: Microphone (HyperX SoloCast)").start()
        sounddevice.RawInputStream.assert_not_called()

    def test_matching_device_index_is_preserved_across_host_apis(self) -> None:
        devices = [{"name": "Microphone", "max_input_channels": 2}] * 2
        self.assertEqual(_resolve_input_device("1: Microphone", devices), 1)

    def test_exact_name_takes_priority_over_windows_instance_fallback(self) -> None:
        devices = [
            {"name": "Microphone (HyperX SoloCast)", "max_input_channels": 2},
            {"name": "Microphone (2- HyperX SoloCast)", "max_input_channels": 2},
        ]
        self.assertEqual(_resolve_input_device("0: Microphone (2- HyperX SoloCast)", devices), 1)

    def test_outputs_are_not_selected_as_microphones(self) -> None:
        devices = [{"name": "Headset", "max_input_channels": 0}]
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            _resolve_input_device("0: Headset", devices)

    def test_saved_microphone_follows_name_when_windows_reorders_devices(self) -> None:
        sounddevice = Mock()
        sounddevice.query_devices.return_value = [
            {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
            {"name": "Microphone (HP USB-C Dock Audio", "max_input_channels": 2},
            {"name": "Microphone (Web Camera)", "max_input_channels": 2},
            {"name": "Microphone (HyperX SoloCast)", "max_input_channels": 2},
        ]
        recorder = Recorder("1: Microphone (2- HyperX SoloCast)")

        with patch.dict("sys.modules", {"sounddevice": sounddevice}):
            recorder.start()

        self.assertEqual(sounddevice.RawInputStream.call_args.kwargs["device"], 3)

    def test_recorder_snapshot_does_not_consume_final_audio(self) -> None:
        recorder = Recorder()
        recorder._chunks.extend(b"firstsecond")

        self.assertEqual(recorder.snapshot(), b"firstsecond")
        self.assertEqual(recorder._chunks, bytearray(b"firstsecond"))


if __name__ == "__main__":
    unittest.main()
