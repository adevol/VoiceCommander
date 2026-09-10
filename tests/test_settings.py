from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voicecommander.settings import Settings, load_settings, save_settings, validate


class SettingsTests(unittest.TestCase):
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
                    path.write_text(f'local_asr_model = "{old}"\n', encoding="utf-8")
                    self.assertEqual(load_settings(path).local_asr_model, new)

    def test_settings_reject_unsupported_asr_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid language"):
            validate(Settings(language="zh-CN"))
        validate(Settings(openrouter_asr_model="custom/transcriber"))
        with self.assertRaisesRegex(ValueError, "OpenRouter transcription model is required"):
            validate(Settings(openrouter_asr_model="  "))
        validate(Settings(local_asr_model="tiny"))
        for name in (
            "parakeet-tdt-v3",
            "parakeet-tdt-v2",
            "parakeet-flash",
            "nemotron-3.5",
            "cohere-transcribe",
        ):
            with self.subTest(name), self.assertRaisesRegex(ValueError, "Invalid local ASR model"):
                validate(Settings(local_asr_model=name))
        with self.assertRaisesRegex(ValueError, "Invalid caption ASR model"):
            validate(Settings(caption_asr_model="nemotron-3.5"))
        with self.assertRaisesRegex(ValueError, "at most 500 characters"):
            validate(Settings(vocabulary="x" * 501))
        with self.assertRaisesRegex(ValueError, "hotkeys must be different"):
            validate(Settings(hotkey="F8", markdown_hotkey="f8"))


if __name__ == "__main__":
    unittest.main()
