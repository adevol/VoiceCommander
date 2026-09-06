"""Settings data, validation, persistence, credentials, and model metadata.

Attributes:
    WHISPER_DEFINITIONS: Quantised multilingual whisper.cpp builds, largest last,
        with their download hashes and preview capability.
    VOCABULARY_LIMIT: Characters accepted in the custom vocabulary.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

APP_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VoiceCommander"
CONFIG_PATH = APP_DIR / "config.toml"
KEYRING_SERVICE = "VoiceCommander"
KEYRING_USER = "OpenRouter"
ASR_PROVIDERS = ("local", "openrouter")
WHISPER_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"


@dataclass(frozen=True, slots=True)
class WhisperModel:
    filename: str
    sha256: str
    preview: bool = True


WHISPER_DEFINITIONS = {
    "tiny": WhisperModel(
        "ggml-tiny-q5_1.bin",
        "818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7",
    ),
    "base": WhisperModel(
        "ggml-base-q5_1.bin",
        "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898",
    ),
    "small": WhisperModel(
        "ggml-small-q5_1.bin",
        "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
    ),
}
LOCAL_ASR_MODELS = tuple(WHISPER_DEFINITIONS)
LOCAL_MODEL_MIGRATIONS = {
    "large-v3-turbo": "small",
    "parakeet-tdt-v3": "base",
    "parakeet-tdt-v2": "base",
    "parakeet-flash": "base",
    "nemotron-3.5": "base",
    "cohere-transcribe": "base",
}
LANGUAGES = {
    "Automatic": "auto",
    "English": "en-US",
    "French": "fr-FR",
    "German": "de-DE",
    "Italian": "it-IT",
    "Portuguese": "pt-PT",
    "Russian": "ru-RU",
    "Spanish": "es-ES",
}
VOCABULARY_LIMIT = 500
POSTPROCESS_STYLES = {
    "faithful": "Keep the speaker's wording and sentence order wherever possible.",
    "clean": "Remove false starts and repetition, and improve readability.",
    "professional": "Use clear, polished professional prose.",
    "concise": "Tighten the wording and remove unnecessary repetition.",
}


@dataclass(frozen=True, slots=True)
class Settings:
    hotkey: str = "f8"
    markdown_hotkey: str = "f7"
    language: str = "auto"
    vocabulary: str = ""
    input_device: str = ""
    max_seconds: int = 300
    asr_provider: str = "local"
    local_asr_model: str = "base"
    caption_asr_model: str = "base"
    live_preview: bool = True
    openrouter_asr_model: str = "google/gemini-2.5-flash"
    postprocess_model: str = "openai/gpt-4o-mini"
    postprocess_style: str = "clean"
    postprocess_strength: int = 0

    @property
    def uses_openrouter(self) -> bool:
        return self.asr_provider == "openrouter" or self.postprocess_strength > 0


def validate(settings: Settings) -> None:
    defaults = Settings()
    for field in fields(Settings):
        if type(getattr(settings, field.name)) is not type(getattr(defaults, field.name)):
            raise ValueError(f"Setting '{field.name}' has an invalid type")
    if not settings.hotkey.strip():
        raise ValueError("Hotkey is required")
    if not settings.markdown_hotkey.strip():
        raise ValueError("Markdown hotkey is required")
    if settings.hotkey.casefold() == settings.markdown_hotkey.casefold():
        raise ValueError("Recording and Markdown hotkeys must be different")
    for name, hotkey in (("recording", settings.hotkey), ("Markdown", settings.markdown_hotkey)):
        if "ctrl" in hotkey.lower() or "control" in hotkey.lower():
            raise ValueError(
                f"The {name} hotkey cannot use Ctrl because Ctrl+recording hotkey opens the menu"
            )
    if not 1 <= settings.max_seconds <= 3600:
        raise ValueError("Recording limit must be between 1 and 3600 seconds")
    if settings.language not in LANGUAGES.values():
        raise ValueError("Invalid language")
    if len(settings.vocabulary) > VOCABULARY_LIMIT:
        raise ValueError(f"Custom vocabulary must be at most {VOCABULARY_LIMIT} characters")
    if settings.asr_provider not in ASR_PROVIDERS:
        raise ValueError("Invalid ASR provider")
    if settings.local_asr_model not in LOCAL_ASR_MODELS:
        raise ValueError("Invalid local ASR model")
    if settings.caption_asr_model not in LOCAL_ASR_MODELS:
        raise ValueError("Invalid caption ASR model")
    if not settings.openrouter_asr_model.strip():
        raise ValueError("OpenRouter transcription model is required")
    if settings.postprocess_strength > 0 and not settings.postprocess_model.strip():
        raise ValueError("Post-processing model is required")
    if settings.postprocess_style not in POSTPROCESS_STYLES:
        raise ValueError("Invalid post-processing style")
    if not 0 <= settings.postprocess_strength <= 100:
        raise ValueError("Post-processing strength must be between 0 and 100")


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    if not path.exists():
        return Settings()
    with path.open("rb") as file:
        data = tomllib.load(file)
    unknown = data.keys() - {field.name for field in fields(Settings)}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    if "local_asr_model" in data:
        data["local_asr_model"] = LOCAL_MODEL_MIGRATIONS.get(
            data["local_asr_model"], data["local_asr_model"]
        )
    settings = Settings(**data)
    validate(settings)
    return settings


def save_settings(settings: Settings, path: Path = CONFIG_PATH) -> None:
    validate(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        f"{field.name} = {json.dumps(getattr(settings, field.name), ensure_ascii=False)}"
        for field in fields(Settings)
    ) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    import keyring

    return (keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or "").strip()


def save_api_key(api_key: str) -> None:
    if not api_key.strip():
        return
    import keyring

    keyring.set_password(KEYRING_SERVICE, KEYRING_USER, api_key.strip())
