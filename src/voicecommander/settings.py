"""Settings: the dataclass, its validation, its TOML file, and the settings window.

Attributes:
    WHISPER_MODELS: Quantised multilingual whisper.cpp builds, largest last, each
        paired with the Hugging Face LFS oid at `WHISPER_REVISION` that
        `local_asr._download` enforces.
    OPENROUTER_ASR_MODELS: Chat models that accept audio input, OpenRouter having no
        transcription endpoint.
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
WHISPER_MODELS = {
    "tiny": ("ggml-tiny-q5_1.bin", "818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7"),
    "base": ("ggml-base-q5_1.bin", "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898"),
    "small": ("ggml-small-q5_1.bin", "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb"),
}
LOCAL_ASR_MODELS = tuple(WHISPER_MODELS)
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
OPENROUTER_ASR_MODELS = (
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3.5-flash-lite",
    "openai/gpt-audio-mini",
    "xiaomi/mimo-v2.5",
)
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
    if settings.openrouter_asr_model not in OPENROUTER_ASR_MODELS:
        raise ValueError("Invalid OpenRouter transcription model")
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
    defaults = Settings()
    unknown = data.keys() - {field.name for field in fields(Settings)}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    values = {field.name: data.get(field.name, getattr(defaults, field.name)) for field in fields(Settings)}
    values["local_asr_model"] = LOCAL_MODEL_MIGRATIONS.get(
        values["local_asr_model"], values["local_asr_model"]
    )
    settings = Settings(**values)
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


def _microphones() -> list[str]:
    try:
        import sounddevice

        return [
            f"{index}: {device['name']}"
            for index, device in enumerate(sounddevice.query_devices())
            if device["max_input_channels"] > 0
        ]
    except Exception:
        return []


def show_settings(settings: Settings, parent: object | None = None) -> Settings | None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk() if parent is None else tk.Toplevel(parent)
    root.title("VoiceCommander Settings")
    root.resizable(False, False)
    values = {field.name: tk.StringVar(value=str(getattr(settings, field.name))) for field in fields(Settings)}
    values["live_preview"] = tk.BooleanVar(value=settings.live_preview)
    values["language"].set(
        next(name for name, code in LANGUAGES.items() if code == settings.language)
    )
    values["postprocess_strength"] = tk.IntVar(value=settings.postprocess_strength)
    values["api_key"] = tk.StringVar()
    result: Settings | None = None

    rows = [
        ("Normal hotkey", "hotkey", None),
        ("Markdown hotkey", "markdown_hotkey", None),
        ("Language", "language", tuple(LANGUAGES)),
        ("Custom vocabulary", "vocabulary", None),
        ("Microphone", "input_device", _microphones()),
        ("Recording limit (seconds)", "max_seconds", None),
        ("ASR provider", "asr_provider", ASR_PROVIDERS),
        ("Local transcription model", "local_asr_model", LOCAL_ASR_MODELS),
        ("Caption model", "caption_asr_model", LOCAL_ASR_MODELS),
        ("OpenRouter transcription model", "openrouter_asr_model", OPENROUTER_ASR_MODELS),
        ("Post-processing model", "postprocess_model", None),
        ("Post-processing style", "postprocess_style", tuple(POSTPROCESS_STYLES)),
    ]
    for row, (label, name, choices) in enumerate(rows):
        ttk.Label(root, text=label).grid(row=row, column=0, padx=8, pady=5, sticky="w")
        if choices is None:
            widget = ttk.Entry(root, textvariable=values[name])
        else:
            widget = ttk.Combobox(root, textvariable=values[name], values=choices)
        if name in {
            "language",
            "asr_provider",
            "local_asr_model",
            "caption_asr_model",
            "openrouter_asr_model",
            "postprocess_style",
        }:
            widget.configure(state="readonly")
        widget.grid(row=row, column=1, padx=8, pady=5, sticky="ew")

    preview_row = len(rows)
    ttk.Label(root, text="Live local preview").grid(
        row=preview_row, column=0, padx=8, pady=5, sticky="w"
    )
    ttk.Checkbutton(root, text="Show while recording", variable=values["live_preview"]).grid(
        row=preview_row, column=1, padx=8, pady=5, sticky="w"
    )

    strength_row = preview_row + 1
    ttk.Label(root, text="Editing strength").grid(
        row=strength_row, column=0, padx=8, pady=5, sticky="w"
    )
    strength = ttk.Frame(root)
    strength.grid(row=strength_row, column=1, padx=8, pady=5, sticky="ew")
    ttk.Label(strength, text="Off").pack(side="left")
    ttk.Scale(strength, from_=0, to=100, variable=values["postprocess_strength"]).pack(
        side="left", fill="x", expand=True, padx=6
    )
    ttk.Label(strength, text="Polished").pack(side="left")

    key_row = strength_row + 1
    ttk.Label(root, text="OpenRouter API key").grid(
        row=key_row, column=0, padx=8, pady=5, sticky="w"
    )
    ttk.Entry(root, textvariable=values["api_key"], show="•").grid(
        row=key_row, column=1, padx=8, pady=5, sticky="ew"
    )

    def save() -> None:
        nonlocal result
        try:
            raw = {
                field.name: str(values[field.name].get()).strip() for field in fields(Settings)
            }
            raw["language"] = LANGUAGES[raw["language"]]
            raw["max_seconds"] = int(raw["max_seconds"])
            raw["live_preview"] = bool(values["live_preview"].get())
            raw["postprocess_strength"] = int(float(raw["postprocess_strength"]))
            updated = Settings(**raw)
            entered_key = values["api_key"].get().strip()
            if updated.uses_openrouter and not (entered_key or get_api_key()):
                raise ValueError("An OpenRouter API key is required for the selected providers")
            save_settings(updated)
            save_api_key(entered_key)
        except Exception as error:
            messagebox.showerror("Invalid settings", str(error), parent=root)
            return
        result = updated
        root.destroy()

    buttons = ttk.Frame(root)
    buttons.grid(row=key_row + 1, column=0, columnspan=2, padx=8, pady=10, sticky="e")
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right", padx=(6, 0))
    ttk.Button(buttons, text="Save", command=save).pack(side="right")

    root.columnconfigure(1, weight=1)
    root.wait_window()
    return result
