from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, fields
from importlib.util import find_spec
from pathlib import Path

APP_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VoiceCommander"
CONFIG_PATH = APP_DIR / "config.toml"
KEYRING_SERVICE = "VoiceCommander"
KEYRING_USER = "OpenRouter"
ASR_PROVIDERS = ("local", "openrouter")
LOCAL_ASR_MODELS = ("whisper", "nemotron")
OPENROUTER_ASR_MODELS = (
    "deepgram/nova-3",
    "google/chirp-3",
    "microsoft/mai-transcribe-1.5",
    "mistralai/voxtral-mini-transcribe",
    "nvidia/parakeet-tdt-0.6b-v3",
    "openai/gpt-4o-mini-transcribe",
    "openai/gpt-4o-transcribe",
    "openai/whisper-1",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "qwen/qwen3-asr-flash-2026-02-10",
)
POSTPROCESS_STYLES = {
    "faithful": "Keep the speaker's wording and sentence order wherever possible.",
    "clean": "Remove false starts and repetition, and improve readability.",
    "professional": "Use clear, polished professional prose.",
    "concise": "Tighten the wording and remove unnecessary repetition.",
}


@dataclass(frozen=True, slots=True)
class Settings:
    hotkey: str = "f8"
    language: str = "en-US"
    input_device: str = ""
    max_seconds: int = 300
    asr_provider: str = "local"
    local_asr_model: str = "whisper"
    openrouter_asr_model: str = "openai/whisper-large-v3"
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
    if "ctrl" in settings.hotkey.lower() or "control" in settings.hotkey.lower():
        raise ValueError("The recording hotkey cannot use Ctrl because Ctrl+hotkey opens the menu")
    if not 1 <= settings.max_seconds <= 3600:
        raise ValueError("Recording limit must be between 1 and 3600 seconds")
    if settings.asr_provider not in ASR_PROVIDERS:
        raise ValueError("Invalid ASR provider")
    if settings.local_asr_model not in LOCAL_ASR_MODELS:
        raise ValueError("Invalid local ASR model")
    if settings.asr_provider == "openrouter" and not settings.openrouter_asr_model.strip():
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
    data.pop("asr_model", None)
    if data.get("asr_provider") == "local" and "local_asr_model" not in data:
        data["local_asr_model"] = "nemotron"
    # Removed setting: post-processing is now enabled purely by strength > 0.
    if data.pop("postprocess_provider", None) == "none":
        data["postprocess_strength"] = 0
    defaults = Settings()
    unknown = data.keys() - {field.name for field in fields(Settings)}
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    values = {field.name: data.get(field.name, getattr(defaults, field.name)) for field in fields(Settings)}
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


def nemotron_runtime_available() -> bool:
    try:
        return all(find_spec(package) is not None for package in ("librosa", "torch", "transformers"))
    except ImportError:
        return False


def local_runtime_available(local_asr_model: str = "whisper") -> bool:
    return local_asr_model == "whisper" or (
        local_asr_model == "nemotron" and nemotron_runtime_available()
    )


def _microphones() -> list[str]:
    try:
        import sounddevice

        return [
            # Recorder.start parses this index prefix.
            f"{index}: {device['name']}"
            for index, device in enumerate(sounddevice.query_devices())
            if device["max_input_channels"] > 0
        ]
    except Exception:
        return []


def show_settings(settings: Settings) -> Settings | None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("VoiceCommander Settings")
    root.resizable(False, False)
    available_local_models = LOCAL_ASR_MODELS if nemotron_runtime_available() else ("whisper",)
    values = {field.name: tk.StringVar(value=str(getattr(settings, field.name))) for field in fields(Settings)}
    values["postprocess_strength"] = tk.IntVar(value=settings.postprocess_strength)
    values["api_key"] = tk.StringVar()
    if settings.local_asr_model not in available_local_models:
        values["local_asr_model"].set("whisper")
    result: Settings | None = None

    rows = [
        ("Hotkey", "hotkey", None),
        ("Language", "language", None),
        ("Microphone", "input_device", _microphones()),
        ("Recording limit (seconds)", "max_seconds", None),
        ("ASR provider", "asr_provider", ASR_PROVIDERS),
        ("Local transcription model", "local_asr_model", available_local_models),
        ("OpenRouter transcription model", "openrouter_asr_model", OPENROUTER_ASR_MODELS),
        ("Post-processing model", "postprocess_model", None),
        ("Post-processing style", "postprocess_style", tuple(POSTPROCESS_STYLES)),
    ]
    for row, (label, name, choices) in enumerate(rows):
        ttk.Label(root, text=label).grid(row=row, column=0, padx=8, pady=5, sticky="w")
        widget = ttk.Combobox(root, textvariable=values[name], values=choices) if choices is not None else ttk.Entry(root, textvariable=values[name])
        if name in {"asr_provider", "local_asr_model", "postprocess_style"}:
            widget.configure(state="readonly")
        widget.grid(row=row, column=1, padx=8, pady=5, sticky="ew")

    strength_row = len(rows)
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
            raw["max_seconds"] = int(raw["max_seconds"])
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
    root.mainloop()
    return result
