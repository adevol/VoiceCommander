"""The VoiceCommander settings window and microphone choices."""

from __future__ import annotations

from dataclasses import fields

from .settings import (
    ASR_PROVIDERS,
    LANGUAGES,
    LOCAL_ASR_MODELS,
    POSTPROCESS_STYLES,
    Settings,
    get_api_key,
    save_api_key,
    save_settings,
)


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
        ("OpenRouter transcription model", "openrouter_asr_model", None),
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
