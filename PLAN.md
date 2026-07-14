# VoiceCommander Plan

## Goal

Press a configurable hotkey to start recording, press it again to stop, then paste the finished transcript into the focused text field.

## Scope

- Windows-only Python application using a virtual environment; no Docker.
- One recording and processing pipeline at a time.
- Local Nemotron or OpenRouter transcription.
- Post-processing is either `none` or `openrouter`; local LLM support is deferred.

## Pipeline

`idle -> record immediately -> transcribe after stop -> optional post-processing -> deliver`

- Audio capture never waits for model loading.
- Nemotron loads once in the background when the application starts.
- During processing, further hotkey presses are ignored.
- Record as temporary 16 kHz mono PCM16 WAV and delete it after success.
- Preserve the WAV and report its location if transcription fails.
- Always copy the result to the clipboard, then attempt to paste into the currently focused control.

## Settings

Provide a small Tkinter settings window with:

- Hotkey, language, microphone, and recording limit.
- ASR provider and model ID.
- Post-processing choice and OpenRouter model ID.
- OpenRouter API key entry.

Store non-secret settings in `config.toml`. Store the API key in Windows Credential Manager, never in the TOML file. Keep audio resolution fixed for the MVP; expose it only if testing shows a real need.

Cloud use is always explicit: remote ASR sends audio, while remote post-processing sends transcript text. Never fall back from local processing to OpenRouter automatically.

## Post-processing

The OpenRouter pass should:

- Remove false starts when the speaker corrects themself.
- Apply spoken formatting cues as paragraphs, bullets, or numbered lists.
- Preserve meaning and avoid adding new information.
- Return only the finished text.

If post-processing is disabled or fails, use the raw transcript.

## Phases

1. Add the settings window, hotkey, and immediate audio recording.
2. Load Nemotron once and transcribe after recording stops.
3. Add clipboard-first delivery and failure recovery.
4. Add optional OpenRouter post-processing.
5. Add OpenRouter transcription.
6. Package the desktop application.

## MVP Acceptance

- Recording starts immediately and no speech is lost during model loading.
- The state machine permits only one pipeline at a time.
- The result remains available on the clipboard if automatic paste fails.
- Post-processing failure falls back to the raw transcript.
- Transcription failure preserves the temporary WAV.
- Provider and model choices require no code changes.
- API keys are not stored in plaintext.

## Deferred

Local LLM post-processing, live transcription, configurable audio resolution, tray UI, installers, mobile support, and provider abstractions wait until the core flow is reliable.
