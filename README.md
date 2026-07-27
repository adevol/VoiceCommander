# VoiceCommander

Press a key, talk, press it again, and your words appear wherever the cursor
is. That is the whole idea. VoiceCommander is a Windows tool that turns speech
into text with a single hotkey, using multilingual Whisper on your own machine,
or a cloud model through OpenRouter.

## Where your voice goes

That is the one question worth answering up front, and you answer it in
settings. Local transcription is the default and nothing leaves your machine:
the pinned, verified CPU-only `whisper.cpp` runtime and multilingual quantized
Base model download on first use, about 100 MiB in total. Choosing an
OpenRouter model instead sends your audio to that provider.

## How you use it

Press your configured hotkey to start recording, press it again to stop, and
the transcript is typed into whatever window you were working in. Hold Ctrl
with that same hotkey (Ctrl+F8 by default) to open the full settings window.
Saved changes, including the hotkey, microphone, and speech model, take effect
immediately.

The editing-strength slider controls post-processing: at 0 it is off and your
words come back exactly as you spoke them; anything above 0 sends the
transcript text (never the audio) to the model you picked on OpenRouter, so it
needs an API key even with local transcription. With post-processing on you
can speak edits — "scratch that", "correction", "new paragraph", "bullet
point", "quote ... unquote" — and they are applied instead of transcribed.

Installed builds start quietly whenever you sign in, so the hotkey is simply
always there.

## Live captions

Installed builds include a **VoiceCommander Captions** Start menu shortcut.
During development, run:

```powershell
uv run voicecommander --captions
```

This opens an always-on-top window that captions whatever your computer is
playing — YouTube included — using the local Whisper model, so no audio
leaves the machine. Captions appear roughly one chunk (about eight seconds)
behind live playback and use the language from settings.

## Developing

You will need Python 3.13. From there, four commands cover the daily routine:

```powershell
uv sync
uv run voicecommander --settings
uv run voicecommander
uv run python -m unittest discover -s tests
```

A few things worth knowing before your first run. Local transcription downloads
its model weights from Hugging Face the first time it starts. Any OpenRouter key
you enter in settings goes into Windows Credential Manager, not into a config
file; during development you can also set the `OPENROUTER_API_KEY` environment
variable, which takes precedence. (`.env` files are not read.) Local
transcription uses multilingual Whisper; OpenRouter models are available from
the curated dropdown. Language choices are limited to seven locales.

## Building the installer

Install Inno Setup first. Then:

```powershell
.\scripts\build.ps1
```

The installer lands in `dist/installer/`.
