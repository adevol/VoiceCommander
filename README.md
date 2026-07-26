# VoiceCommander

Press a key, talk, press it again, and your words appear wherever the cursor
is. That is the whole idea. VoiceCommander is a Windows tool that turns speech
into text with a single hotkey, using multilingual Whisper or NVIDIA Nemotron
on your own machine, or a cloud model through OpenRouter.

## Which edition should you pick?

That depends on one question: are you comfortable sending audio to the cloud?

| Edition | Installed size | What happens to your voice |
| --- | ---: | --- |
| Standard | ~100 MiB after the first Whisper download | Local multilingual Whisper or OpenRouter |
| NVIDIA | ~5.4 GiB after model downloads | Adds local Nemotron for NVIDIA laptops |

The Standard edition downloads the pinned, verified CPU-only `whisper.cpp`
runtime and multilingual quantized Base model on first use. The NVIDIA edition
also bundles the CUDA-enabled PyTorch runtime and can download the ~2.4 GiB
Nemotron weights.

The two installers are named so you cannot mix them up —
`VoiceCommander-standard-Setup-x64.exe` and `VoiceCommander-nvidia-Setup-x64.exe`.
They are alternatives, not companions: both install into the same directory.

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
transcription defaults to multilingual Whisper. Run `uv sync --extra nvidia` to
make Nemotron appear as a second local option. OpenRouter models remain
available from the curated dropdown. Language choices are limited to the seven
locales supported by every bundled transcription model. On an NVIDIA GPU,
Nemotron runs in FP16; on a CPU, it falls back to FP32.

## Building the installers

Install Inno Setup first. Then:

```powershell
.\scripts\build.ps1
.\scripts\build.ps1 -Nvidia
```

The plain command builds the lightweight Whisper/OpenRouter edition; add
`-Nvidia` for the CUDA-enabled Nemotron option. Either way, the installers land in
`dist/installer/`.
