# VoiceCommander

Press a key, talk, then press it again. VoiceCommander types the result wherever
your cursor is. It is a Windows dictation tool with local multilingual Whisper
transcription and optional OpenRouter transcription and text editing.

The project is preparing for an incremental rebuild. The
[rebuild work order](docs/plans/rebuild/plan.md) records its scope. The current
implementation remains usable while the replacement's requirements are decided.
Its architecture is not a constraint on the rebuild.

## Use it

| Hotkey | Action |
| --- | --- |
| `F8` | Start or stop normal dictation |
| `F7` | Start or stop Markdown dictation |
| `Ctrl+F8` | Open settings while idle |

If you change the normal dictation hotkey, use `Ctrl` plus that key for settings.
Installed builds start when you sign in.

Recording stops automatically after five minutes by default. Settings allow
limits from 1 to 3,600 seconds, microphone selection, automatic or fixed language
detection, and custom vocabulary as comma-separated terms.

Local models download when selected. `base` is the default, `tiny` favors speed,
and `small` favors accuracy. Downloads of models and the native runtime are pinned
and verified. Live local preview shows tentative text in a four-line window;
disable it in settings if unwanted. Preview truncation does not shorten the final
transcript.

At editing strength 0, normal dictation pastes the raw transcript. Higher values
send the final text to OpenRouter for editing. If editing fails, normal dictation
falls back to the raw transcript.

Markdown mode always requires an OpenRouter API key. Dictate headings, lists,
tables, code blocks, and formulas. Inline formulas use `$...$`; display formulas
use `$$...$$`. If formatting fails, the app reports an error without pasting the
spoken instructions.

The separate **VoiceCommander Captions** Start menu shortcut captions audio
playing on your computer in an always-on-top window. Captions stay local, have
their own model setting, and follow the dictation language setting.

## Privacy and recovery

Transcription is local by default. Selecting cloud transcription sends the
recording to the OpenRouter audio model configured in settings. Optional editing
and Markdown formatting send the final transcript. Cloud requests set
`data_collection` to `deny` to restrict provider routing; OpenRouter still
processes requests and records metadata under its own policy.

API keys are stored in Windows Credential Manager. During development,
`OPENROUTER_API_KEY` takes precedence.

| Files | Location |
| --- | --- |
| Settings | `%APPDATA%\VoiceCommander\config.toml` |
| Models and runtime | `%APPDATA%\VoiceCommander\whisper.cpp\` |
| Log | `%LOCALAPPDATA%\VoiceCommander\voicecommander.log` |
| Recordings | `%TEMP%\VoiceCommander\recording-*.wav` |

If transcription or delivery fails after saving the recording, the error shows
the WAV location. Copy it somewhere permanent for recovery; the app has no saved
recording retry command. After sending a paste command, the app attempts to
delete the WAV. It cannot confirm that the target application accepted the text.
The transcript remains on the clipboard.

For "No speech detected", check microphone mute and the selected input device.
A saved microphone is matched by name if Windows changes its device number.
If it is unavailable, reconnect it or select another microphone.

## Develop and build

Python 3.13 and `uv` are required.

```powershell
uv sync --locked --dev
uv run voicecommander --settings
uv run voicecommander
uv run voicecommander --captions
```

Run the same checks as Windows CI:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests
```

To measure local transcription on a 16 kHz mono, 16-bit PCM WAV:

```powershell
uv run python scripts/benchmark_asr.py benchmarks/jfk.wav --model base
```

The bundled [JFK sample](benchmarks/jfk.wav) comes from
[whisper.cpp v1.9.1](https://github.com/ggml-org/whisper.cpp/blob/v1.9.1/samples/jfk.wav).
Short English benchmarks do not establish accuracy or performance for long
recordings or other languages. Those require separate runtime checks.

Install Inno Setup 6, then run `.\scripts\build.ps1`. The build checks lint,
formatting, and tests before creating the installer in `dist/installer/`.
