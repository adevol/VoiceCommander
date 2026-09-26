# VoiceCommander

VoiceCommander is being rebuilt from scratch. The previous application and its
supporting files have been removed. Commit `5b9e169` retains the old implementation.
There is no runnable replacement application yet.

The replacement must run directly on Windows, work without a dedicated GPU,
and support the chosen engine's full language set, including English and Mandarin.
See the [rebuild plan](docs/plans/rebuild/plan.md) and
[experiment results](docs/plans/rebuild/status.md).

The local R2T2 experiment remains in ignored `build/r2t2/`, with its own Python
environment, native build, models, and measurements. It is not included in a fresh
clone and is not yet the selected transcription engine.

On the machine where the experiment was prepared, run its Mandarin sample:

```powershell
build/r2t2/.venv/Scripts/python.exe -X utf8 -u build/r2t2/run.py build/r2t2/upstream/resources/test.wav --language Chinese --chunk-ms 1000 --context-tokens 4096 --output build/r2t2/chinese-rebuild-check.json
```

Microphone input, hotkeys, and pasting belong to later steps, after the
transcription experiment meets the agreed requirements.
