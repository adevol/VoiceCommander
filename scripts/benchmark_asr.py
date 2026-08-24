from __future__ import annotations

import argparse
import wave
from pathlib import Path
from statistics import median
from time import perf_counter

from voicecommander.local_asr import load_local_model
from voicecommander.settings import LOCAL_ASR_MODELS, Settings


def audio_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as recording:
            return recording.getnframes() / recording.getframerate()
    except (OSError, wave.Error, ZeroDivisionError) as error:
        raise ValueError(f"Could not read WAV duration: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VoiceCommander's real local ASR path")
    parser.add_argument("audio", type=Path, help="16 kHz mono WAV recording")
    parser.add_argument("--model", choices=LOCAL_ASR_MODELS, default="base")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be non-negative and --runs must be positive")
    duration = audio_seconds(args.audio)
    settings = Settings(local_asr_model=args.model, language=args.language)

    started = perf_counter()
    model = load_local_model(args.model)
    load_seconds = perf_counter() - started

    for _ in range(args.warmup):
        model.start(settings).finish(args.audio)

    timings = []
    transcript = ""
    for _ in range(args.runs):
        started = perf_counter()
        transcript = model.start(settings).finish(args.audio).text
        timings.append(perf_counter() - started)

    middle = median(timings)
    print(f"model:      {args.model}")
    print(f"audio:      {duration:.2f} s")
    print(f"load:       {load_seconds:.3f} s")
    print(f"runs:       {', '.join(f'{value:.3f} s' for value in timings)}")
    print(f"median:     {middle:.3f} s")
    print(f"speed:      {duration / middle:.1f}x realtime")
    print(f"transcript: {transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
