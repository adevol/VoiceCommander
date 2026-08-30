from __future__ import annotations

import argparse
import wave
from pathlib import Path
from statistics import median
from time import perf_counter

from voicecommander.local_asr import load_local_model
from voicecommander.settings import LOCAL_ASR_MODELS, Settings


def read_audio(path: Path) -> tuple[float, bytes]:
    try:
        with wave.open(str(path), "rb") as recording:
            if (
                recording.getframerate() != 16_000
                or recording.getnchannels() != 1
                or recording.getsampwidth() != 2
            ):
                raise ValueError("audio must be 16 kHz mono 16-bit PCM")
            return (
                recording.getnframes() / recording.getframerate(),
                recording.readframes(recording.getnframes()),
            )
    except (OSError, wave.Error) as error:
        raise ValueError(f"Could not read WAV duration: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VoiceCommander's real local ASR path")
    parser.add_argument("audio", type=Path, help="16 kHz mono WAV recording")
    parser.add_argument("--model", choices=LOCAL_ASR_MODELS, default="base")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--preview", action="store_true", help="benchmark the warm preview server")
    args = parser.parse_args()

    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be non-negative and --runs must be positive")
    duration, pcm16 = read_audio(args.audio)
    settings = Settings(local_asr_model=args.model, language=args.language)

    started = perf_counter()
    model = load_local_model(args.model)
    load_seconds = perf_counter() - started

    session = model.start(settings)

    def run_once() -> str:
        result = session.feed(pcm16) if args.preview else session.finish(args.audio)
        if result is None:
            raise RuntimeError("Preview request was dropped")
        return result

    first_preview = None
    transcript = ""
    try:
        if args.preview:
            started = perf_counter()
            transcript = run_once()
            first_preview = perf_counter() - started
        for _ in range(args.warmup):
            run_once()
        timings = []
        for _ in range(args.runs):
            started = perf_counter()
            transcript = run_once()
            timings.append(perf_counter() - started)
    finally:
        model.close()

    middle = median(timings)
    print(f"model:      {args.model}")
    print(f"mode:       {'preview' if args.preview else 'final'}")
    print(f"audio:      {duration:.2f} s")
    print(f"load:       {load_seconds:.3f} s")
    if first_preview is not None:
        print(f"first:      {first_preview:.3f} s")
    print(f"runs:       {', '.join(f'{value:.3f} s' for value in timings)}")
    print(f"median:     {middle:.3f} s")
    print(f"speed:      {duration / middle:.1f}x realtime")
    print(f"transcript: {transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
