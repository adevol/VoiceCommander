from __future__ import annotations

import argparse
import wave
from pathlib import Path
from statistics import median
from time import perf_counter

from voicecommander.local_asr import (
    MIN_REUSABLE_PREVIEW_SECONDS,
    PreviewText,
    load_local_model,
)
from voicecommander.preview import PREVIEW_WINDOW_BYTES
from voicecommander.settings import LOCAL_ASR_MODELS, Settings

BYTES_PER_SECOND = 16_000 * 2
FINAL_TAIL_SECONDS = 4


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
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")
    duration, pcm16 = read_audio(args.audio)
    settings = Settings(local_asr_model=args.model, language=args.language)

    started = perf_counter()
    model = load_local_model(args.model)
    load_seconds = perf_counter() - started

    preview_audio = pcm16[-PREVIEW_WINDOW_BYTES:]
    preview_timings = []
    final_timings = []
    preview_final = None
    try:
        started = perf_counter()
        preview = model.preview(preview_audio, settings)
        first_preview = perf_counter() - started
        if preview is None:
            raise RuntimeError("Preview request was dropped")
        preview_text = preview.text
        for _ in range(args.runs):
            started = perf_counter()
            model.preview(preview_audio, settings)
            preview_timings.append(perf_counter() - started)
        for _ in range(args.runs):
            started = perf_counter()
            transcript = model.transcribe(args.audio, settings)
            final_timings.append(perf_counter() - started)
        if duration >= MIN_REUSABLE_PREVIEW_SECONDS + FINAL_TAIL_SECONDS:
            prefix = model.preview(
                pcm16[: -FINAL_TAIL_SECONDS * BYTES_PER_SECOND], settings
            )
            if (
                prefix is not None
                and prefix.segments
                and prefix.segments[-1].end >= MIN_REUSABLE_PREVIEW_SECONDS
            ):
                started = perf_counter()
                preview_final = model.transcribe(
                    args.audio,
                    settings,
                    PreviewText(prefix.text, "", prefix.segments[-1].end),
                )
                preview_final = perf_counter() - started, preview_final == transcript
    finally:
        model.close()

    print(f"model:      {args.model}")
    print(f"audio:      {duration:.2f} s")
    print(f"load:       {load_seconds:.3f} s")
    print(f"preview:    {first_preview:.3f} s first, {median(preview_timings):.3f} s warm")
    print(f"full final: {median(final_timings):.3f} s")
    if preview_final is not None:
        print(
            f"warm final: {preview_final[0]:.3f} s, "
            f"{'same' if preview_final[1] else 'different'} text"
        )
    else:
        print("warm final: skipped; recording needs a 22 s committed prefix")
    print(f"preview text: {preview_text}")
    print(f"transcript: {transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
