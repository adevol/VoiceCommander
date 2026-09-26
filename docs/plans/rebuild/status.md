# Rebuild status

## Current checkout

On 2026-09-26 the user chose a clean start and explicitly authorized removing
all inherited files. The earlier decision to preserve the working app is
superseded. The old implementation remains in Git history at `5b9e169`.

Removed application source and tests, CI, installer and build scripts, dependency
files, benchmark audio, local `.env`, the old `.venv`, and empty local skill
directories. The nested Kilo checkout had no modified, untracked, or ignored
files and was removed using `git worktree remove`. No legacy app or Python
processes were running at the time of removal. App data outside the repo was
outside the cleanup scope.

The tracked checkout now contains a rewritten README and ignore rules plus the
rebuild plan and this status record. The separate R2T2 experiment remains under
ignored `build/r2t2/`, including its own environment, source builds, model files,
and measurements. A fresh clone does not contain that experiment.

The user authorized committing this cleanup on `master`, based on `5b9e169`.
No push is included. There is no replacement
application, CI, package definition, or test suite yet. No independent review
covers this cleanup; the review setup remains undecided.

## Clean-start verification

After deleting the legacy environment, the Mandarin sample completed using only
`build/r2t2/.venv` and its local assets, with 4,096 context tokens on CPU. Its
transcript matched the previous Mandarin result exactly. Evidence is in
`build/r2t2/chinese-rebuild-check.json` and the corresponding log. This checks
experiment independence, not general transcription accuracy.

The first redirected run encountered Windows' default CP1252 output encoding,
which cannot print Chinese. The documented commands now use Python's `-X utf8`;
the rerun exited successfully without changing the experiment code.

Verified that only four tracked files remain present, the top-level directory
contains only Git metadata, the rebuild docs, README, ignore rules, and local
experiment files, and all local Markdown links resolve. Git whitespace checks
passed, Git lists only the main checkout, and the old app can be read at `5b9e169`.
No old application tests were run after their removal.

## Historical cleanup

The first cleanup retired the failed OpenRouter review system and removed
obsolete documentation and generated files. It was committed as `5b9e169`.
At that point, the old app passed Ruff lint and formatting plus 67 unit tests
with one skipped. Those checks describe the retired app, not this rebuild.

The following experiment results were measured before the clean start. Whisper
results remain historical comparisons; its runtime and benchmark script have
now been removed from the checkout. R2T2 is still an evaluation candidate.

## Native CPU experiment results

The user authorized installation of the build prerequisites on 2026-09-25.
Installed Build Tools 2026 18.10.2, MSVC 14.51.36231, Windows SDK 10.0.26100,
and CMake 4.3.1-msvc1. Built and ran R2T2 on Windows with Python 3.13.12,
CUDA/Vulkan disabled, `use_gpu=False`, and `n_gpu_layers=0`.

Hardware: Ryzen AI 7 350, 8 physical cores, 16 logical CPUs, 31.3 GiB RAM.
Inference used 8 threads, Q4_K_M model weights and the Q8_0 audio projector.
Both GGUF files passed verification against the published SHA-256 hashes.

| Recording / update interval | Audio length | First preview text | Completion after audio ends | Peak process memory |
| --- | ---: | ---: | ---: | ---: |
| English JFK / 160 ms | 11.00 s | 3.20 s | 55.36 s | 6.32 GiB |
| English JFK / 1,000 ms | 11.00 s | 1.87 s | 3.81 s | 6.32 GiB |
| Upstream Mandarin / 1,000 ms | 6.74 s | 2.86 s | 1.52 s | 6.30 GiB |

These initial runs used upstream's 32,768-token context default. One run per
R2T2 setting, with audio paced as if recorded live. Times exclude
Python imports and model loading; model loading itself took 3.33-5.71 seconds.
These short samples do not establish performance for longer recordings.

Both English results matched the expected JFK words when punctuation and case
were ignored. Mandarin produced `之前有顾客自己带酒水也没加收钱或者不让喝。`.
There is no independently labeled Mandarin reference, so its error rate is
unscored. The model declares 30 languages; broader accuracy remains untested.

The 160 ms run twice returned nonempty preview text that shortened an earlier
prefix, dropping `not` after `ask`. Empty updates on an incomplete final chunk
are a separate no-update case. Do not assume this native path always produces
append-only preview text. The one-second English run had no nonempty prefix
retractions.

On the same recordings, the existing Whisper base path took a median 0.974 s
for English final transcription and 0.912 s for Mandarin, over three runs each.
Its warm eight-second English preview took 1.011 s. These are different
operations from R2T2's continuous streaming, not identical latency measures.

All experiment files are temporary under ignored `build/r2t2/`, using an
independent environment that survives removal of the old application. `run.py` reuses upstream's streaming
driver and token schedule, redirecting token limits through the native adapter
to avoid the optional vLLM import. `compile.py` records the build command and
passes a normalized environment to avoid duplicate Windows PATH entries.

Reproduce the original Mandarin setting from the repo root on the prepared machine:

```powershell
build/r2t2/.venv/Scripts/python.exe build/r2t2/compile.py
build/r2t2/.venv/Scripts/python.exe -X utf8 -u build/r2t2/run.py build/r2t2/upstream/resources/test.wav --language Chinese --chunk-ms 1000 --context-tokens 32768 --output build/r2t2/chinese-1000ms.json
```

The historical English runs used `benchmarks/jfk.wav`, removed with the old app.
That fixture remains available in commit `5b9e169`. It originated from
[whisper.cpp v1.9.1](https://github.com/ggml-org/whisper.cpp/blob/v1.9.1/samples/jfk.wav).
To test another recording, pass a mono 16 kHz WAV to `run.py --language English`.

The Windows source fix is `static_cast<ssize_t>` to `static_cast<py::ssize_t>`
in `r2t2_llama/native_ext.cpp`. The patch, environment package versions, model
checksums, build log, transcription logs, and result JSON are saved in the
experiment folder. This is a local prototype, not a packaged installation.

Source pins: R2T2 `26d55a54ce5670cff9947a167d8ed95d569fd4d9`, llama.cpp
`ad6c66839af3c5646fba8c6c2e2087a1e4e38948`, GGUF model
`86ff0251cb9f456b63aeef5f80137f104e22869a`, processor
`185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`. No independent review has run.

### Memory diagnosis

The initial 6.3 GiB figure reflected an oversized experiment context, not a
demonstrated minimum memory requirement. The native log reports a 3,584 MiB
FP16 key/value cache for 32,768 tokens. Other reported allocations include
1,043.68 MiB of mapped model weights, 645.75 MiB of CPU-repacked weights,
a 332.18 MiB audio model, and about 342 MiB of compute buffers. Those allocation
figures are not an exact additive breakdown of process residency; Python and
its imported libraries also contribute to the measured peak working set.

Changed only `n_ctx` to 4,096 and reran the same English one-second-chunk test.
The cache dropped to 448 MiB, and peak process memory fell from 6,468.44 to
3,332.36 MiB, a 48.5% reduction. The final transcript was identical. First text
arrived at 1.73 s, and completion was 3.05 s after audio ended. This single run
does not establish a speed improvement.

The experiment now defaults to 4,096 context tokens for short recordings.
Use `--context-tokens 32768` to reproduce the original settings, or `4096`
with output `build/r2t2/english-context4096.json` for the smaller-context test.
The result and allocation log are retained under `build/r2t2/`. Long recordings
have not been validated with the smaller context; choosing a production limit
or audio segmentation policy remains separate work.
