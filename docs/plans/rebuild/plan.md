# Rebuild plan

## Scope and authority

Rebuild VoiceCommander from user requirements without inheriting the old
architecture. On 2026-09-26 the user explicitly chose to remove the old
implementation and all inherited supporting files. This supersedes the earlier
plan to keep the old app usable during the rebuild. Git history at `5b9e169`
preserves the previous application.

Keep the new rebuild records and the local R2T2 experiment. Remove old application
source, tests, CI, packaging, scripts, dependency definitions, benchmark files,
local configuration, the old environment, and the unused legacy checkout.
Replace the README and ignore rules with files describing the current state.
Do not add application scaffolding until it serves a working step.

This authorizes local cleanup. It does not authorize paid reviews, publishing,
remote configuration changes, releases, or changes to app data outside this repo.
The OpenRouter review experiment was retired by user decision. Select the rebuild's
review setup later. Any future review/repair loop is limited to two repair passes
unless the user sets another limit. Ask only one question at a time.

## Requirements

- Run directly on Windows without WSL or Docker.
- Work on laptops without a dedicated GPU. GPU acceleration may be optional.
- Support English and Mandarin Chinese at minimum. Expose the selected engine's
  full supported language set without an artificial allowlist. Distinguish
  upstream language claims from accuracy measured here.

Architecture, implementation language, and transcription engine remain open.
The experiment uses CPU execution to check feasibility; it does not prohibit
later use of integrated graphics.

## Next working step

Keep the next step small: a saved recording produces streaming console text.
The R2T2 native Windows CPU experiment does this today. See
[results and limitations](status.md#native-cpu-experiment-results).
No application classes or provider framework are needed for this step.

Before selecting an engine, measure first text, completion delay, peak memory,
and transcript errors on representative English and Mandarin recordings. Include
mixed languages, custom vocabulary, quiet speech, pauses, silence, long recordings,
and final words at stop. Verify whether committed text changes or loses words.
The old Whisper measurements are historical comparison data; its implementation
is no longer part of this checkout.

Add microphone input, then a hotkey and paste after the experiment is satisfactory.
For those steps, cancellation and runtime failures must preserve recoverable audio
and report failure without pasting incomplete text as a successful result.

## R2T2 evaluation candidate

[R2T2](https://github.com/netease-youdao/Confucius4-R2T2) is a candidate, not a
selected backend. Its [native runtime documentation](https://github.com/netease-youdao/Confucius4-R2T2/blob/master/r2t2_llama/README.md)
describes the build path used by the local experiment. A Windows CPU build works
with one portability fix. The source pins, measurements, and limitations are in
the status record. Code and weights have
[separate licenses](https://github.com/netease-youdao/Confucius4-R2T2#license).

## Cleanup acceptance

- The tracked tree contains only the new README, ignore rules, and rebuild records.
- No inherited app, tests, dependencies, packaging, CI, or local settings remain
  in the main checkout. The unused nested checkout is removed through Git after
  verifying it has no uncommitted or ignored files.
- Git history and the R2T2 experiment remain intact. Run a short transcription
  after cleanup to verify the experiment does not need the old app environment.
- Check documentation links and the diff. Historical app test results are not
  validation of the rebuild. Independent review remains unconfigured.
