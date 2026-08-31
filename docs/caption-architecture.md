# Caption ASR design

## Current decision

`CaptionSession` uses `LocalAsrEngine` directly. The project has one local ASR
implementation, so it does not have an adapter interface, provider registry,
base class, or dependency injection container for caption models.

An adapter seam would currently repeat the `preview` and `close` methods without
hiding any extra behavior. Model download, process startup, timestamp parsing,
and shutdown already belong to `LocalAsrEngine`. Rolling windows, silence
filtering, overlap reconciliation, and caption history belong to
`CaptionSession`.

## When to add an adapter

Add the seam in the same change that adds a second working local streaming
engine. Nemotron or Parakeet may justify that change once its Windows runtime,
model packaging, cancellation, and timestamped results work in this project.

The new adapter must return the existing `PreviewResult` shape so
`CaptionSession` does not branch on runtimes. Both adapters must pass the same
caption-session tests for silence, overlap replacement, duplicate suppression,
stable text, dropped work, and shutdown.

Until then, model selection chooses Whisper weights for `LocalAsrEngine`. It
does not select an implementation family.
