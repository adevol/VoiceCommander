# Rebuild work order

## Outcome and scope

Rebuild VoiceCommander incrementally from user requirements, without treating
the current architecture as a design constraint. Start by removing unnecessary
documentation and files. Keep the existing app usable during this first step.

The user authorized local cleanup on 2026-09-21 and explicitly chose to retire
the experimental OpenRouter review system. Choose a replacement review setup
later. No paid reviews, publishing, remote configuration changes, or releases
are authorized by this work order.

## Working slices

1. Clean the repository. Remove the experimental reviewer and its associated
   tests, workflow, policy, skills, and setup notes. Remove generated analysis,
   build output, and caches. Consolidate useful user and development information
   in the README. Preserve application code, application tests, dependency locks,
   build inputs, benchmark audio and attribution, local secrets, the environment,
   and other worktrees.
2. Establish the replacement's intended behavior with the user. Decide which
   current capabilities belong in the first usable version, its privacy and
   failure guarantees, and observable acceptance examples. The README describes
   the existing app; it is not an approved replacement specification.
3. Propose the smallest design that meets those requirements and build one
   end-to-end slice around the largest uncertainty. Select that experiment and
   its success criteria after slice 2. Expand only after observing its behavior
   and reviewing whether its structure is necessary.

## Cleanup acceptance

- Existing application source and tests, CI, packaging inputs, and lockfile do
  not change. Existing lint, formatting, and application tests pass.
- No remaining code or documentation requires the removed reviewer or docs.
- Secrets, the working environment, and the separate Kilo worktree survive.
- Generated analysis and build output no longer clutter the checkout.

Architecture, implementation language, transcription engine, migration strategy,
and independent reviewer are open decisions. Do not add speculative replacement
modules now. Future review/repair loops have at most two repair passes unless
the user sets another limit; paid services need separate authority.
