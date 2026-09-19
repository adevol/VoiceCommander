# Independent audit before merge

This repository's default branch is `master`. The audit workflow also recognizes
`main` if the branch is renamed later. No branch rename is required.

The proposed gate uses `z-ai/glm-5.3-flash` through OpenRouter with low reasoning
effort. High effort did not produce a complete review with either a 12,000- or
24,000-token response allowance in the initial live trials, so model settings
must be calibrated before activation. A third trial at low effort also returned
an unusable review response. None of the three trials completed a usable audit.
This configuration is experimental and must pass a live clean-candidate and
seeded-defect trial before it is used for unattended merges.

It audits all tracked text files supported by the policy and uses previous
versions of changed files to check for regressions. Candidate files are read
through GitHub's API. They are never
checked out, imported or executed by the privileged reviewer. The existing Windows
CI job still runs lint, formatting and application tests separately.

## Current state

The workflow, runner, tests and repository skills are prepared locally. They are
not an active merge restriction until the GitHub App, environment and required
checks below are configured. The connection used during preparation had read-only
access, so no GitHub protection or secrets were changed.

Preparation notes record seven P2 and two P3 findings from an independent Sol
audit of a working-tree snapshot, despite the existing suite passing. That
snapshot included four pre-existing uncommitted files. The report is not included
in this repository, so these notes cannot establish approval of any commit.
Obtain a fresh review of the exact candidate revision before merging.

The proposed gate blocks on critical, high and medium findings. Because it
audits the whole codebase, an existing defect can block an unrelated PR.

## One-time activation

1. Bootstrap these files into the protected default branch through a repository
   administrator's reviewed setup change. Later PRs cannot use this automatic
   lane to change `.github/`, `.agents/`, the audit script or its tests. Those need
   a separately authorized policy update, never approval by their own new policy.
2. Create a dedicated GitHub App for this repository. Grant repository contents
   read, pull requests read and checks write. Install it only where needed. It
   must not be the author's App and needs no permission to push or merge.
3. Create the GitHub Actions environment `factory-review`. Select only the
   protected default branch as an allowed deployment branch, with no allowed
   tags. Do not use an unrestricted environment. No recurring human approval is
   needed once this restriction is correctly configured.
4. In that environment, create a variable named `FACTORY_REVIEW_APP_ID` for the
   App ID. Create two environment secrets: `FACTORY_REVIEW_APP_PRIVATE_KEY` for
   the private key and `OPENROUTER_API_KEY` for a dedicated OpenRouter key with
   a spending limit. The workflow reads the App ID from `vars` and the keys from
   `secrets`. Keep both keys out of repository-level secrets, source files and
   PR text. An account or daily spending limit is separate from the script's
   per-request estimate and token and price limits.
5. Run the workflow on a disposable internal PR so the App publishes a
   `factory/review` check. Inspect the independent report in the check details.
6. Inspect existing branch rules before changing them. Preserve other required
   checks and approvals. Require pull requests and require `factory/review` from
   this specific App. Also require the existing CI job named `checks` from GitHub
   Actions once that job has run. Add the new requirement to existing protection;
   do not replace existing rules with a minimal example.
7. Require branches to be up to date before merging. Enforce the requirements for
   administrators and bots too, with no author bypass, force pushes or direct
   pushes to the protected branch. Strict freshness is essential: GitHub attaches
   a check to the PR head, while the runner also records the base commit. A queued
   re-audit alone cannot close the stale-base window.

This version does not implement merge queues. Do not enable one until the
`merge_group` event and group-commit review semantics are implemented and tested.
It does not merge or deploy anything automatically.

## What the runner enforces

Each relevant event reconciles all open PRs. One global concurrency group prevents
competing verdicts, and a twice-hourly sweep recovers events GitHub coalesces. Only
PRs by owners, members and collaborators enter the paid internal audit lane.
External contributions remain blocked until separately handled. A bot author must
also meet this policy before an unattended authoring loop can use the lane.

The runner records head and base SHAs, prompt and runner digests, and the model.
It checks for PR changes before publishing. A new revision needs a new audit.
Completed substantive failed reviews cannot be rerolled by editing the title or
description. A requirements change after such a failure needs a new candidate
commit. Infrastructure failures can retry; unchanged findings cannot be sampled
repeatedly until a reviewer approves them.

Malformed reports, duplicate JSON keys, unfinished responses, incomplete file
coverage, unknown file types, truncated trees and exceeded budgets never pass.
The model gets no tools or secrets. Source `.env` files, symlinks and submodules
block instead of being silently sent. The existing `benchmarks/jfk.wav` is declared
as an excluded audio fixture, and PRs changing it need separate verification.

The initial limits are 150 files, 600,000 source bytes and 24,000 output tokens.
The request caps provider prices at $0.15 input and $0.50 output per million
tokens, disables provider fallbacks and requires non-collecting routing. A
conservative byte-based estimate above $1 blocks the request. These are request
controls, not a guarantee against all account charges. Confirm actual usage in
OpenRouter and set a dedicated key budget. Routing policies do not mean OpenRouter
never processes or records metadata about a request.

## Verify before relying on it

Use disposable PRs to establish that a clean change passes, a seeded privacy or
data-loss defect fails, a timeout cannot permit a merge, and a new push invalidates
the old result. Also test a base-branch update, draft-to-ready transition, changed
review instructions, skipped CI, and a forged check from another identity. Check
the actual merge restriction, not merely the presence of a red or green icon.

The model is a static reviewer. It does not execute Windows audio, clipboard or
installer behaviours. Its path-coverage declaration is not proof that it understood
every file. Keep behavioural tests and runtime validation, and measure missed
defects and false alarms before enabling an unattended repair/merge loop.

## Repository skills

`factory-review` coordinates an audit and explains feedback. `factory-repair`
addresses findings and requests fresh review. The controller, credentials and
branch protection enforce the boundary; these instruction files alone cannot.

## Sources

- [Required checks and their expected App](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub check runs](https://docs.github.com/en/rest/checks/runs)
- [Restricted environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)
- [Secure use of Actions](https://docs.github.com/en/actions/reference/security/secure-use)
- [OpenRouter provider controls](https://openrouter.ai/docs/guides/routing/provider-selection)
- [GLM 5.3 Flash model information](https://openrouter.ai/z-ai/glm-5.3-flash)
