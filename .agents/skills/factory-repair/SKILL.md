---
name: factory-repair
description: Repair evidenced findings from an independent pre-merge review on the author branch, rerun relevant checks and return the revision for fresh review within a supplied budget.
---

# Repair a reviewed candidate

Read the task requirements, review findings, reviewed head SHA and controller budget. Verify the working branch matches the expected revision and preserve unrelated user changes. If the candidate has moved, report the mismatch so the controller can refresh the task. Do not repair against stale evidence by guessing.

For each blocking finding, establish its triggering condition and the requirement it violates. Reproduce a behavioural defect where practical. Make a focused correction and add a regression check when it can meaningfully detect the reported failure. Run relevant existing checks and record actual commands and results. A disagreement with a finding needs evidence and a fresh independent decision, not deletion of the finding.

Use the assigned author model. A coordinator may delegate a bounded implementation task to a configured cheaper worker when the work has a clear interface and independent verification. Preserve the controller's model choices, file scope and budget. Do not introduce additional agents just to fill roles.

Do not weaken protected tests, change the active review policy, disable CI or publish a passing gate as part of the repair. A faulty test or requirement can be reported with evidence for separate resolution under the project's policy. Keep the repair on the candidate branch. Commit or push only through the already-authorized publishing path.

Return the resulting revision, changes mapped to findings, verification evidence and unresolved issues. Request a fresh independent review for the new revision. A writer's claim that a finding is fixed is not reviewer approval.

Respect the supplied repair-attempt, cost and time limits. The controller counts attempts and API usage independently. If no loop budget is supplied, perform at most the single requested repair pass and return control. At exhaustion or an unresolved requirement, leave the candidate unmerged and record why work stopped.
