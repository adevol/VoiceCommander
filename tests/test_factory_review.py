from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

SPEC = importlib.util.spec_from_file_location(
    "factory_review", Path(__file__).resolve().parents[1] / "scripts/factory_review.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def clean_report():
    return {"complete": True, "reviewed_files": ["app.py"], "findings": [], "limitations": []}


def defect(severity="high"):
    return {
        "severity": severity,
        "file": "app.py",
        "line": 1,
        "title": "Cross-user access",
        "evidence": "Query returns all users' files",
        "consequence": "Private files disclosed",
        "verification": "Bob must not be able to list Alice's files",
    }


class WorkflowTests(unittest.TestCase):
    def test_privileged_audit_requires_explicit_repository_activation(self):
        workflow = yaml.safe_load(
            (audit.ROOT / ".github/workflows/factory-review.yml").read_text(encoding="utf-8")
        )
        job = workflow["jobs"]["audit"]
        self.assertEqual(job.get("if"), "${{ vars.FACTORY_REVIEW_ENABLED == 'true' }}")
        self.assertEqual(job["environment"], "factory-review")
        self.assertFalse(job.get("continue-on-error", False))
        for step in job["steps"]:
            self.assertFalse(step.get("continue-on-error", False))


class ReportTests(unittest.TestCase):
    files = {"app.py": "print('example')\n"}

    def test_clean_complete_report_passes(self):
        self.assertTrue(audit.validate_report(clean_report(), self.files))

    def test_supported_defects_block_but_low_is_advisory(self):
        for severity in ["critical", "high", "medium", "low"]:
            report = clean_report()
            report["findings"] = [defect(severity)]
            self.assertEqual(audit.validate_report(report, self.files), severity == "low")

    def test_incomplete_never_passes_even_without_findings(self):
        report = clean_report()
        report["complete"] = False
        self.assertFalse(audit.validate_report(report, self.files))

    def test_missing_extra_or_duplicate_coverage_rejected(self):
        for coverage in [[], ["other.py"], ["app.py", "app.py"], ["app.py", "other.py"]]:
            report = clean_report()
            report["reviewed_files"] = coverage
            with self.assertRaises(audit.AuditError):
                audit.validate_report(report, self.files)

    def test_bad_evidence_and_types_rejected(self):
        for field, value in [
            ("line", True),
            ("line", 200),
            ("file", "unknown.py"),
            ("severity", {}),
            ("evidence", ""),
        ]:
            report = clean_report()
            finding = defect()
            finding[field] = value
            report["findings"] = [finding]
            with self.assertRaises(audit.AuditError):
                audit.validate_report(report, self.files)

    def test_secret_paths_unknown_files_and_symlinks_are_blocked(self):
        for path, mode in [
            (".env", "100644"),
            ("../outside.py", "100644"),
            ("payload.exe", "100644"),
            ("app.py", "120000"),
        ]:
            read = Mock(return_value=b"data")
            with self.assertRaises(audit.AuditError):
                audit.build_bundle([{"path": path, "mode": mode}], read)
            read.assert_not_called()

    def test_no_silent_context_truncation(self):
        with self.assertRaises(audit.AuditError):
            audit.build_bundle([{"path": "app.py"}], lambda entry: b"x" * (audit.MAX_BYTES + 1))

    def test_audio_fixture_is_explicitly_excluded(self):
        bundle = audit.build_bundle(
            [{"path": "app.py"}, {"path": "benchmarks/jfk.wav", "sha": "fixture-sha"}],
            lambda entry: b"pass\n",
        )
        self.assertEqual(bundle["files"], {"app.py": "pass\n"})
        self.assertEqual(bundle["excluded_audio_fixtures"], {"benchmarks/jfk.wav": "fixture-sha"})

    def test_truncated_provider_response_cannot_pass(self):
        bundle = {"files": self.files, "excluded_audio_fixtures": {}}
        response = {"model": audit.MODEL, "choices": [{"finish_reason": "length"}]}
        with patch.object(audit, "api", return_value=response):
            with self.assertRaises(audit.AuditError):
                audit.review(bundle, {}, "fake-token", "trusted instructions")

    def test_duplicate_json_keys_cannot_override_a_failed_verdict(self):
        with self.assertRaises(audit.AuditError):
            audit.strict_json('{"complete": false, "complete": true}')

    def test_request_is_text_only_and_has_provider_limits(self):
        bundle = {"files": self.files, "excluded_audio_fixtures": {}}
        response = {
            "model": audit.MODEL,
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(clean_report())}}
            ],
        }
        with patch.object(audit, "api", return_value=response) as request:
            result = audit.review(bundle, {}, "fake-token", "trusted instructions")
        self.assertTrue(result["passed"])
        body = request.call_args.args[2]
        self.assertNotIn("tools", body)
        self.assertEqual(body["provider"]["data_collection"], "deny")
        self.assertEqual(body["provider"]["max_price"], audit.MAX_PRICE)


class GateTests(unittest.TestCase):
    def fake_github(self, changed="app.py", stale=False, cache=None):
        github = Mock()
        github.publisher_id = 42
        pr = {
            "number": 1,
            "state": "open",
            "draft": False,
            "title": "Fix",
            "body": "Spec",
            "author_association": "OWNER",
            "head": {"sha": "a" * 40},
            "base": {"sha": "b" * 40, "ref": "master"},
            "changed_files": 1,
        }
        reads = 0

        def call(path, payload=None, method=None):
            nonlocal reads
            if path.startswith("/pulls/"):
                reads += 1
                current = copy.deepcopy(pr)
                if stale and reads > 1:
                    current["head"]["sha"] = "c" * 40
                return current
            if path.startswith("/commits/"):
                return {"check_runs": cache or []}
            if path == "/check-runs":
                return {"id": 123}
            return {}

        github.call.side_effect = call
        github.pull_request = pr
        github.pages.return_value = [{"filename": changed}]
        github.history.return_value = cache or []
        github.source.return_value = {"files": {"app.py": "pass\n"}, "excluded_audio_fixtures": {}}
        return github

    def test_new_commit_invalidates_result_and_saved_artifact(self):
        github = self.fake_github(stale=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(audit, "review", return_value={"passed": True}):
                self.assertFalse(audit.audit_pr(github, 1, "fake", "brief", Path(directory)))
            saved = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertFalse(saved["passed"])
        self.assertEqual(github.call.call_args.args[1]["conclusion"], "failure")

    def test_review_cannot_edit_its_own_policy(self):
        github = self.fake_github(changed=".github/factory-review.md")
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, "review") as review:
            self.assertFalse(audit.audit_pr(github, 1, "fake", "brief", Path(directory)))
            review.assert_not_called()

    def test_api_failure_is_published_as_blocking(self):
        github = self.fake_github()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(audit, "review", side_effect=audit.AuditError("API unavailable")):
                self.assertFalse(audit.audit_pr(github, 1, "fake", "brief", Path(directory)))
        self.assertEqual(github.call.call_args.args[1]["conclusion"], "failure")

    def test_complete_clean_candidate_gets_success_on_exact_head(self):
        github = self.fake_github()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(audit, "review", return_value={"passed": True}):
                self.assertTrue(audit.audit_pr(github, 1, "fake", "brief", Path(directory)))
        created = next(call for call in github.call.call_args_list if call.args[0] == "/check-runs")
        self.assertEqual(created.args[1]["head_sha"], "a" * 40)
        self.assertEqual(github.call.call_args.args[1]["conclusion"], "success")

    def test_title_edits_and_reopened_prs_do_not_reroll_failed_code(self):
        github = self.fake_github()
        result = {"passed": False, "report": {"complete": True}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(audit, "review", return_value=result) as review:
                self.assertFalse(audit.audit_pr(github, 1, "fake", "brief", Path(directory)))
                created = next(
                    call for call in github.call.call_args_list if call.args[0] == "/check-runs"
                )
                published = github.call.call_args.args[1]
                github.history.return_value = [
                    {
                        "app": {"id": 42},
                        "status": "completed",
                        "conclusion": "failure",
                        "external_id": created.args[1]["external_id"],
                        "output": published["output"],
                    }
                ]
                github.pull_request["title"] += " "
                self.assertFalse(audit.audit_pr(github, 1, "fake", "brief", Path(directory)))
                self.assertFalse(audit.audit_pr(github, 2, "fake", "brief", Path(directory)))
                self.assertEqual(review.call_count, 1)

    def test_draft_to_ready_and_outages_can_recover(self):
        for cause in ["draft", "outage"]:
            github = self.fake_github()
            github.pull_request["draft"] = cause == "draft"
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(audit, "review", side_effect=audit.AuditError("outage")):
                    self.assertFalse(audit.audit_pr(github, 1, "fake", "brief", Path(directory)))
                created = next(
                    call for call in github.call.call_args_list if call.args[0] == "/check-runs"
                )
                github.history.return_value = [
                    {
                        "app": {"id": 42},
                        "status": "completed",
                        "conclusion": "failure",
                        "external_id": created.args[1]["external_id"],
                        "output": github.call.call_args.args[1]["output"],
                    }
                ]
                github.pull_request["draft"] = False
                with patch.object(audit, "review", return_value={"passed": True}) as review:
                    self.assertTrue(audit.audit_pr(github, 1, "fake", "brief", Path(directory)))
                    review.assert_called_once()

    def test_history_is_paginated(self):
        github = audit.GitHub("owner/repo", "fake", 42)
        with patch.object(
            github,
            "call",
            side_effect=[{"check_runs": [{}] * 100}, {"check_runs": [{"id": "older failure"}]}],
        ):
            self.assertEqual(len(github.history("a" * 40)), 101)


if __name__ == "__main__":
    unittest.main()
