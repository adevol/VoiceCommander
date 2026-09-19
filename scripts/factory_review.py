"""Read-only GLM audit. Candidate code is data and is never imported or executed."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

MODEL = "z-ai/glm-5.3-flash"
REASONING_EFFORT = "low"
CHECK = "factory/review"
MAX_BYTES = 600_000
MAX_OUTPUT = 24_000
MAX_FILES = 150
MAX_PRICE = {"prompt": 0.15, "completion": 0.50, "request": 0}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".lock", ".yaml", ".yml", ".ps1", ".iss", ".spec"}
TEXT_NAMES = {".gitignore", ".python-version", "LICENSE"}
ROOT = Path(__file__).resolve().parents[1]


class AuditError(Exception):
    """A safe-to-display reason why an audit cannot authorize a merge."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def strict_json(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise AuditError("Duplicate JSON keys are not valid review evidence.")
            value[key] = item
        return value

    return json.loads(text, object_pairs_hook=unique)


def api(url, token=None, payload=None, method=None):
    headers = {"Accept": "application/json", "User-Agent": "VoiceCommander-audit"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload).encode()
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=240) as response:
            data = response.read(8_000_001)
        if len(data) > 8_000_000:
            raise AuditError("API response exceeded its size limit.")
        return strict_json(data)
    except HTTPError as error:
        raise AuditError(f"API request failed with HTTP {error.code}.") from None
    except (URLError, TimeoutError, json.JSONDecodeError):
        raise AuditError("API unavailable, timed out, or returned invalid JSON.") from None


def file_kind(path):
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path:
        raise AuditError("Unsafe source path.")
    if any(part == ".env" or part.startswith(".env.") for part in p.parts):
        raise AuditError("Tracked environment secrets require separate handling.")
    if path == "benchmarks/jfk.wav":
        return "fixture"
    if p.suffix in TEXT_SUFFIXES or p.name in TEXT_NAMES:
        return "text"
    raise AuditError(f"No review policy for file: {path}")


def build_bundle(entries, read):
    if not entries or len(entries) > MAX_FILES:
        raise AuditError("Empty or oversized source manifest.")
    files, fixtures, size = {}, {}, 0
    for entry in entries:
        path = entry["path"]
        if entry.get("mode", "100644") not in {"100644", "100755"}:
            raise AuditError("Symlinks and submodules need a separate review policy.")
        kind = file_kind(path)
        if kind == "fixture":
            fixtures[path] = entry.get("sha", "local fixture")
            continue
        if entry.get("size", 0) > MAX_BYTES:
            raise AuditError("A source file exceeds the input budget.")
        content = read(entry)
        size += len(content)
        if size > MAX_BYTES:
            raise AuditError("Source exceeds the input budget; no files were silently omitted.")
        try:
            files[path] = content.decode("utf-8")
        except UnicodeDecodeError:
            raise AuditError(f"Non-UTF-8 source needs separate review: {path}") from None
    if not files:
        raise AuditError("No reviewable source files.")
    return {"files": files, "excluded_audio_fixtures": fixtures}


def validate_report(report, files):
    required = {"complete", "reviewed_files", "findings", "limitations"}
    if not isinstance(report, dict) or set(report) != required:
        raise AuditError("Reviewer returned an invalid report structure.")
    if type(report["complete"]) is not bool:
        raise AuditError("Reviewer completion field is not a boolean.")
    reviewed = report["reviewed_files"]
    if not isinstance(reviewed, list) or not all(isinstance(p, str) for p in reviewed):
        raise AuditError("Reviewer coverage is malformed.")
    if len(reviewed) != len(set(reviewed)) or set(reviewed) != set(files):
        raise AuditError("Reviewer did not account for every supplied source file.")
    if not isinstance(report["limitations"], list) or not all(
        isinstance(item, str) for item in report["limitations"]
    ):
        raise AuditError("Reviewer limitations are malformed.")
    if not isinstance(report["findings"], list):
        raise AuditError("Reviewer findings are malformed.")
    for finding in report["findings"]:
        fields = {"severity", "file", "line", "title", "evidence", "consequence", "verification"}
        if not isinstance(finding, dict) or set(finding) != fields:
            raise AuditError("A finding is malformed.")
        for field in fields - {"line"}:
            if not isinstance(finding[field], str) or not finding[field].strip():
                raise AuditError("A finding lacks supporting detail.")
        if finding["severity"] not in {"critical", "high", "medium", "low"}:
            raise AuditError("A finding has an invalid severity.")
        if finding["file"] not in files:
            raise AuditError("A finding references an unknown file.")
        line = finding["line"]
        if type(line) is not int or not 1 <= line <= len(files[finding["file"]].splitlines()):
            raise AuditError("A finding references an invalid line.")
    return report["complete"] and not any(
        finding["severity"] in {"critical", "high", "medium"} for finding in report["findings"]
    )


def review(bundle, context, token, brief):
    messages = [
        {"role": "system", "content": brief},
        {"role": "user", "content": json.dumps({"context": context, **bundle})},
    ]
    # UTF-8 byte count is a conservative text-token estimate, with framing allowance.
    input_bound = len(json.dumps(messages).encode()) + 4096
    upper_cost = (input_bound * MAX_PRICE["prompt"] + MAX_OUTPUT * MAX_PRICE["completion"]) / 1e6
    if upper_cost > 1.0:
        raise AuditError("Conservative per-request cost estimate exceeds $1; split the review.")
    response = api(
        "https://openrouter.ai/api/v1/chat/completions",
        token,
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_OUTPUT,
            "reasoning": {"effort": REASONING_EFFORT},
            "response_format": {"type": "json_object"},
            "provider": {
                "require_parameters": True,
                "data_collection": "deny",
                "max_price": MAX_PRICE,
                "allow_fallbacks": False,
            },
        },
    )
    try:
        if response["model"] != MODEL:
            raise AuditError("Provider returned a different model.")
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            raise AuditError(
                "Reviewer response was unfinished or refused.",
                {
                    "model": response["model"],
                    "finish_reason": choice["finish_reason"],
                    "usage": response.get("usage", {}),
                },
            )
        report = strict_json(choice["message"]["content"])
        passed = validate_report(report, bundle["files"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise AuditError("Provider returned an unusable review response.") from None
    return {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "context": {
            key: value
            for key, value in context.items()
            if key not in {"previous_versions", "changes", "requirements"}
        },
        "source_sha256": hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest(),
        "passed": passed,
        "report": report,
        "usage": response.get("usage", {}),
        "cost_estimate_upper_usd": upper_cost,
        "excluded_audio_fixtures": bundle["excluded_audio_fixtures"],
    }


def protected_path(path):
    return path.startswith((".github/", ".agents/")) or path in {
        "scripts/factory_review.py",
        "tests/test_factory_review.py",
    }


class GitHub:
    def __init__(self, repository, token, publisher_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise AuditError("Invalid repository name.")
        self.root = f"https://api.github.com/repos/{repository}"
        self.token = token
        self.publisher_id = int(publisher_id)

    def call(self, path, payload=None, method=None):
        return api(self.root + path, self.token, payload, method)

    def pages(self, path):
        result = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 31):
            items = self.call(f"{path}{separator}per_page=100&page={page}")
            result.extend(items)
            if len(items) < 100:
                return result
        raise AuditError("GitHub pagination limit reached.")

    def source(self, sha):
        tree = self.call(f"/git/trees/{sha}?recursive=1")
        if tree.get("truncated"):
            raise AuditError("GitHub truncated the source tree.")
        entries = [entry for entry in tree["tree"] if entry["type"] != "tree"]

        def read(entry):
            blob = self.call(f"/git/blobs/{entry['sha']}")
            if blob["encoding"] != "base64":
                raise AuditError("Unexpected GitHub blob encoding.")
            return base64.b64decode(blob["content"])

        return build_bundle(entries, read)

    def history(self, head):
        history = []
        for page in range(1, 31):
            data = self.call(
                f"/commits/{head}/check-runs?check_name={quote(CHECK, safe='')}"
                f"&filter=all&per_page=100&page={page}"
            )["check_runs"]
            history.extend(data)
            if len(data) < 100:
                return history
        raise AuditError("Review history exceeded its pagination limit.")


def audit_pr(github, number, token, brief, output):
    pr = github.call(f"/pulls/{number}")
    if pr["state"] != "open" or pr["base"]["ref"] not in {"master", "main"}:
        return True
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    context = {
        "pr": number,
        "head": head,
        "base": base,
        "title": pr["title"],
        "requirements": pr.get("body") or "",
        "policy_sha256": hashlib.sha256(brief.encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": MODEL,
    }
    source_context = {
        key: value for key, value in context.items() if key not in {"pr", "title", "requirements"}
    }
    source_id = hashlib.sha256(json.dumps(source_context, sort_keys=True).encode()).hexdigest()
    context_id = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    identity = f"{source_id}:{context_id}"
    check = github.call(
        "/check-runs",
        {"name": CHECK, "head_sha": head, "status": "in_progress", "external_id": identity},
    )
    result = None
    title = "Independent audit unavailable"
    try:
        if pr.get("draft"):
            raise AuditError("Draft PR: review is pending readiness.")
        if pr.get("author_association") not in {"OWNER", "MEMBER", "COLLABORATOR"}:
            raise AuditError("This internal audit lane requires a trusted repository contributor.")
        cached = None
        for previous_check in github.history(head):
            if (
                previous_check["app"]["id"] != github.publisher_id
                or previous_check["status"] != "completed"
            ):
                continue
            old_title = previous_check.get("output", {}).get("title")
            old_id = previous_check.get("external_id") or ""
            if old_title in {"Independent audit found defects", "Independent audit incomplete"}:
                if old_id.startswith(source_id + ":"):
                    cached = previous_check
                    break  # Metadata edits cannot reroll a substantive failed source audit.
            elif old_title == "Independent audit passed" and old_id == identity:
                cached = previous_check
        changes = github.pages(f"/pulls/{number}/files")
        if len(changes) != pr["changed_files"]:
            raise AuditError("GitHub did not return all changed files.")
        for change in changes:
            paths = [change["filename"], change.get("previous_filename", "")]
            if any(protected_path(path) for path in paths):
                raise AuditError(
                    "Review-control changes need a separately authorized policy update."
                )
            if any(path and file_kind(path) == "fixture" for path in paths):
                raise AuditError("Binary audio changes require separate verification.")
        if cached:
            passed = cached["conclusion"] == "success"
            summary = cached["output"]["summary"]
            title = cached["output"]["title"]
        else:
            context["changes"] = changes
            bundle = github.source(head)
            previous = github.source(base)
            context["previous_versions"] = {
                path: text
                for path, text in previous["files"].items()
                if path not in bundle["files"] or bundle["files"][path] != text
            }
            result = review(bundle, context, token, brief)
            passed = result["passed"]
            summary = json.dumps(result, indent=2)
            if passed:
                title = "Independent audit passed"
            elif result["report"]["complete"]:
                title = "Independent audit found defects"
            else:
                title = "Independent audit incomplete"
        current = github.call(f"/pulls/{number}")
        if (
            current["state"] != "open"
            or current.get("draft")
            or current["head"]["sha"] != head
            or current["base"]["sha"] != base
            or current["base"]["ref"] != pr["base"]["ref"]
            or current["title"] != pr["title"]
            or current.get("body") != pr.get("body")
        ):
            raise AuditError("PR or target changed during review; a fresh audit is required.")
    except AuditError as error:
        passed, summary = False, str(error)
        result = None
        title = "Independent audit unavailable"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"pr-{number}-{head}.json").write_text(
        json.dumps(result or {"passed": passed, "message": summary, "context": context}, indent=2),
        encoding="utf-8",
    )
    github.call(
        f"/check-runs/{check['id']}",
        {
            "status": "completed",
            "conclusion": "success" if passed else "failure",
            "output": {
                "title": title,
                "summary": summary[:60_000],
            },
        },
        "PATCH",
    )
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local", type=Path, help="Audit a manifest snapshot without GitHub writes"
    )
    parser.add_argument("--output", type=Path, default=Path("audit-results"))
    args = parser.parse_args()
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise AuditError("OPENROUTER_API_KEY is not configured.")
    brief = (ROOT / ".github/factory-review.md").read_text(encoding="utf-8")
    if args.local:
        root = args.local.resolve()
        entries = json.loads((root / "AUDIT-MANIFEST.json").read_text(encoding="utf-8-sig"))

        def read(entry):
            source = root / entry["path"]
            if source.is_symlink() or not source.resolve().is_relative_to(root):
                raise AuditError("Local manifest escapes its snapshot.")
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest().lower() != entry["sha256"].lower():
                raise AuditError("Snapshot file no longer matches its manifest.")
            return content

        args.output.mkdir(parents=True, exist_ok=True)
        try:
            result = review(build_bundle(entries, read), {"mode": "local snapshot"}, token, brief)
        except AuditError as error:
            (args.output / "glm-audit-error.json").write_text(
                json.dumps(
                    {
                        "passed": False,
                        "error": str(error),
                        "diagnostics": error.details,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
        (args.output / "glm-audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Audit complete. Passed: {result['passed']}. Report saved.")
        return 0 if result["passed"] else 1
    github = GitHub(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["FACTORY_GITHUB_TOKEN"],
        os.environ["FACTORY_REVIEW_APP_ID"],
    )
    # Every event reconciles all PRs. The workflow serializes runs; its scheduled sweep
    # recovers events GitHub coalesces while another run is pending.
    numbers = [pr["number"] for pr in github.pages("/pulls?state=open")]
    passed = True
    for number in numbers:
        passed = audit_pr(github, number, token, brief, args.output) and passed
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
