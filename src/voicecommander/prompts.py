"""Prompts used to turn raw dictation into finished text."""

from __future__ import annotations

from .settings import POSTPROCESS_STYLES, Settings


def postprocess_prompt(settings: Settings, markdown: bool = False) -> str:
    number_format_rule = (
        "Write numbers, dates, and times naturally for the language of the dictated text."
        if settings.language == "auto"
        else f"Write numbers, dates, and times naturally for the {settings.language} locale."
    )
    if markdown:
        rules = [
            "Turn this dictated description into the finished Markdown content the speaker intends;"
            " do not merely transcribe the description.",
            "Use valid, conventional Markdown with paragraphs, headings, lists, blockquotes, tables,"
            " and code blocks when the requested content calls for them.",
            "Interpret spoken layout and hierarchy instructions instead of including those instructions"
            " in the result.",
            "Format mathematical expressions as LaTeX: use $...$ for inline math and $$...$$ for"
            " display math.",
            "Preserve the speaker's meaning and supplied facts; do not invent content.",
            f"Style: {POSTPROCESS_STYLES[settings.postprocess_style]}",
            "Reply in the language of the dictated text.",
            'Apply spoken self-corrections: when the speaker says "scratch that" or "correction"'
            " or restarts a phrase, keep only the corrected version and never write the command itself.",
            number_format_rule,
            "Return only raw Markdown, without commentary or an outer Markdown code fence.",
        ]
    else:
        if settings.postprocess_strength <= 25:
            strength_rule = (
                "Only fix punctuation and obvious mis-hearings; keep the wording exactly as spoken."
            )
        elif settings.postprocess_strength <= 75:
            strength_rule = (
                "Lightly clean up the text, keeping the speaker's wording and sentence order."
            )
        else:
            strength_rule = "Rewrite freely for clarity while preserving the meaning."
        rules = [
            "Edit this dictated text without changing its meaning or adding information.",
            strength_rule,
            f"Style: {POSTPROCESS_STYLES[settings.postprocess_style]}",
            "Reply in the language of the dictated text.",
            'Apply spoken self-corrections: when the speaker says "scratch that" or "correction"'
            " or restarts a phrase, keep only the corrected version and never write the command itself.",
            'Convert spoken commands ("new paragraph", "bullet point", "quote ... unquote") and'
            " formatting cues into paragraphs, bullets, numbered lists, and punctuation.",
            number_format_rule,
            "Return only the finished text.",
        ]
    if settings.vocabulary:
        rules.insert(2, f"Keep these terms exactly as spelled here: {settings.vocabulary}")
    return "\n".join(rules)
