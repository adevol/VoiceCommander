"""Load editable prompt text from packaged YAML."""

from __future__ import annotations

from functools import cache
from importlib.resources import files

import yaml

from ..settings import POSTPROCESS_STYLES, Settings


@cache
def _templates() -> dict:
    resource = files(__package__).joinpath("postprocess.yaml")
    return yaml.safe_load(resource.read_text(encoding="utf-8"))


def postprocess_prompt(settings: Settings, markdown: bool = False) -> str:
    templates = _templates()
    strength = "light" if settings.postprocess_strength <= 25 else "medium"
    if settings.postprocess_strength > 75:
        strength = "strong"
    values = {
        "language": settings.language,
        "numbers": templates["numbers"]["auto" if settings.language == "auto" else "locale"],
        "strength": templates["strength"][strength],
        "style": POSTPROCESS_STYLES[settings.postprocess_style],
    }
    rules = [rule.format_map(values) for rule in templates["markdown" if markdown else "plain"]]
    if settings.vocabulary:
        rules.insert(2, templates["vocabulary"].format(vocabulary=settings.vocabulary))
    return "\n".join(rules)
