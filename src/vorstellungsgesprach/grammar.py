"""Grammar checking for user-written German text (e.g. practice interview answers).

Uses LanguageTool (via language_tool_python) running locally. Requires Java
to be installed on the system (see README for setup instructions).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import language_tool_python


@dataclass
class GrammarIssue:
    message: str
    suggestions: List[str]
    context: str
    offset: int
    error_length: int


@lru_cache(maxsize=1)
def _tool() -> language_tool_python.LanguageTool:
    """Loads the local LanguageTool instance once and reuses it (it's slow to start)."""
    return language_tool_python.LanguageTool("de-DE")


def check_text(text: str) -> List[GrammarIssue]:
    """Checks German text and returns a list of grammar/spelling issues found."""
    if not text or not text.strip():
        return []

    matches = _tool().check(text)

    return [
        GrammarIssue(
            message=match.message,
            suggestions=match.replacements[:5],  # limita a 5 sugestões para não sobrecarregar a UI
            context=match.context,
            offset=match.offset,
            error_length=match.errorLength,
        )
        for match in matches
    ]


def correct_text(text: str) -> str:
    """Returns the text with all detected issues auto-corrected (best guess)."""
    if not text or not text.strip():
        return text
    return _tool().correct(text)


def format_issues_for_display(issues: List[GrammarIssue]) -> str:
    """Formats issues as a readable summary, useful for quick CLI/notebook checks."""
    if not issues:
        return "Keine Grammatikfehler gefunden. ✅"

    lines = [f"{len(issues)} Problem(e) gefunden:\n"]
    for i, issue in enumerate(issues, start=1):
        suggestion_text = ", ".join(issue.suggestions) if issue.suggestions else "—"
        lines.append(f"{i}. {issue.message}")
        lines.append(f"   Vorschlag: {suggestion_text}")
        lines.append(f"   Kontext: ...{issue.context}...\n")

    return "\n".join(lines)