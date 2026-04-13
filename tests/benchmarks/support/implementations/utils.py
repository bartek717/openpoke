"""Shared helpers for benchmark implementations."""

from __future__ import annotations

import re
from html import escape
from typing import Iterable, List


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def render_agent_xml(agent_names: Iterable[str]) -> str:
    rendered: List[str] = []
    for agent_name in agent_names:
        name = escape(agent_name or "agent", quote=True)
        rendered.append(f'<agent name="{name}" />')
    return "\n".join(rendered) if rendered else "None"
