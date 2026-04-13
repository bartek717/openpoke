"""Benchmark metrics collection and reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PromptRenderingResult:
    agent_count: int
    conversation_turns: int
    roster_load_ms: float
    render_ms: float
    payload_bytes: int
    estimated_tokens: int


@dataclass
class RosterIOResult:
    agent_count: int
    load_ms: float
    save_ms: float
    add_agent_ms: float
    contention_valid_json: bool = True
    contention_correct_count: bool = True


@dataclass
class BenchmarkReport:
    prompt_rendering: List[PromptRenderingResult] = field(default_factory=list)
    roster_io: List[RosterIOResult] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "prompt_rendering": [asdict(r) for r in self.prompt_rendering],
            "roster_io": [asdict(r) for r in self.roster_io],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def print_prompt_table(self) -> None:
        if not self.prompt_rendering:
            return
        header = f"{'Agents':>8} {'Turns':>7} {'Roster ms':>11} {'Render ms':>11} {'Payload KB':>11} {'Tokens':>10}"
        sep = "-" * len(header)
        print("\nPrompt Rendering")
        print(sep)
        print(header)
        print(sep)
        for r in self.prompt_rendering:
            print(
                f"{r.agent_count:>8} {r.conversation_turns:>7} "
                f"{r.roster_load_ms:>11.2f} {r.render_ms:>11.2f} "
                f"{r.payload_bytes / 1024:>11.1f} {r.estimated_tokens:>10,}"
            )
        print(sep)

    def print_roster_table(self) -> None:
        if not self.roster_io:
            return
        header = f"{'Agents':>8} {'Load ms':>9} {'Save ms':>9} {'Add ms':>9} {'JSON OK':>9} {'Count OK':>10}"
        sep = "-" * len(header)
        print("\nRoster I/O")
        print(sep)
        print(header)
        print(sep)
        for r in self.roster_io:
            print(
                f"{r.agent_count:>8} {r.load_ms:>9.2f} {r.save_ms:>9.2f} "
                f"{r.add_agent_ms:>9.2f} {'yes':>9 if r.contention_valid_json else 'NO':>9} "
                f"{'yes':>10 if r.contention_correct_count else 'NO':>10}"
            )
        print(sep)
