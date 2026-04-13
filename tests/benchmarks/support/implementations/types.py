"""Shared types for benchmark implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass(frozen=True)
class RenderedAgentContext:
    xml: str
    selected_agents: List[str]
    total_agents: int
    impl_overhead_ms: float


class BenchmarkImplementation(Protocol):
    name: str

    def select_agents(
        self,
        latest_text: str,
        transcript: str,
        agents: List[str],
    ) -> List[str]:
        ...
