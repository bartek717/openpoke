"""Baseline implementation: expose the full roster."""

from __future__ import annotations

from typing import List


class BaselineFullRosterImplementation:
    name = "baseline"

    def select_agents(
        self,
        latest_text: str,
        transcript: str,
        agents: List[str],
    ) -> List[str]:
        return list(agents)
