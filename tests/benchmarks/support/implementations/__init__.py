"""Benchmark-only implementations for agent-overload experiments."""

from __future__ import annotations

import os
import time
from typing import Dict, List

from server.agents.interaction_agent import agent as interaction_agent_agent
from server.services.execution import get_agent_roster

from .baseline.implementation import BaselineFullRosterImplementation
from .keyword_topk.implementation import KeywordTopKImplementation
from .types import BenchmarkImplementation, RenderedAgentContext
from .utils import render_agent_xml


_DEFAULT_IMPLEMENTATION = "baseline"

IMPLEMENTATIONS: Dict[str, BenchmarkImplementation] = {
    "baseline": BaselineFullRosterImplementation(),
    "keyword_topk": KeywordTopKImplementation(),
}


def get_selected_implementation_name() -> str:
    return os.getenv("OPENPOKE_BENCHMARK_IMPLEMENTATION", _DEFAULT_IMPLEMENTATION)


def get_selected_implementation() -> BenchmarkImplementation:
    name = get_selected_implementation_name().strip().lower()
    implementation = IMPLEMENTATIONS.get(name)
    if implementation is None:
        raise ValueError(
            "OPENPOKE_BENCHMARK_IMPLEMENTATION must be one of "
            f"{sorted(IMPLEMENTATIONS)}, got {name!r}"
        )
    return implementation


def render_active_agents_for_benchmark(
    latest_text: str,
    transcript: str,
) -> RenderedAgentContext:
    roster = get_agent_roster()
    roster.load()
    agents = roster.get_agents()
    if not agents:
        return RenderedAgentContext(
            xml="None",
            selected_agents=[],
            total_agents=0,
            impl_overhead_ms=0.0,
        )

    implementation = get_selected_implementation()
    t0 = time.perf_counter()
    selected = implementation.select_agents(latest_text, transcript, list(agents))
    impl_overhead_ms = (time.perf_counter() - t0) * 1000

    selected_set = set(selected)
    ordered_selected = [agent for agent in agents if agent in selected_set]
    if not ordered_selected:
        ordered_selected = list(agents)

    return RenderedAgentContext(
        xml=render_agent_xml(ordered_selected),
        selected_agents=ordered_selected,
        total_agents=len(agents),
        impl_overhead_ms=impl_overhead_ms,
    )


def prepare_message_with_history_impl(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> list[dict[str, str]]:
    sections: List[str] = []
    sections.append(interaction_agent_agent._render_conversation_history(transcript))
    rendered = render_active_agents_for_benchmark(latest_text, transcript)
    sections.append(f"<active_agents>\n{rendered.xml}\n</active_agents>")
    sections.append(interaction_agent_agent._render_current_turn(latest_text, message_type))
    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


__all__ = [
    "BenchmarkImplementation",
    "RenderedAgentContext",
    "get_selected_implementation_name",
    "get_selected_implementation",
    "prepare_message_with_history_impl",
    "render_active_agents_for_benchmark",
]
