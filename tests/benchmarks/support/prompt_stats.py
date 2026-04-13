"""Shared prompt-rendering helpers for benchmark reporting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from server.agents.interaction_agent import agent as interaction_agent_agent
from server.agents.interaction_agent.agent import build_system_prompt
from server.agents.interaction_agent.tools import get_tool_schemas

from .factories import populate_roster, write_conversation_log
from .implementations import (
    render_active_agents_for_benchmark,
)


_ENCODER = tiktoken.get_encoding("cl100k_base")

FIXED_CONVERSATION_TURNS = 50
DEFAULT_BENCHMARK_USER_MESSAGE = (
    "Can you draft an email to Keith about the Monday standup?"
)


@dataclass(frozen=True)
class PromptStats:
    agent_count: int
    conversation_turns: int
    roster_load_ms: float
    render_ms: float
    impl_overhead_ms: float
    payload_bytes: int
    estimated_tokens: int
    system_tokens: int
    tools_tokens: int
    roster_tokens: int
    conversation_tokens: int


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def build_openrouter_payload(
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reproduce the request body sent to OpenRouter."""
    return {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": False,
        "tools": tools,
    }


def measure_prompt_stats(
    env: Any,
    data_dir: Path,
    agent_count: int,
    *,
    conversation_turns: int = FIXED_CONVERSATION_TURNS,
    user_message: str = DEFAULT_BENCHMARK_USER_MESSAGE,
) -> PromptStats:
    """Measure prompt rendering size and timing for a roster size."""

    env.roster.clear()
    env.conversation_log.clear()
    populate_roster(env.roster, agent_count)

    conv_path = data_dir / "conversation" / "poke_conversation.log"
    write_conversation_log(conv_path, conversation_turns)

    transcript = env.conversation_log.load_transcript()

    t0 = time.perf_counter()
    rendered_agents = render_active_agents_for_benchmark(user_message, transcript)
    agent_context_ms = (time.perf_counter() - t0) * 1000
    roster_xml = rendered_agents.xml

    t0 = time.perf_counter()
    sections = [
        interaction_agent_agent._render_conversation_history(transcript),
        f"<active_agents>\n{roster_xml}\n</active_agents>",
        interaction_agent_agent._render_current_turn(user_message, "user"),
    ]
    messages = [{"role": "user", "content": "\n\n".join(sections)}]
    render_ms = (time.perf_counter() - t0) * 1000

    system_prompt = build_system_prompt()
    tools = get_tool_schemas()
    model = getattr(getattr(env, "settings", None), "interaction_agent_model", "")
    payload = build_openrouter_payload(model, system_prompt, messages, tools)
    payload_json = json.dumps(payload, default=str)

    return PromptStats(
        agent_count=agent_count,
        conversation_turns=conversation_turns,
        roster_load_ms=max(agent_context_ms - rendered_agents.impl_overhead_ms, 0.0),
        render_ms=render_ms,
        impl_overhead_ms=rendered_agents.impl_overhead_ms,
        payload_bytes=len(payload_json.encode("utf-8")),
        estimated_tokens=count_tokens(payload_json),
        system_tokens=count_tokens(system_prompt),
        tools_tokens=count_tokens(json.dumps(tools)),
        roster_tokens=count_tokens(roster_xml),
        conversation_tokens=count_tokens(transcript),
    )
