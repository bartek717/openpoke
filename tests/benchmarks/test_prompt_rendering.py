"""V1 Benchmark: Prompt rendering size and timing at scale.

Reports four numbers per case:
1. Roster load time (roster.load() + _render_active_agents())
2. Message render time (prepare_message_with_history())
3. Serialized payload bytes (full OpenRouter request body)
4. Estimated tokens (tiktoken cl100k_base)
"""

from __future__ import annotations

import json
import time

import pytest
import tiktoken

from server.agents.interaction_agent.agent import (
    build_system_prompt,
    prepare_message_with_history,
)
from server.agents.interaction_agent.tools import get_tool_schemas

from .factories import populate_roster, write_conversation_log
from .metrics import BenchmarkReport, PromptRenderingResult


_ENCODER = tiktoken.get_encoding("cl100k_base")

AGENT_COUNTS = [100, 500, 1000]
CONVERSATION_TURNS = [50, 200, 500]


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _build_openrouter_payload(
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
) -> dict:
    """Reproduce the payload that request_chat_completion sends."""
    return {
        "model": "anthropic/claude-sonnet-4",
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": False,
        "tools": tools,
    }


@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
@pytest.mark.parametrize("conversation_turns", CONVERSATION_TURNS)
def test_prompt_size_at_scale(
    agent_count: int,
    conversation_turns: int,
    wired_env,
    data_dir,
):
    """Measure prompt rendering cost at different roster and conversation sizes."""
    env = wired_env

    # --- Populate data ---
    populate_roster(env.roster, agent_count)
    conv_path = data_dir / "conversation" / "poke_conversation.log"
    write_conversation_log(conv_path, conversation_turns)
    # Re-create the conversation log to pick up the written file
    # (the fixture created it before we wrote content)

    # --- Measure roster load + render ---
    t0 = time.perf_counter()
    env.roster.load()
    from server.agents.interaction_agent.agent import _render_active_agents
    _render_active_agents()
    roster_load_ms = (time.perf_counter() - t0) * 1000

    # --- Measure message rendering ---
    transcript = env.conversation_log.load_transcript()
    user_message = "Can you draft an email to Keith about the Monday standup?"

    t0 = time.perf_counter()
    messages = prepare_message_with_history(user_message, transcript)
    render_ms = (time.perf_counter() - t0) * 1000

    # --- Build full OpenRouter payload ---
    system_prompt = build_system_prompt()
    tools = get_tool_schemas()
    payload = _build_openrouter_payload(system_prompt, messages, tools)
    payload_json = json.dumps(payload, default=str)
    payload_bytes = len(payload_json.encode("utf-8"))

    # --- Token estimate ---
    estimated_tokens = _count_tokens(payload_json)

    # --- Report ---
    result = PromptRenderingResult(
        agent_count=agent_count,
        conversation_turns=conversation_turns,
        roster_load_ms=roster_load_ms,
        render_ms=render_ms,
        payload_bytes=payload_bytes,
        estimated_tokens=estimated_tokens,
    )

    print(
        f"\n  agents={agent_count:>5}, turns={conversation_turns:>4} | "
        f"roster={roster_load_ms:>6.2f}ms, render={render_ms:>6.2f}ms, "
        f"payload={payload_bytes / 1024:>6.1f}KB, tokens={estimated_tokens:>,}"
    )

    # --- Sanity assertions ---
    assert payload_bytes > 0
    assert estimated_tokens > 0
    # Roster portion should scale roughly linearly
    assert agent_count * 10 < payload_bytes  # at least ~10 bytes per agent


@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
def test_prompt_token_breakdown(agent_count: int, wired_env, data_dir):
    """Show where the tokens come from: system prompt vs roster vs conversation vs tools."""
    env = wired_env
    populate_roster(env.roster, agent_count)
    conv_path = data_dir / "conversation" / "poke_conversation.log"
    write_conversation_log(conv_path, 50)  # fixed 50 turns for comparison

    # System prompt alone
    system_prompt = build_system_prompt()
    system_tokens = _count_tokens(system_prompt)

    # Tools alone
    tools = get_tool_schemas()
    tools_tokens = _count_tokens(json.dumps(tools))

    # Roster portion
    from server.agents.interaction_agent.agent import _render_active_agents
    env.roster.load()
    roster_xml = _render_active_agents()
    roster_tokens = _count_tokens(roster_xml)

    # Conversation portion
    transcript = env.conversation_log.load_transcript()
    conv_tokens = _count_tokens(transcript)

    total = system_tokens + tools_tokens + roster_tokens + conv_tokens

    print(
        f"\n  agents={agent_count:>5} | "
        f"system={system_tokens:>,}, tools={tools_tokens:>,}, "
        f"roster={roster_tokens:>,}, conv={conv_tokens:>,}, "
        f"TOTAL={total:>,}"
    )

    # At 200K context, flag if we're over 50%
    if total > 100_000:
        print(f"  WARNING: {total:>,} tokens is >{total * 100 // 200_000}% of 200K context")
