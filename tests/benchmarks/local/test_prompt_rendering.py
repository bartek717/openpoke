"""V1 Benchmark: Prompt rendering size and timing at scale.

Reports four numbers per case:
1. Roster load time (roster.load() + _render_active_agents())
2. Message render time (prepare_message_with_history())
3. Serialized payload bytes (full OpenRouter request body)
4. Estimated tokens (tiktoken cl100k_base)

This benchmark intentionally keeps conversation history fixed so we can
isolate how the active-agent roster alone affects prompt growth.
"""

from __future__ import annotations

import pytest

from ..conftest import AGENT_COUNTS
from ..support.metrics import BenchmarkReport, PromptRenderingResult
from ..support.prompt_stats import FIXED_CONVERSATION_TURNS, measure_prompt_stats


@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
def test_prompt_size_at_scale(
    agent_count: int,
    wired_env,
    data_dir,
):
    """Measure prompt rendering cost while varying roster size only."""
    stats = measure_prompt_stats(wired_env, data_dir, agent_count)

    # --- Report ---
    result = PromptRenderingResult(
        agent_count=agent_count,
        conversation_turns=FIXED_CONVERSATION_TURNS,
        roster_load_ms=stats.roster_load_ms,
        render_ms=stats.render_ms,
        payload_bytes=stats.payload_bytes,
        estimated_tokens=stats.estimated_tokens,
    )

    print(
        f"\n  agents={agent_count:>5}, turns={FIXED_CONVERSATION_TURNS:>4} | "
        f"roster={stats.roster_load_ms:>6.2f}ms, render={stats.render_ms:>6.2f}ms, "
        f"payload={stats.payload_bytes / 1024:>6.1f}KB, tokens={stats.estimated_tokens:>,}"
    )

    # --- Sanity assertions ---
    assert stats.payload_bytes > 0
    assert stats.estimated_tokens > 0
    # Roster portion should scale roughly linearly
    assert agent_count * 10 < stats.payload_bytes  # at least ~10 bytes per agent


@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
def test_prompt_token_breakdown(agent_count: int, wired_env, data_dir):
    """Show where the tokens come from: system prompt vs roster vs conversation vs tools."""
    stats = measure_prompt_stats(wired_env, data_dir, agent_count)
    total = (
        stats.system_tokens
        + stats.tools_tokens
        + stats.roster_tokens
        + stats.conversation_tokens
    )

    print(
        f"\n  agents={agent_count:>5} | "
        f"system={stats.system_tokens:>,}, tools={stats.tools_tokens:>,}, "
        f"roster={stats.roster_tokens:>,}, conv={stats.conversation_tokens:>,}, "
        f"TOTAL={total:>,}"
    )

    # At 200K context, flag if we're over 50%
    if total > 100_000:
        print(f"  WARNING: {total:>,} tokens is >{total * 100 // 200_000}% of 200K context")
