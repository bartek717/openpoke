"""V3 Benchmark: Live accuracy tests against the real interaction runtime.

Measures whether the interaction loop can still reuse or create the right
agent as the roster grows. Marked with @pytest.mark.live — skipped by
default, run with: pytest -m live
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import pytest

from server.agents.interaction_agent.runtime import InteractionAgentRuntime

from ..conftest import (
    AGENT_COUNTS,
    LIVE_TRIALS,
    ToolCallRecorder,
    get_benchmark_implementation_name,
    is_live_group_enabled,
)
from ..support.factories import populate_roster, write_conversation_log
from ..support.prompt_stats import FIXED_CONVERSATION_TURNS, measure_prompt_stats


TRIALS = LIVE_TRIALS
LIVE_RESULTS: Dict[str, Dict[int, Dict[str, Any]]] = {
    "reuse": {},
    "creation": {},
}
PROMPT_RESULTS: Dict[int, Dict[str, Any]] = {}
SUMMARY_METADATA: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_calls_from_recorder(recorder: ToolCallRecorder) -> List[Dict[str, Any]]:
    """Convert recorded tool invocations into a serializable benchmark shape."""
    return [
        {
            "name": invocation.name,
            "arguments": invocation.arguments,
            "result": invocation.result.payload,
        }
        for invocation in recorder.invocations
    ]


async def _run_live_interaction(
    user_message: str,
    recorder: ToolCallRecorder,
) -> tuple[bool, str | None, List[Dict[str, Any]]]:
    """Execute the real InteractionAgentRuntime and return its observed tool calls."""

    recorder.invocations.clear()
    runtime = InteractionAgentRuntime()
    result = await runtime.execute(user_message)
    return result.success, result.error, _tool_calls_from_recorder(recorder)


def _score_reuse(tool_calls: List[Dict], target_name: str) -> float:
    """Score agent reuse accuracy.

    1.0 = exactly one delegation to the expected existing agent, else 0.0
    """
    agent_calls = [tc for tc in tool_calls if tc["name"] == "send_message_to_agent"]
    if len(agent_calls) != 1:
        return 0.0

    chosen_name = str(agent_calls[0]["arguments"].get("agent_name", "")).strip()
    return 1.0 if chosen_name == target_name else 0.0


def _score_creation(tool_calls: List[Dict], roster_names: List[str]) -> float:
    """Score agent creation accuracy.

    1.0 = exactly one delegation to a non-empty agent name not in roster, else 0.0
    """
    agent_calls = [tc for tc in tool_calls if tc["name"] == "send_message_to_agent"]
    if len(agent_calls) != 1:
        return 0.0

    chosen_name = str(agent_calls[0]["arguments"].get("agent_name", "")).strip()
    roster_set = set(roster_names)

    if chosen_name and chosen_name not in roster_set:
        return 1.0
    return 0.0


def _record_live_result(
    group: str,
    agent_count: int,
    scores: List[float],
    mean_score: float,
) -> None:
    LIVE_RESULTS[group][agent_count] = {
        "scores": scores.copy(),
        "mean": mean_score,
    }


def _record_prompt_result(agent_count: int, stats: Any) -> None:
    first_call_ms = stats.roster_load_ms + stats.render_ms
    PROMPT_RESULTS[agent_count] = {
        "payload_bytes": stats.payload_bytes,
        "estimated_tokens": stats.estimated_tokens,
        "roster_tokens": stats.roster_tokens,
        "impl_overhead_ms": stats.impl_overhead_ms,
        "first_call_ms": first_call_ms,
        "total_pre_llm_ms": stats.impl_overhead_ms + first_call_ms,
    }


def _format_accuracy_summary_table() -> str:
    enabled_groups = [
        group for group in ("reuse", "creation") if is_live_group_enabled(group)
    ]
    if not enabled_groups:
        return "No live groups enabled."

    baseline = PROMPT_RESULTS.get(5)
    headers = [
        "agents",
        "prompt_kb",
        "kb_vs_5",
        "tokens",
        "toks_vs_5",
        "roster_toks",
        "roster_vs_5",
        "roster_pct",
        "impl_overhead_ms",
        "first_call_ms",
        "total_pre_llm_ms",
        *enabled_groups,
    ]
    align = [
        "-----:",
        "--------:",
        "-------:",
        "------:",
        "---------:",
        "-----------:",
        "-----------:",
        "----------:",
        "----------------:",
        "-------------:",
        "----------------:",
        *["-----:" for _ in enabled_groups],
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]

    for agent_count in AGENT_COUNTS:
        prompt = PROMPT_RESULTS.get(agent_count)
        if prompt is not None and prompt["estimated_tokens"] > 0:
            roster_pct = prompt["roster_tokens"] / prompt["estimated_tokens"] * 100
        else:
            roster_pct = None

        if prompt is not None and baseline is not None:
            kb_vs_5 = (prompt["payload_bytes"] - baseline["payload_bytes"]) / 1024
            toks_vs_5 = prompt["estimated_tokens"] - baseline["estimated_tokens"]
            roster_vs_5 = prompt["roster_tokens"] - baseline["roster_tokens"]
        else:
            kb_vs_5 = toks_vs_5 = roster_vs_5 = None

        row = [
            str(agent_count),
            f"{prompt['payload_bytes'] / 1024:.1f}" if prompt is not None else "-",
            f"{kb_vs_5:+.1f}" if kb_vs_5 is not None else "-",
            f"{prompt['estimated_tokens']:,}" if prompt is not None else "-",
            f"{toks_vs_5:+,}" if toks_vs_5 is not None else "-",
            f"{prompt['roster_tokens']:,}" if prompt is not None else "-",
            f"{roster_vs_5:+,}" if roster_vs_5 is not None else "-",
            f"{roster_pct:.1f}%" if roster_pct is not None else "-",
            f"{prompt['impl_overhead_ms']:.2f}" if prompt is not None else "-",
            f"{prompt['first_call_ms']:.2f}" if prompt is not None else "-",
            f"{prompt['total_pre_llm_ms']:.2f}" if prompt is not None else "-",
        ]
        for group in enabled_groups:
            result = LIVE_RESULTS[group].get(agent_count)
            row.append(f"{result['mean']:.2f}" if result is not None else "-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(f"model: {SUMMARY_METADATA['model']}")
    lines.append(f"implementation: {SUMMARY_METADATA['implementation']}")
    lines.append(f"trials: {SUMMARY_METADATA['trials']}")
    lines.append(f"agent_counts: {SUMMARY_METADATA['agent_counts']}")
    lines.append(f"live_groups: {SUMMARY_METADATA['live_groups']}")
    lines.append(
        f"prompt_history_turns: {FIXED_CONVERSATION_TURNS}"
    )
    lines.append(
        "impl_overhead_ms is fix-specific pre-LLM work "
        "(retrieval, embeddings, reranking, filtering). Baseline is 0.00ms."
    )
    lines.append(
        "first_call_ms is deterministic prompt work excluding fix-specific overhead; "
        "total_pre_llm_ms adds impl_overhead_ms. Both exclude network."
    )
    lines.append("")

    for group in enabled_groups:
        parts = []
        for agent_count in AGENT_COUNTS:
            result = LIVE_RESULTS[group].get(agent_count)
            if result is None:
                continue
            parts.append(
                f"{agent_count}={result['scores']} (mean={result['mean']:.2f})"
            )
        if parts:
            lines.append(f"{group}: " + ", ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reuse accuracy
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not is_live_group_enabled("reuse"),
    reason="OPENPOKE_BENCHMARK_LIVE_GROUPS excludes reuse",
)
@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
async def test_reuse_accuracy(
    agent_count: int,
    wired_env_live,
    data_dir,
    tool_recorder_live: ToolCallRecorder,
):
    """Does the LLM pick the right existing agent from a large roster?"""
    env = wired_env_live

    scores = []
    for trial in range(TRIALS):
        # Fresh roster each trial with different seed for position randomization
        env.roster.clear()
        env.conversation_log.clear()
        names = populate_roster(env.roster, agent_count, seed=42)

        # Pick target at randomized position
        rng = random.Random(trial * 1000 + agent_count)
        target_idx = rng.randint(0, agent_count - 1)
        target_name = names[target_idx]

        # Write minimal conversation log
        conv_path = data_dir / "conversation" / "poke_conversation.log"
        write_conversation_log(conv_path, 10, seed=trial)

        # Build a user message that should trigger reuse of the target
        # Extract the person/topic from the agent name for a natural query
        user_message = f"Follow up on the task handled by the agent called '{target_name}'"

        success, error, tool_calls = await _run_live_interaction(
            user_message,
            tool_recorder_live,
        )

        score = _score_reuse(tool_calls, target_name)
        scores.append(score)

        chosen = "none"
        agent_calls = [tc for tc in tool_calls if tc["name"] == "send_message_to_agent"]
        if agent_calls:
            chosen = agent_calls[0]["arguments"].get("agent_name", "?")

        print(
            f"\n  reuse trial {trial + 1}/{TRIALS} | agents={agent_count}, "
            f"target='{target_name}' (pos {target_idx}), "
            f"chosen='{chosen}', success={success}, "
            f"error={error or 'none'}, score={score}"
        )

    mean_score = sum(scores) / len(scores)
    _record_live_result("reuse", agent_count, scores, mean_score)
    print(
        f"\n  REUSE agents={agent_count:>5} | "
        f"scores={scores}, mean={mean_score:.2f}"
    )

    # No hard assertion on score — this is a measurement, not a pass/fail


# ---------------------------------------------------------------------------
# Creation accuracy
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not is_live_group_enabled("creation"),
    reason="OPENPOKE_BENCHMARK_LIVE_GROUPS excludes creation",
)
@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
async def test_creation_accuracy(
    agent_count: int,
    wired_env_live,
    data_dir,
    tool_recorder_live: ToolCallRecorder,
):
    """Does the LLM create a new agent when no existing one matches?"""
    env = wired_env_live

    scores = []
    # Topics that definitely don't match any generated agent name
    novel_topics = [
        "Book a flight to Mars for next Tuesday",
        "Translate this document from Klingon to Esperanto",
        "Order 500 rubber ducks for the office prank",
    ]

    for trial in range(TRIALS):
        env.roster.clear()
        env.conversation_log.clear()
        names = populate_roster(env.roster, agent_count, seed=42)

        conv_path = data_dir / "conversation" / "poke_conversation.log"
        write_conversation_log(conv_path, 10, seed=trial)

        user_message = novel_topics[trial]

        success, error, tool_calls = await _run_live_interaction(
            user_message,
            tool_recorder_live,
        )

        score = _score_creation(tool_calls, names)
        scores.append(score)

        chosen = "none"
        agent_calls = [tc for tc in tool_calls if tc["name"] == "send_message_to_agent"]
        if agent_calls:
            chosen = agent_calls[0]["arguments"].get("agent_name", "?")

        print(
            f"\n  creation trial {trial + 1}/{TRIALS} | agents={agent_count}, "
            f"topic='{user_message[:50]}...', "
            f"chosen='{chosen}', in_roster={chosen in names}, "
            f"success={success}, error={error or 'none'}, score={score}"
        )

    mean_score = sum(scores) / len(scores)
    _record_live_result("creation", agent_count, scores, mean_score)
    print(
        f"\n  CREATION agents={agent_count:>5} | "
        f"scores={scores}, mean={mean_score:.2f}"
    )


# ---------------------------------------------------------------------------
# Combined summary (runs all live accuracy tests and prints a table)
# ---------------------------------------------------------------------------

@pytest.mark.live
async def test_accuracy_summary(wired_env_live, data_dir):
    """Print a summary table after all live accuracy measurements."""
    env = wired_env_live
    for agent_count in AGENT_COUNTS:
        stats = measure_prompt_stats(env, data_dir, agent_count)
        _record_prompt_result(agent_count, stats)

    SUMMARY_METADATA.clear()
    SUMMARY_METADATA.update(
        {
            "model": env.settings.interaction_agent_model,
            "implementation": get_benchmark_implementation_name(),
            "trials": str(TRIALS),
            "agent_counts": ",".join(str(count) for count in AGENT_COUNTS),
            "live_groups": ",".join(
                group for group in ("reuse", "creation")
                if is_live_group_enabled(group)
            ),
        }
    )

    print(
        "\n\n"
        "Accuracy Summary\n"
        f"{_format_accuracy_summary_table()}\n"
    )
