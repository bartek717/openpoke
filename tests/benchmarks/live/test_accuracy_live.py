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
    is_live_group_enabled,
)
from ..support.factories import populate_roster, write_conversation_log


TRIALS = LIVE_TRIALS
LIVE_RESULTS: Dict[str, Dict[int, Dict[str, Any]]] = {
    "reuse": {},
    "creation": {},
}


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


def _format_accuracy_summary_table() -> str:
    enabled_groups = [
        group for group in ("reuse", "creation") if is_live_group_enabled(group)
    ]
    if not enabled_groups:
        return "No live groups enabled."

    headers = ["agents", *enabled_groups]
    align = ["-----:", *["-----:" for _ in enabled_groups]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]

    for agent_count in AGENT_COUNTS:
        row = [str(agent_count)]
        for group in enabled_groups:
            result = LIVE_RESULTS[group].get(agent_count)
            row.append(f"{result['mean']:.2f}" if result is not None else "-")
        lines.append("| " + " | ".join(row) + " |")

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
async def test_accuracy_summary():
    """Print a summary table after all live accuracy measurements."""
    print(
        "\n\n"
        "Accuracy Summary\n"
        f"{_format_accuracy_summary_table()}\n"
    )
