"""V3 Benchmark: Live accuracy tests against the real interaction runtime.

Measures whether the interaction loop can still reuse or create the right
agent and whether it notifies the user before delegating as the roster grows.
Marked with @pytest.mark.live — skipped by default, run with: pytest -m live
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import pytest

from server.agents.interaction_agent.runtime import InteractionAgentRuntime

from .conftest import ToolCallRecorder
from .factories import populate_roster, write_conversation_log


AGENT_COUNTS = [5, 25, 100, 500, 1000]
TRIALS = 3


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


def _score_instruction_order(tool_calls: List[Dict]) -> float:
    """Score whether send_message_to_user comes before send_message_to_agent.

    1.0 = correct order, 0.0 = missing delegation or missing/wrong notification order
    """
    user_idx = None
    agent_idx = None

    for i, tc in enumerate(tool_calls):
        if tc["name"] == "send_message_to_user" and user_idx is None:
            user_idx = i
        if tc["name"] == "send_message_to_agent" and agent_idx is None:
            agent_idx = i

    if agent_idx is None:
        return 0.0

    if user_idx is not None and user_idx < agent_idx:
        return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# Reuse accuracy
# ---------------------------------------------------------------------------

@pytest.mark.live
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
    print(
        f"\n  REUSE agents={agent_count:>5} | "
        f"scores={scores}, mean={mean_score:.2f}"
    )

    # No hard assertion on score — this is a measurement, not a pass/fail


# ---------------------------------------------------------------------------
# Creation accuracy
# ---------------------------------------------------------------------------

@pytest.mark.live
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
    print(
        f"\n  CREATION agents={agent_count:>5} | "
        f"scores={scores}, mean={mean_score:.2f}"
    )


# ---------------------------------------------------------------------------
# Instruction order accuracy
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
async def test_instruction_order_accuracy(
    agent_count: int,
    wired_env_live,
    data_dir,
    tool_recorder_live: ToolCallRecorder,
):
    """Does the LLM call send_message_to_user before send_message_to_agent?"""
    env = wired_env_live

    scores = []
    for trial in range(TRIALS):
        env.roster.clear()
        env.conversation_log.clear()
        populate_roster(env.roster, agent_count, seed=42)

        conv_path = data_dir / "conversation" / "poke_conversation.log"
        write_conversation_log(conv_path, 10, seed=trial)

        user_message = "Draft an email to someone about the quarterly review"

        success, error, tool_calls = await _run_live_interaction(
            user_message,
            tool_recorder_live,
        )

        score = _score_instruction_order(tool_calls)
        scores.append(score)

        tool_names = [tc["name"] for tc in tool_calls]
        print(
            f"\n  order trial {trial + 1}/{TRIALS} | agents={agent_count}, "
            f"tools={tool_names}, success={success}, "
            f"error={error or 'none'}, score={score}"
        )

    mean_score = sum(scores) / len(scores)
    print(
        f"\n  ORDER agents={agent_count:>5} | "
        f"scores={scores}, mean={mean_score:.2f}"
    )


# ---------------------------------------------------------------------------
# Combined summary (runs all 3 tests and prints a table)
# ---------------------------------------------------------------------------

@pytest.mark.live
async def test_accuracy_summary(wired_env_live, data_dir, capsys):
    """Print a summary table header. Run after all parametrized tests."""
    # This test just prints the header for readability when running the full suite
    print(
        "\n\n"
        "Accuracy Summary\n"
        "Run individual tests above for per-scale results.\n"
        "Expected output format:\n"
        "  REUSE   agents=N | scores=[...], mean=X.XX\n"
        "  CREATE  agents=N | scores=[...], mean=X.XX\n"
        "  ORDER   agents=N | scores=[...], mean=X.XX\n"
    )
