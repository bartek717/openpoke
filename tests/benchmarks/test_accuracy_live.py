"""V3 Benchmark: Live accuracy tests with real OpenRouter API calls.

Measures whether the LLM can correctly reuse/create agents and follow
instruction order at scale. Marked with @pytest.mark.live — skipped by
default, run with: pytest -m live

Each test runs 3 trials per agent_count with the target agent at a
randomized roster position. Scoring is derived from the actual tool_calls
in the API response.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import pytest

from server.agents.interaction_agent.agent import build_system_prompt, prepare_message_with_history
from server.agents.interaction_agent.tools import get_tool_schemas
from server.openrouter_client import request_chat_completion

from .factories import generate_agent_names, populate_roster, write_conversation_log


AGENT_COUNTS = [5, 25, 100, 500, 1000]
TRIALS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract parsed tool calls from an OpenRouter response."""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message", {})
    raw = message.get("tool_calls") or []

    parsed = []
    for tc in raw:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_raw = func.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = args_raw or {}
        parsed.append({"name": name, "arguments": args})
    return parsed


def _score_reuse(tool_calls: List[Dict], target_name: str) -> float:
    """Score agent reuse accuracy.

    1.0 = exact match, 0.5 = partial match, 0.0 = wrong/new agent
    """
    agent_calls = [tc for tc in tool_calls if tc["name"] == "send_message_to_agent"]
    if not agent_calls:
        return 0.0

    chosen_name = agent_calls[0]["arguments"].get("agent_name", "")

    if chosen_name == target_name:
        return 1.0

    # Partial match: check if key words overlap
    target_words = set(target_name.lower().split())
    chosen_words = set(chosen_name.lower().split())
    overlap = target_words & chosen_words
    # Need at least 2 meaningful words in common (exclude short words)
    meaningful_overlap = {w for w in overlap if len(w) > 2}
    if len(meaningful_overlap) >= 2:
        return 0.5

    return 0.0


def _score_creation(tool_calls: List[Dict], roster_names: List[str]) -> float:
    """Score agent creation accuracy.

    1.0 = created new agent not in roster, 0.0 = reused existing
    """
    agent_calls = [tc for tc in tool_calls if tc["name"] == "send_message_to_agent"]
    if not agent_calls:
        return 0.0  # didn't delegate at all

    chosen_name = agent_calls[0]["arguments"].get("agent_name", "")
    roster_set = set(roster_names)

    if chosen_name not in roster_set:
        return 1.0
    return 0.0


def _score_instruction_order(tool_calls: List[Dict]) -> float:
    """Score whether send_message_to_user comes before send_message_to_agent.

    1.0 = correct order, 0.0 = wrong order or missing user notification
    """
    user_idx = None
    agent_idx = None

    for i, tc in enumerate(tool_calls):
        if tc["name"] == "send_message_to_user" and user_idx is None:
            user_idx = i
        if tc["name"] == "send_message_to_agent" and agent_idx is None:
            agent_idx = i

    if agent_idx is None:
        # No delegation — instruction order not applicable, count as pass
        return 1.0

    if user_idx is not None and user_idx < agent_idx:
        return 1.0

    return 0.0


async def _run_llm_loop(
    user_message: str,
    transcript: str,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4",
    max_turns: int = 3,
) -> List[Dict[str, Any]]:
    """Run up to max_turns of the interaction agent loop, collecting all tool calls.

    The LLM may split send_message_to_user and send_message_to_agent across
    turns (notify user first, delegate second). This loop simulates that by
    feeding synthetic tool results back for each turn.

    Returns the aggregated list of tool calls across all turns.
    """
    system_prompt = build_system_prompt()
    messages = prepare_message_with_history(user_message, transcript)
    tools = get_tool_schemas()
    all_tool_calls: List[Dict[str, Any]] = []

    for turn in range(max_turns):
        response = await request_chat_completion(
            model=model,
            messages=messages,
            system=system_prompt,
            api_key=api_key,
            tools=tools,
        )

        choice = (response.get("choices") or [{}])[0]
        assistant_message = choice.get("message", {})

        # Append assistant message to conversation
        assistant_entry: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message.get("content", "") or "",
        }
        raw_tool_calls = assistant_message.get("tool_calls") or []
        if raw_tool_calls:
            assistant_entry["tool_calls"] = raw_tool_calls
        messages.append(assistant_entry)

        # Parse tool calls from this turn
        turn_calls = _extract_tool_calls(response)
        all_tool_calls.extend(turn_calls)

        # If no tool calls, the LLM is done
        if not turn_calls:
            break

        # Feed synthetic tool results back so the loop can continue
        for tc in raw_tool_calls:
            tool_id = tc.get("id", "unknown")
            func = tc.get("function", {})
            name = func.get("name", "")

            if name == "send_message_to_user":
                result = json.dumps({"tool": name, "status": "success", "result": {"status": "delivered"}})
            elif name == "send_message_to_agent":
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                agent_name = args.get("agent_name", "unknown")
                result = json.dumps({
                    "tool": name, "status": "success",
                    "result": {"status": "submitted", "agent_name": agent_name, "new_agent_created": True},
                })
            elif name == "send_draft":
                result = json.dumps({"tool": name, "status": "success", "result": {"status": "draft_recorded"}})
            elif name == "wait":
                result = json.dumps({"tool": name, "status": "success", "result": {"status": "waiting"}})
            else:
                result = json.dumps({"tool": name, "status": "success"})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
            })

    return all_tool_calls


# ---------------------------------------------------------------------------
# Reuse accuracy
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
async def test_reuse_accuracy(
    agent_count: int,
    wired_env_live,
    data_dir,
):
    """Does the LLM pick the right existing agent from a large roster?"""
    env = wired_env_live

    scores = []
    for trial in range(TRIALS):
        # Fresh roster each trial with different seed for position randomization
        env.roster.clear()
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

        transcript = env.conversation_log.load_transcript()
        tool_calls = await _run_llm_loop(
            user_message, transcript, env.settings.openrouter_api_key
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
            f"chosen='{chosen}', score={score}"
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
        names = populate_roster(env.roster, agent_count, seed=42)

        conv_path = data_dir / "conversation" / "poke_conversation.log"
        write_conversation_log(conv_path, 10, seed=trial)

        user_message = novel_topics[trial]
        transcript = env.conversation_log.load_transcript()

        tool_calls = await _run_llm_loop(
            user_message, transcript, env.settings.openrouter_api_key
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
            f"chosen='{chosen}', in_roster={chosen in names}, score={score}"
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
):
    """Does the LLM call send_message_to_user before send_message_to_agent?"""
    env = wired_env_live

    scores = []
    for trial in range(TRIALS):
        env.roster.clear()
        names = populate_roster(env.roster, agent_count, seed=42)

        conv_path = data_dir / "conversation" / "poke_conversation.log"
        write_conversation_log(conv_path, 10, seed=trial)

        user_message = "Draft an email to someone about the quarterly review"
        transcript = env.conversation_log.load_transcript()

        tool_calls = await _run_llm_loop(
            user_message, transcript, env.settings.openrouter_api_key
        )

        score = _score_instruction_order(tool_calls)
        scores.append(score)

        tool_names = [tc["name"] for tc in tool_calls]
        print(
            f"\n  order trial {trial + 1}/{TRIALS} | agents={agent_count}, "
            f"tools={tool_names}, score={score}"
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
