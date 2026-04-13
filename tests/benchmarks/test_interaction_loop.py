"""V2 Benchmark: Interaction agent loop with mocked LLM.

Measures end-to-end execute() timing, records tool invocation sequences,
and captures per-LLM-call payload sizes at different roster scales.
"""

from __future__ import annotations

import time

import psutil
import pytest

from server.agents.interaction_agent.runtime import InteractionAgentRuntime

from .conftest import ToolCallRecorder
from .factories import populate_roster, write_conversation_log
from .mock_llm import (
    MockOpenRouterResponder,
    scenario_dispatch_one,
    scenario_fan_out,
    scenario_noop,
    scenario_reuse_agent,
)


AGENT_COUNTS = [5, 25, 100, 500, 1000, 2000]


def _format_tool_sequence(recorder: ToolCallRecorder) -> str:
    """Format recorded tool calls as a readable chain."""
    names = [inv.name for inv in recorder.invocations]
    short = [n.replace("send_message_to_", "send_") for n in names]
    return " -> ".join(short) + " -> done" if short else "none"


# ---------------------------------------------------------------------------
# dispatch_one: send_message_to_user + send_message_to_agent -> final
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
async def test_loop_dispatch_one(
    agent_count: int,
    monkeypatch: pytest.MonkeyPatch,
    wired_env,
    data_dir,
):
    """Full execute() with one user message + one agent dispatch."""
    env = wired_env

    # Populate
    populate_roster(env.roster, agent_count)
    conv_path = data_dir / "conversation" / "poke_conversation.log"
    write_conversation_log(conv_path, 50)

    # Set up scenario-driven mock
    scenario = scenario_dispatch_one("Benchmark Agent")
    mock = MockOpenRouterResponder(scenario)
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.request_chat_completion", mock
    )

    # Set up tool recorder
    from server.agents.interaction_agent.tools import handle_tool_call
    from .conftest import ToolCallRecorder
    recorder = ToolCallRecorder(handle_tool_call)
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.handle_tool_call", recorder
    )

    # Measure
    process = psutil.Process()
    mem_before = process.memory_info().rss

    t0 = time.perf_counter()
    runtime = InteractionAgentRuntime()
    result = await runtime.execute("Draft an email to Keith about the meeting")
    wall_ms = (time.perf_counter() - t0) * 1000

    mem_after = process.memory_info().rss
    mem_delta_mb = (mem_after - mem_before) / 1024 / 1024

    # Results
    tool_seq = _format_tool_sequence(recorder)
    call1_kb = mock.calls[0].full_payload_bytes / 1024 if mock.calls else 0

    print(
        f"\n  agents={agent_count:>5} | wall={wall_ms:.1f}ms, "
        f"llm_calls={len(mock.calls)}, "
        f"tools=[{tool_seq}], "
        f"call1={call1_kb:.1f}KB, "
        f"mem_delta={mem_delta_mb:+.1f}MB"
    )

    # Assertions
    assert result.success
    assert len(mock.calls) == 2  # tool call turn + final response
    assert len(recorder.invocations) == 2  # send_user + send_agent
    assert recorder.invocations[0].name == "send_message_to_user"
    assert recorder.invocations[1].name == "send_message_to_agent"

    # Per-call payload should scale with roster size
    for i, call in enumerate(mock.calls):
        print(
            f"    call[{i}]: payload={call.full_payload_bytes / 1024:.1f}KB, "
            f"system={call.system_length}B, "
            f"msgs_content={call.messages_content_length}B"
        )


# ---------------------------------------------------------------------------
# fan_out_3: dispatch to 3 agents simultaneously
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
async def test_loop_fan_out(
    agent_count: int,
    monkeypatch: pytest.MonkeyPatch,
    wired_env,
    data_dir,
):
    """Full execute() dispatching to 3 agents in one turn."""
    env = wired_env

    populate_roster(env.roster, agent_count)
    conv_path = data_dir / "conversation" / "poke_conversation.log"
    write_conversation_log(conv_path, 50)

    fan_out_names = ["Task Agent A", "Task Agent B", "Task Agent C"]
    scenario = scenario_fan_out(fan_out_names)
    mock = MockOpenRouterResponder(scenario)
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.request_chat_completion", mock
    )

    from server.agents.interaction_agent.tools import handle_tool_call
    recorder = ToolCallRecorder(handle_tool_call)
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.handle_tool_call", recorder
    )

    t0 = time.perf_counter()
    runtime = InteractionAgentRuntime()
    result = await runtime.execute("Do tasks A, B, and C at the same time")
    wall_ms = (time.perf_counter() - t0) * 1000

    tool_seq = _format_tool_sequence(recorder)
    call1_kb = mock.calls[0].full_payload_bytes / 1024 if mock.calls else 0

    print(
        f"\n  agents={agent_count:>5} | wall={wall_ms:.1f}ms, "
        f"llm_calls={len(mock.calls)}, "
        f"tools=[{tool_seq}], "
        f"call1={call1_kb:.1f}KB"
    )

    assert result.success
    assert len(mock.calls) == 2
    # 1 send_user + 3 send_agent = 4 tool invocations
    assert len(recorder.invocations) == 4
    assert recorder.invocations[0].name == "send_message_to_user"
    agent_calls = [inv for inv in recorder.invocations if inv.name == "send_message_to_agent"]
    assert len(agent_calls) == 3


# ---------------------------------------------------------------------------
# noop: wait tool, no agent dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
async def test_loop_noop(
    agent_count: int,
    monkeypatch: pytest.MonkeyPatch,
    wired_env,
    data_dir,
):
    """Full execute() where the LLM calls wait (no-op)."""
    env = wired_env

    populate_roster(env.roster, agent_count)
    conv_path = data_dir / "conversation" / "poke_conversation.log"
    write_conversation_log(conv_path, 50)

    scenario = scenario_noop()
    mock = MockOpenRouterResponder(scenario)
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.request_chat_completion", mock
    )

    from server.agents.interaction_agent.tools import handle_tool_call
    recorder = ToolCallRecorder(handle_tool_call)
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.handle_tool_call", recorder
    )

    t0 = time.perf_counter()
    runtime = InteractionAgentRuntime()
    result = await runtime.execute("Just checking in")
    wall_ms = (time.perf_counter() - t0) * 1000

    call1_kb = mock.calls[0].full_payload_bytes / 1024 if mock.calls else 0

    print(
        f"\n  agents={agent_count:>5} | wall={wall_ms:.1f}ms, "
        f"llm_calls={len(mock.calls)}, "
        f"tools=[wait -> done], "
        f"call1={call1_kb:.1f}KB"
    )

    assert result.success
    assert len(mock.calls) == 2
    assert len(recorder.invocations) == 1
    assert recorder.invocations[0].name == "wait"


# ---------------------------------------------------------------------------
# reuse_agent: LLM picks an existing agent from the roster
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
async def test_loop_reuse_agent(
    agent_count: int,
    monkeypatch: pytest.MonkeyPatch,
    wired_env,
    data_dir,
):
    """Full execute() where the LLM reuses an existing agent by name."""
    env = wired_env

    names = populate_roster(env.roster, agent_count)
    conv_path = data_dir / "conversation" / "poke_conversation.log"
    write_conversation_log(conv_path, 50)

    # Pick an agent from the middle of the roster to reuse
    target_agent = names[agent_count // 2]

    scenario = scenario_reuse_agent(target_agent)
    mock = MockOpenRouterResponder(scenario)
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.request_chat_completion", mock
    )

    from server.agents.interaction_agent.tools import handle_tool_call
    recorder = ToolCallRecorder(handle_tool_call)
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.handle_tool_call", recorder
    )

    t0 = time.perf_counter()
    runtime = InteractionAgentRuntime()
    result = await runtime.execute("Follow up on the previous task")
    wall_ms = (time.perf_counter() - t0) * 1000

    print(
        f"\n  agents={agent_count:>5} | wall={wall_ms:.1f}ms, "
        f"reused='{target_agent}', "
        f"llm_calls={len(mock.calls)}"
    )

    assert result.success
    # The send_message_to_agent should reference an existing agent (not create new)
    agent_call = next(
        (inv for inv in recorder.invocations if inv.name == "send_message_to_agent"),
        None,
    )
    assert agent_call is not None
    assert agent_call.result.payload["new_agent_created"] is False
