"""V2 Benchmark: Batch manager coordination under concurrent agent load.

Uses barrier-based mocking for deterministic timing. Patches
_dispatch_to_interaction_agent with a collector to verify single-batch dispatch.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import pytest

from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.agents.execution_agent.runtime import ExecutionResult


CONCURRENT_AGENTS = [10, 50, 100]


class DispatchCollector:
    """Captures dispatch payloads instead of creating a new InteractionAgentRuntime."""

    def __init__(self):
        self.payloads: List[str] = []

    async def __call__(self, payload: str) -> None:
        self.payloads.append(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_agents", CONCURRENT_AGENTS)
async def test_batch_single_dispatch(
    concurrent_agents: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """All N agents must complete in exactly one batch dispatch."""
    manager = ExecutionBatchManager(timeout_seconds=10)
    collector = DispatchCollector()

    # Patch _dispatch_to_interaction_agent to capture instead of creating runtime
    monkeypatch.setattr(
        manager,
        "_dispatch_to_interaction_agent",
        collector,
    )

    # Use a barrier so all tasks register before any completes
    barrier = asyncio.Barrier(concurrent_agents)

    async def mock_execute(agent_name: str, instructions: str) -> ExecutionResult:
        return ExecutionResult(
            agent_name=agent_name,
            success=True,
            response=f"Completed: {agent_name}",
        )

    # Patch ExecutionAgentRuntime to use our mock
    monkeypatch.setattr(
        "server.agents.execution_agent.batch_manager.ExecutionAgentRuntime",
        type("MockRuntime", (), {
            "__init__": lambda self, agent_name: setattr(self, "name", agent_name),
            "execute": lambda self, instructions: _barrier_execute(barrier, self.name, instructions),
        }),
    )

    agent_names = [f"Agent-{i}" for i in range(concurrent_agents)]

    t0 = time.perf_counter()
    tasks = [
        asyncio.create_task(
            manager.execute_agent(name, f"Task for {name}")
        )
        for name in agent_names
    ]
    results = await asyncio.gather(*tasks)
    wall_ms = (time.perf_counter() - t0) * 1000

    # --- Assertions ---

    # All agents completed successfully
    assert all(r.success for r in results)
    assert len(results) == concurrent_agents

    # Exactly one batch dispatch
    assert len(collector.payloads) == 1, (
        f"Expected 1 dispatch, got {len(collector.payloads)}"
    )

    # Dispatch payload contains all agent names
    payload = collector.payloads[0]
    for name in agent_names:
        assert name in payload, f"Agent '{name}' missing from dispatch payload"

    print(
        f"\n  concurrent={concurrent_agents:>5} | wall={wall_ms:.1f}ms, "
        f"dispatches={len(collector.payloads)}, "
        f"all_present=yes, "
        f"payload_size={len(payload)}B"
    )


async def _barrier_execute(barrier, agent_name, instructions):
    """Wait for all agents to register, then complete."""
    await barrier.wait()
    return ExecutionResult(
        agent_name=agent_name,
        success=True,
        response=f"Completed: {agent_name}",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_agents", CONCURRENT_AGENTS)
async def test_batch_result_ordering(
    concurrent_agents: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Results should all be collected regardless of completion order."""
    manager = ExecutionBatchManager(timeout_seconds=10)
    collector = DispatchCollector()

    monkeypatch.setattr(
        manager,
        "_dispatch_to_interaction_agent",
        collector,
    )

    # Agents complete with varying delays to test ordering
    async def staggered_execute(instructions):
        import random
        delay = random.uniform(0.001, 0.01)
        await asyncio.sleep(delay)
        return ExecutionResult(
            agent_name=self_name,
            success=True,
            response=f"Done after {delay:.3f}s",
        )

    class StaggeredRuntime:
        def __init__(self, agent_name):
            self._name = agent_name

        async def execute(self, instructions):
            delay = (hash(self._name) % 10) * 0.001
            await asyncio.sleep(delay)
            return ExecutionResult(
                agent_name=self._name,
                success=True,
                response=f"Completed: {self._name}",
            )

    monkeypatch.setattr(
        "server.agents.execution_agent.batch_manager.ExecutionAgentRuntime",
        StaggeredRuntime,
    )

    agent_names = [f"Staggered-{i}" for i in range(concurrent_agents)]

    t0 = time.perf_counter()
    tasks = [
        asyncio.create_task(
            manager.execute_agent(name, f"Task for {name}")
        )
        for name in agent_names
    ]
    results = await asyncio.gather(*tasks)
    wall_ms = (time.perf_counter() - t0) * 1000

    assert all(r.success for r in results)
    assert len(collector.payloads) == 1

    # All agents present in dispatch regardless of completion order
    payload = collector.payloads[0]
    for name in agent_names:
        assert name in payload

    # Count SUCCESS entries
    success_count = payload.count("[SUCCESS]")
    assert success_count == concurrent_agents

    print(
        f"\n  concurrent={concurrent_agents:>5} | wall={wall_ms:.1f}ms, "
        f"dispatches=1, "
        f"success_count={success_count}, "
        f"all_present=yes"
    )
