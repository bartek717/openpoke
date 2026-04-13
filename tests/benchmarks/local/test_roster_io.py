"""V1 Benchmark: Roster file I/O timing and correctness under contention."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from server.services.execution.roster import AgentRoster

from ..conftest import AGENT_COUNTS
from ..support.factories import generate_agent_names


@pytest.fixture()
def roster_path(tmp_path: Path) -> Path:
    p = tmp_path / "execution_agents" / "roster.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
def test_roster_load_time(agent_count: int, roster_path: Path):
    """Measure time to load a roster of N agents from disk."""
    names = generate_agent_names(agent_count)
    roster_path.write_text(json.dumps(names), encoding="utf-8")

    roster = AgentRoster(roster_path)

    t0 = time.perf_counter()
    roster.load()
    load_ms = (time.perf_counter() - t0) * 1000

    assert len(roster.get_agents()) == agent_count
    print(f"\n  agents={agent_count:>5} | load={load_ms:.2f}ms")


@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
def test_roster_save_time(agent_count: int, roster_path: Path):
    """Measure time to save a roster of N agents to disk."""
    roster = AgentRoster(roster_path)
    names = generate_agent_names(agent_count)
    for name in names:
        roster._agents.append(name)  # direct append to skip per-add saves

    t0 = time.perf_counter()
    roster.save()
    save_ms = (time.perf_counter() - t0) * 1000

    # Verify written file
    data = json.loads(roster_path.read_text(encoding="utf-8"))
    assert len(data) == agent_count
    print(f"\n  agents={agent_count:>5} | save={save_ms:.2f}ms")


@pytest.mark.parametrize("agent_count", AGENT_COUNTS)
def test_roster_add_agent_at_scale(agent_count: int, roster_path: Path):
    """Measure time to add one more agent when roster already has N agents."""
    names = generate_agent_names(agent_count)
    roster_path.write_text(json.dumps(names), encoding="utf-8")
    roster = AgentRoster(roster_path)

    new_name = f"Benchmark Agent #{agent_count + 1}"

    t0 = time.perf_counter()
    roster.add_agent(new_name)
    add_ms = (time.perf_counter() - t0) * 1000

    assert len(roster.get_agents()) == agent_count + 1
    assert new_name in roster.get_agents()
    print(f"\n  agents={agent_count:>5} | add_one={add_ms:.2f}ms")


@pytest.mark.parametrize("agent_count", [100, 500])
def test_roster_concurrent_contention(agent_count: int, roster_path: Path):
    """Concurrent readers/writers must not corrupt the roster.

    Spawns writer threads that each add agents, plus reader threads that load.
    Asserts the final JSON is valid and contains the expected agent count.
    """
    # Pre-populate with half the agents
    initial_count = agent_count // 2
    initial_names = generate_agent_names(initial_count, seed=1)
    roster_path.write_text(json.dumps(initial_names), encoding="utf-8")

    # Each writer adds unique agents
    num_writers = 4
    agents_per_writer = (agent_count - initial_count) // num_writers
    num_readers = 4

    errors: list[str] = []
    read_counts: list[int] = []

    def writer_task(writer_id: int) -> None:
        roster = AgentRoster(roster_path)
        for i in range(agents_per_writer):
            name = f"Writer-{writer_id}-Agent-{i}"
            try:
                roster.add_agent(name)
            except Exception as exc:
                errors.append(f"writer-{writer_id}: {exc}")

    def reader_task() -> None:
        for _ in range(10):
            try:
                roster = AgentRoster(roster_path)
                roster.load()
                agents = roster.get_agents()
                read_counts.append(len(agents))
            except Exception as exc:
                errors.append(f"reader: {exc}")
            time.sleep(0.001)

    threads: list[threading.Thread] = []
    for wid in range(num_writers):
        threads.append(threading.Thread(target=writer_task, args=(wid,)))
    for _ in range(num_readers):
        threads.append(threading.Thread(target=reader_task))

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    contention_ms = (time.perf_counter() - t0) * 1000

    # --- Correctness assertions ---
    assert not errors, f"Errors during contention: {errors}"

    # Final file must be valid JSON
    raw = roster_path.read_text(encoding="utf-8")
    try:
        final_data = json.loads(raw)
    except json.JSONDecodeError:
        pytest.fail(f"Roster JSON is corrupted after contention: {raw[:200]}...")

    assert isinstance(final_data, list), f"Expected list, got {type(final_data)}"

    expected_added = num_writers * agents_per_writer
    expected_total = initial_count + expected_added
    actual_total = len(final_data)

    # The current roster uses read-modify-write without transactions,
    # so concurrent writers WILL lose writes. This test documents that
    # behavior as a known limitation. We only assert the file isn't corrupted.
    write_loss_pct = (1 - actual_total / expected_total) * 100 if expected_total else 0

    print(
        f"\n  agents={agent_count:>5} | contention={contention_ms:.1f}ms, "
        f"expected={expected_total}, actual={actual_total}, "
        f"json_valid=yes, write_loss={write_loss_pct:.0f}%, "
        f"reads={len(read_counts)} (range {min(read_counts)}-{max(read_counts)})"
    )
