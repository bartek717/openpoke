"""Benchmark fixtures — replaces all singletons with temp-backed instances."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from server.config import Settings, get_settings
from server.services.execution.roster import AgentRoster
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.conversation.log import ConversationLog
from server.services.conversation.summarization.working_memory_log import WorkingMemoryLog
from server.agents.interaction_agent.tools import ToolResult

from .support.mock_llm import MockOpenRouterResponder


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

DEFAULT_AGENT_COUNTS = [5, 25, 100, 500, 1000, 2000]
DEFAULT_LIVE_TRIALS = 3
DEFAULT_LIVE_GROUPS = ("reuse", "creation")
CONVERSATION_TURNS = [50, 200, 500]


def _parse_agent_counts(value: str | None) -> list[int]:
    if not value:
        return DEFAULT_AGENT_COUNTS.copy()

    counts: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            count = int(part)
        except ValueError as exc:
            raise ValueError(
                "OPENPOKE_BENCHMARK_AGENT_COUNTS must be a comma-separated list "
                f"of integers, got {value!r}"
            ) from exc
        if count <= 0:
            raise ValueError(
                "OPENPOKE_BENCHMARK_AGENT_COUNTS values must be positive, "
                f"got {count}"
            )
        counts.append(count)

    if not counts:
        raise ValueError(
            "OPENPOKE_BENCHMARK_AGENT_COUNTS must include at least one agent count"
        )

    return counts


def _parse_live_trials(value: str | None) -> int:
    if not value:
        return DEFAULT_LIVE_TRIALS

    try:
        trials = int(value)
    except ValueError as exc:
        raise ValueError(
            f"OPENPOKE_BENCHMARK_TRIALS must be an integer, got {value!r}"
        ) from exc

    if trials <= 0:
        raise ValueError(
            f"OPENPOKE_BENCHMARK_TRIALS must be positive, got {trials}"
        )

    return trials


def _parse_live_groups(value: str | None) -> set[str]:
    if not value:
        return set(DEFAULT_LIVE_GROUPS)

    allowed = {"all", "reuse", "creation"}
    aliases = {"create": "creation"}
    groups: set[str] = set()

    for raw_part in value.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        normalized = aliases.get(part, part)
        if normalized not in allowed:
            raise ValueError(
                "OPENPOKE_BENCHMARK_LIVE_GROUPS must contain only "
                f"{sorted(allowed)}, got {value!r}"
            )
        groups.add(normalized)

    if not groups:
        raise ValueError(
            "OPENPOKE_BENCHMARK_LIVE_GROUPS must include at least one group"
        )

    if "all" in groups:
        return {"reuse", "creation"}

    return groups


AGENT_COUNTS = _parse_agent_counts(os.getenv("OPENPOKE_BENCHMARK_AGENT_COUNTS"))
LIVE_TRIALS = _parse_live_trials(os.getenv("OPENPOKE_BENCHMARK_TRIALS"))
LIVE_GROUPS = _parse_live_groups(os.getenv("OPENPOKE_BENCHMARK_LIVE_GROUPS"))


def is_live_group_enabled(group: str) -> bool:
    return group in LIVE_GROUPS


# ---------------------------------------------------------------------------
# Core temp-backed singletons
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Root temp data directory for a single test."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture()
def temp_roster(data_dir: Path) -> AgentRoster:
    roster_path = data_dir / "execution_agents" / "roster.json"
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    return AgentRoster(roster_path)


@pytest.fixture()
def temp_conversation_log(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    temp_working_memory: WorkingMemoryLog,
) -> ConversationLog:
    log_path = data_dir / "conversation" / "poke_conversation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "server.services.conversation.log._resolve_working_memory_log",
        lambda: temp_working_memory,
    )
    return ConversationLog(log_path)


@pytest.fixture()
def temp_working_memory(data_dir: Path) -> WorkingMemoryLog:
    wm_path = data_dir / "conversation" / "poke_working_memory.log"
    wm_path.parent.mkdir(parents=True, exist_ok=True)
    return WorkingMemoryLog(wm_path)


@pytest.fixture()
def temp_exec_logs(data_dir: Path) -> ExecutionAgentLogStore:
    exec_dir = data_dir / "execution_agents"
    exec_dir.mkdir(parents=True, exist_ok=True)
    return ExecutionAgentLogStore(exec_dir)


@pytest.fixture()
def fake_settings() -> Settings:
    get_settings.cache_clear()
    benchmark_model = os.getenv("OPENPOKE_BENCHMARK_MODEL", "anthropic/claude-sonnet-4")
    return Settings(
        openrouter_api_key="fake-benchmark-key",
        conversation_summary_threshold=0,  # disables summarization
        interaction_agent_model=benchmark_model,
        execution_agent_model=benchmark_model,
    )


@pytest.fixture()
def mock_llm() -> MockOpenRouterResponder:
    return MockOpenRouterResponder()


# ---------------------------------------------------------------------------
# Wired runtime — patches every import site
# ---------------------------------------------------------------------------

def _patch_settings_aliases(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Patch cached settings getters at every import site used by benchmarks."""

    get_settings.cache_clear()
    getter = lambda: settings
    monkeypatch.setattr("server.config.get_settings", getter)
    monkeypatch.setattr("server.agents.interaction_agent.runtime.get_settings", getter)
    monkeypatch.setattr("server.agents.execution_agent.runtime.get_settings", getter)
    monkeypatch.setattr("server.services.conversation.log.get_settings", getter)
    monkeypatch.setattr("server.services.conversation.summarization.summarizer.get_settings", getter)
    monkeypatch.setattr("server.openrouter_client.client.get_settings", getter)


def _wire_shared_singletons(
    monkeypatch: pytest.MonkeyPatch,
    *,
    temp_roster: AgentRoster,
    temp_conversation_log: ConversationLog,
    temp_working_memory: WorkingMemoryLog,
    temp_exec_logs: ExecutionAgentLogStore,
) -> None:
    """Patch production singletons and getters to temp-backed benchmark instances."""

    monkeypatch.setattr(
        "server.services.execution.roster._agent_roster", temp_roster
    )
    _roster_getter = lambda: temp_roster
    monkeypatch.setattr(
        "server.services.execution.roster.get_agent_roster", _roster_getter
    )
    monkeypatch.setattr(
        "server.services.execution.get_agent_roster", _roster_getter
    )
    monkeypatch.setattr(
        "server.services.get_agent_roster", _roster_getter
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.agent.get_agent_roster", _roster_getter
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.tools.get_agent_roster", _roster_getter
    )

    monkeypatch.setattr(
        "server.services.conversation.log._conversation_log", temp_conversation_log
    )
    _conv_getter = lambda: temp_conversation_log
    monkeypatch.setattr(
        "server.services.conversation.log.get_conversation_log", _conv_getter
    )
    monkeypatch.setattr(
        "server.services.conversation.get_conversation_log", _conv_getter
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.tools.get_conversation_log", _conv_getter
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.get_conversation_log", _conv_getter
    )
    monkeypatch.setattr(
        "server.services.conversation.summarization.summarizer._resolve_conversation_log",
        _conv_getter,
    )

    monkeypatch.setattr(
        "server.services.conversation.summarization.working_memory_log._working_memory_log",
        temp_working_memory,
    )
    _wm_getter = lambda: temp_working_memory
    monkeypatch.setattr(
        "server.services.conversation.summarization.working_memory_log.get_working_memory_log",
        _wm_getter,
    )
    monkeypatch.setattr(
        "server.services.conversation.summarization.get_working_memory_log",
        _wm_getter,
    )
    monkeypatch.setattr(
        "server.services.conversation.get_working_memory_log", _wm_getter
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.get_working_memory_log", _wm_getter
    )
    monkeypatch.setattr(
        "server.services.conversation.log._resolve_working_memory_log",
        _wm_getter,
    )
    monkeypatch.setattr(
        "server.services.conversation.summarization.summarizer.get_working_memory_log",
        _wm_getter,
    )

    monkeypatch.setattr(
        "server.services.execution.log_store._execution_agent_logs", temp_exec_logs
    )
    _exec_getter = lambda: temp_exec_logs
    monkeypatch.setattr(
        "server.services.execution.log_store.get_execution_agent_logs", _exec_getter
    )
    monkeypatch.setattr(
        "server.services.execution.get_execution_agent_logs", _exec_getter
    )
    monkeypatch.setattr(
        "server.services.get_execution_agent_logs", _exec_getter
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.tools.get_execution_agent_logs", _exec_getter
    )
    monkeypatch.setattr(
        "server.agents.execution_agent.agent.get_execution_agent_logs", _exec_getter
    )


def _patch_batch_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub execution-agent dispatch so benchmarks stay inside the interaction loop."""

    class _StubBatchManager:
        async def execute_agent(self, agent_name, instructions, request_id=None):
            class _Result:
                def __init__(self):
                    self.agent_name = agent_name
                    self.success = True
                    self.response = "Mocked execution."

            return _Result()

    monkeypatch.setattr(
        "server.agents.interaction_agent.tools._get_execution_batch_manager",
        lambda: _StubBatchManager(),
    )


def _install_tool_recorder(monkeypatch: pytest.MonkeyPatch) -> "ToolCallRecorder":
    """Wrap handle_tool_call and patch both the tools and runtime import sites."""

    from server.agents.interaction_agent.tools import handle_tool_call

    recorder = ToolCallRecorder(handle_tool_call)
    monkeypatch.setattr(
        "server.agents.interaction_agent.tools.handle_tool_call", recorder
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.handle_tool_call", recorder
    )
    return recorder


@pytest.fixture()
def wired_env(
    monkeypatch: pytest.MonkeyPatch,
    temp_roster: AgentRoster,
    temp_conversation_log: ConversationLog,
    temp_working_memory: WorkingMemoryLog,
    temp_exec_logs: ExecutionAgentLogStore,
    fake_settings: Settings,
    mock_llm: MockOpenRouterResponder,
):
    """Patch all singletons and return a namespace with the temp objects."""

    _patch_settings_aliases(monkeypatch, fake_settings)
    _wire_shared_singletons(
        monkeypatch,
        temp_roster=temp_roster,
        temp_conversation_log=temp_conversation_log,
        temp_working_memory=temp_working_memory,
        temp_exec_logs=temp_exec_logs,
    )

    monkeypatch.setattr(
        "server.openrouter_client.client.request_chat_completion", mock_llm
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.request_chat_completion", mock_llm
    )
    monkeypatch.setattr(
        "server.agents.execution_agent.runtime.request_chat_completion", mock_llm
    )

    _patch_batch_manager(monkeypatch)

    class _WiredEnv:
        roster = temp_roster
        conversation_log = temp_conversation_log
        working_memory = temp_working_memory
        exec_logs = temp_exec_logs
        settings = fake_settings
        llm = mock_llm

    return _WiredEnv()


# ---------------------------------------------------------------------------
# V2: Tool call recorder
# ---------------------------------------------------------------------------

@dataclass
class ToolInvocation:
    """A single recorded tool call."""
    name: str
    arguments: Dict[str, Any]
    result: ToolResult


class ToolCallRecorder:
    """Wraps handle_tool_call to record ordered invocations."""

    def __init__(self, original_fn: Callable):
        self._original = original_fn
        self.invocations: List[ToolInvocation] = []

    def __call__(self, name: str, arguments: Any) -> ToolResult:
        result = self._original(name, arguments)
        self.invocations.append(ToolInvocation(
            name=name,
            arguments=arguments if isinstance(arguments, dict) else {},
            result=result,
        ))
        return result


@pytest.fixture()
def tool_recorder(monkeypatch: pytest.MonkeyPatch, wired_env) -> ToolCallRecorder:
    """Wrap handle_tool_call with a recorder. Must be used after wired_env."""
    return _install_tool_recorder(monkeypatch)


# ---------------------------------------------------------------------------
# V3: Live environment — temp-backed files but REAL LLM calls
# ---------------------------------------------------------------------------

@pytest.fixture()
def live_settings() -> Settings:
    """Settings using the real API key and a cheap benchmark model override."""
    get_settings.cache_clear()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set — skipping live test")
    benchmark_model = os.getenv("OPENPOKE_BENCHMARK_MODEL", "anthropic/claude-sonnet-4")
    return Settings(
        openrouter_api_key=api_key,
        conversation_summary_threshold=0,
        interaction_agent_model=benchmark_model,
        execution_agent_model=benchmark_model,
    )


@pytest.fixture()
def wired_env_live(
    monkeypatch: pytest.MonkeyPatch,
    temp_roster: AgentRoster,
    temp_conversation_log: ConversationLog,
    temp_working_memory: WorkingMemoryLog,
    temp_exec_logs: ExecutionAgentLogStore,
    live_settings: Settings,
):
    """Like wired_env but uses the REAL OpenRouter API. No LLM mock."""

    _patch_settings_aliases(monkeypatch, live_settings)
    _wire_shared_singletons(
        monkeypatch,
        temp_roster=temp_roster,
        temp_conversation_log=temp_conversation_log,
        temp_working_memory=temp_working_memory,
        temp_exec_logs=temp_exec_logs,
    )
    _patch_batch_manager(monkeypatch)

    class _LiveEnv:
        roster = temp_roster
        conversation_log = temp_conversation_log
        working_memory = temp_working_memory
        exec_logs = temp_exec_logs
        settings = live_settings

    return _LiveEnv()


@pytest.fixture()
def tool_recorder_live(
    monkeypatch: pytest.MonkeyPatch,
    wired_env_live,
) -> ToolCallRecorder:
    """Live-tool recorder that still executes the real tool functions."""

    return _install_tool_recorder(monkeypatch)
