"""Benchmark fixtures — replaces all singletons with temp-backed instances."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from server.config import Settings, get_settings
from server.services.execution.roster import AgentRoster
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.conversation.log import ConversationLog
from server.services.conversation.summarization.working_memory_log import WorkingMemoryLog

from .factories import populate_roster, write_conversation_log
from .mock_llm import MockOpenRouterResponder


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

AGENT_COUNTS = [100, 500, 1000]
CONVERSATION_TURNS = [50, 200, 500]


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
def temp_conversation_log(data_dir: Path) -> ConversationLog:
    log_path = data_dir / "conversation" / "poke_conversation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    return Settings(
        openrouter_api_key="fake-benchmark-key",
        conversation_summary_threshold=0,  # disables summarization
    )


@pytest.fixture()
def mock_llm() -> MockOpenRouterResponder:
    return MockOpenRouterResponder()


# ---------------------------------------------------------------------------
# Wired runtime — patches every import site
# ---------------------------------------------------------------------------

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

    # --- Settings (lru_cache must be cleared first) ---
    get_settings.cache_clear()
    monkeypatch.setattr("server.config.get_settings", lambda: fake_settings)

    # --- AgentRoster: module-level singleton + getter at every import site ---
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

    # --- ConversationLog: module-level singleton + getter ---
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

    # --- WorkingMemoryLog: module-level singleton + getter ---
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
    # ConversationLog.__init__ resolves working memory via lazy import
    monkeypatch.setattr(
        "server.services.conversation.log._resolve_working_memory_log",
        _wm_getter,
    )

    # --- ExecutionAgentLogStore: module-level singleton + getter ---
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

    # --- OpenRouter mock at every import site ---
    monkeypatch.setattr(
        "server.openrouter_client.client.request_chat_completion", mock_llm
    )
    monkeypatch.setattr(
        "server.agents.interaction_agent.runtime.request_chat_completion", mock_llm
    )
    monkeypatch.setattr(
        "server.agents.execution_agent.runtime.request_chat_completion", mock_llm
    )

    class _WiredEnv:
        roster = temp_roster
        conversation_log = temp_conversation_log
        working_memory = temp_working_memory
        exec_logs = temp_exec_logs
        settings = fake_settings
        llm = mock_llm

    return _WiredEnv()
