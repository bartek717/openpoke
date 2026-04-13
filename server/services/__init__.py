"""Service layer components."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ConversationLog": (".conversation", "ConversationLog"),
    "SummaryState": (".conversation", "SummaryState"),
    "get_conversation_log": (".conversation", "get_conversation_log"),
    "get_working_memory_log": (".conversation", "get_working_memory_log"),
    "schedule_summarization": (".conversation", "schedule_summarization"),
    "handle_chat_request": (".conversation.chat_handler", "handle_chat_request"),
    "AgentRoster": (".execution", "AgentRoster"),
    "ExecutionAgentLogStore": (".execution", "ExecutionAgentLogStore"),
    "get_agent_roster": (".execution", "get_agent_roster"),
    "get_execution_agent_logs": (".execution", "get_execution_agent_logs"),
    "GmailSeenStore": (".gmail", "GmailSeenStore"),
    "ImportantEmailWatcher": (".gmail", "ImportantEmailWatcher"),
    "classify_email_importance": (".gmail", "classify_email_importance"),
    "disconnect_account": (".gmail", "disconnect_account"),
    "execute_gmail_tool": (".gmail", "execute_gmail_tool"),
    "fetch_status": (".gmail", "fetch_status"),
    "get_active_gmail_user_id": (".gmail", "get_active_gmail_user_id"),
    "get_important_email_watcher": (".gmail", "get_important_email_watcher"),
    "initiate_connect": (".gmail", "initiate_connect"),
    "get_trigger_scheduler": (".trigger_scheduler", "get_trigger_scheduler"),
    "get_trigger_service": (".triggers", "get_trigger_service"),
    "TimezoneStore": (".timezone_store", "TimezoneStore"),
    "get_timezone_store": (".timezone_store", "get_timezone_store"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolve exported service helpers on first access."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
