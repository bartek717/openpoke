"""Mock OpenRouter client for benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMCallRecord:
    """Metadata captured for a single LLM call."""

    call_index: int
    model: str
    system_length: int
    messages_content_length: int
    tool_schemas_count: int
    full_payload_bytes: int


class MockOpenRouterResponder:
    """Returns pre-scripted LLM responses and records call metadata."""

    def __init__(self, scenario: Optional[List[Dict[str, Any]]] = None):
        self._scenario = scenario or []
        self._call_index = 0
        self.calls: List[LLMCallRecord] = []

    async def __call__(
        self,
        *,
        model: str = "",
        messages: List[Dict[str, Any]] = None,
        system: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        base_url: str = "",
    ) -> Dict[str, Any]:
        messages = messages or []

        # Build the payload that would be sent to OpenRouter
        payload: Dict[str, Any] = {
            "model": model,
            "messages": (
                [{"role": "system", "content": system}, *messages] if system else messages
            ),
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        full_payload_bytes = len(json.dumps(payload, default=str).encode("utf-8"))
        messages_content_length = sum(
            len(str(m.get("content", ""))) for m in messages
        )

        record = LLMCallRecord(
            call_index=self._call_index,
            model=model,
            system_length=len(system) if system else 0,
            messages_content_length=messages_content_length,
            tool_schemas_count=len(tools) if tools else 0,
            full_payload_bytes=full_payload_bytes,
        )
        self.calls.append(record)

        if self._call_index < len(self._scenario):
            response = self._scenario[self._call_index]
        else:
            response = _default_final_response()

        self._call_index += 1
        return response


def _default_final_response() -> Dict[str, Any]:
    """A terminal response with no tool calls."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Done.",
                    "tool_calls": [],
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Pre-built scenarios for V2
# ---------------------------------------------------------------------------

def _tool_call(call_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single tool_call entry in OpenRouter response format."""
    import json as _json
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json.dumps(arguments),
        },
    }


def scenario_dispatch_one(agent_name: str = "Benchmark Agent") -> List[Dict[str, Any]]:
    """LLM calls send_message_to_user + send_message_to_agent, then stops.

    Call 0: returns two tool calls (user message + agent dispatch)
    Call 1: (after tool results) returns final text, no tool calls
    """
    return [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            _tool_call(
                                "call_user_1",
                                "send_message_to_user",
                                {"message": "On it, working on that now."},
                            ),
                            _tool_call(
                                "call_agent_1",
                                "send_message_to_agent",
                                {
                                    "agent_name": agent_name,
                                    "instructions": "Complete the benchmark task.",
                                },
                            ),
                        ],
                    }
                }
            ]
        },
        # Second call: final response after tool results
        _default_final_response(),
    ]


def scenario_reuse_agent(agent_name: str) -> List[Dict[str, Any]]:
    """LLM reuses an existing agent by name."""
    return [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            _tool_call(
                                "call_user_1",
                                "send_message_to_user",
                                {"message": "Following up on that."},
                            ),
                            _tool_call(
                                "call_agent_1",
                                "send_message_to_agent",
                                {
                                    "agent_name": agent_name,
                                    "instructions": "Follow up on the previous task.",
                                },
                            ),
                        ],
                    }
                }
            ]
        },
        _default_final_response(),
    ]


def scenario_fan_out(agent_names: List[str]) -> List[Dict[str, Any]]:
    """LLM dispatches to multiple agents in one turn."""
    tool_calls = [
        _tool_call(
            "call_user_1",
            "send_message_to_user",
            {"message": "Working on all of those now."},
        ),
    ]
    for i, name in enumerate(agent_names):
        tool_calls.append(
            _tool_call(
                f"call_agent_{i + 1}",
                "send_message_to_agent",
                {
                    "agent_name": name,
                    "instructions": f"Task {i + 1} for {name}.",
                },
            )
        )
    return [
        {"choices": [{"message": {"content": "", "tool_calls": tool_calls}}]},
        _default_final_response(),
    ]


def scenario_noop() -> List[Dict[str, Any]]:
    """LLM calls wait tool (no-op)."""
    return [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            _tool_call(
                                "call_wait_1",
                                "wait",
                                {"reason": "Message already sent."},
                            ),
                        ],
                    }
                }
            ]
        },
        _default_final_response(),
    ]
