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
