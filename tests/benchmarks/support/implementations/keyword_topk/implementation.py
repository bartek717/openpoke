"""Lexical top-k benchmark implementation."""

from __future__ import annotations

from typing import List

from ..utils import tokenize


_DEFAULT_KEYWORD_TOP_K = 25


class KeywordTopKImplementation:
    """Simple lexical pre-filter for benchmark experimentation."""

    name = "keyword_topk"

    def __init__(self, top_k: int = _DEFAULT_KEYWORD_TOP_K) -> None:
        self.top_k = max(1, top_k)

    def select_agents(
        self,
        latest_text: str,
        transcript: str,
        agents: List[str],
    ) -> List[str]:
        if len(agents) <= self.top_k:
            return list(agents)

        query_tokens = tokenize(f"{latest_text}\n{transcript}")
        if not query_tokens:
            return list(agents[: self.top_k])

        scored: list[tuple[int, int, int, str]] = []
        for index, agent_name in enumerate(agents):
            name_tokens = tokenize(agent_name)
            overlap = len(query_tokens & name_tokens)
            exact_phrase = int(agent_name.lower() in latest_text.lower())
            score = overlap * 10 + exact_phrase * 100
            scored.append((score, overlap, -index, agent_name))

        scored.sort(reverse=True)
        selected = [name for _, _, _, name in scored[: self.top_k]]

        if not any(score > 0 for score, _, _, _ in scored[: self.top_k]):
            return list(agents)

        return selected
