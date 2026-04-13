# Execution Agent Overload

OpenPoke's multi-agent architecture has a scaling problem: the entire roster of execution agents is injected into the Interaction Agent's context window on every single request, with no filtering, pruning, or relevance scoring.

## How it works today

When a user sends a message, the Interaction Agent receives three sections bundled into a single LLM prompt:

```
<conversation_history>
  ... summarized or full transcript ...
</conversation_history>

<active_agents>
  <agent name="Email to Alice" />
  <agent name="Q3 Budget Analysis" />
  <agent name="Tokyo Restaurant Search" />
  ... every agent ever created ...
</active_agents>

<new_user_message>
  Did Alice reply yet?
</new_user_message>
```

The `<active_agents>` block is built by `_render_active_agents()` in `server/agents/interaction_agent/agent.py` (lines 45-58). It calls `get_agent_roster().get_agents()`, which returns **every agent name ever added to `roster.json`** as a flat list. There is no filtering by recency, relevance, or activity status.

The roster itself (`server/services/execution/roster.py`) is a simple JSON array of strings on disk. Agents are added via `add_agent()` whenever `send_message_to_agent` is called with a new name (see `server/agents/interaction_agent/tools.py`, lines 112-150), but they are **never removed** except when the user explicitly clears all history via `DELETE /api/v1/chat/history`.

## Why this is a problem

### Context window bloat

Each agent name consumes tokens in the Interaction Agent's prompt. After days of use, hundreds of agents accumulate: "Email to Alice", "Email to Bob", "Weekly Report", "Lunch with Sarah", "Flight Confirmation Lookup", "Q3 Budget Analysis", etc. These names are injected on every request whether relevant or not.

The conversation transcript already undergoes summarization to stay within context limits (see `server/services/conversation/summarization/`), but the agent roster has no equivalent compression mechanism. As the roster grows, it takes up an increasing share of the context budget that should be going to conversation history and the current task.

### Relevance noise

The Interaction Agent's system prompt instructs it to "always prefer to send messages to a relevant existing agent rather than starting a new one" (line 19 of `system_prompt.md`). With hundreds of agents in the `<active_agents>` block, the LLM has to scan all of them to decide which one to reuse. This creates two failure modes:

1. **Wrong agent selection** - the LLM picks an agent whose name sounds similar but handles a different context (e.g., reusing "Email to Alice" from a September lunch thread when the user asks about a December project update from Alice).

2. **Unnecessary new agents** - the LLM misses the relevant agent in a long list and spawns a duplicate, which then lacks the original agent's execution history.

### Per-agent log growth

Each execution agent has its own append-only log file at `server/data/execution_agents/{slug}.log` (managed by `ExecutionAgentLogStore` in `server/services/execution/log_store.py`). When an agent is reactivated, its **entire history** is loaded into the execution agent's system prompt via `build_system_prompt_with_history()` in `server/agents/execution_agent/agent.py` (lines 63-96).

A `conversation_limit` parameter exists but is **never set** by the code that spawns agents (see `ExecutionAgentRuntime.__init__` in `server/agents/execution_agent/runtime.py`, line 33 - it creates `ExecutionAgent(agent_name)` with no limit). So every reactivation loads the full history, which grows unbounded.

### Cost multiplication

Every user message triggers at least one LLM call to the Interaction Agent. A bloated roster means more input tokens per call. If the Interaction Agent then delegates to execution agents, each of those also makes LLM calls with their own growing histories. The blog post estimates ~$50/month per user in LLM costs even at current scale; unbounded roster and log growth makes this worse over time.

## What the codebase lacks

| Mechanism | Status |
|-----------|--------|
| Roster size cap | None |
| Agent archival/retirement | None |
| Recency-based filtering | None |
| Semantic/relevance search over agents | None |
| Agent activity timestamps | Not tracked |
| Agent last-used metadata | Not tracked |
| Execution log summarization | None |
| Execution `conversation_limit` being set | Never passed |

## Potential approaches

The blog post mentions several options. Here is how they map to the current code:

### 1. Semantic search over agent descriptions

Instead of dumping all agent names into `<active_agents>`, embed agent names (and optionally a summary of their execution history) into a vector store. When a new user message arrives, retrieve only the top-K most relevant agents. This would replace the `_render_active_agents()` function with a retrieval step.

### 2. Activity-based archival

Track `last_used_at` timestamps on agents. Agents not reactivated within some window (e.g., 7 days) get moved to an "archived" state. Archived agents are excluded from the `<active_agents>` block but can be resurrected if semantic search identifies them as relevant.

### 3. Hot cache of recent agents

Maintain a fixed-size window (e.g., last 10-20 agents used) that always appears in the prompt. Older agents are available only through search. This is the simplest approach and would cap context usage at a predictable level.

### 4. Agent clustering

Group related agents (e.g., all "Email to Alice" variants, all "Weekly Report" agents) under a single umbrella entry. The Interaction Agent sees "Alice email thread (3 agents)" rather than three separate entries. When it delegates to the cluster, a routing layer picks the right sub-agent.

### 5. Execution log summarization

Mirror the conversation summarization system (`server/services/conversation/summarization/`) for execution agent logs. After N interactions, compress older entries into a summary while keeping recent ones at full fidelity. This would cap the per-agent context cost when agents are reactivated.

### 6. Actually use `conversation_limit`

The simplest immediate fix: pass a `conversation_limit` when constructing `ExecutionAgent` instances. The machinery to slice logs already exists in `build_system_prompt_with_history()` (lines 77-92 of `agent.py`) - it just never gets a non-None value.
