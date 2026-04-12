# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is OpenPoke

Open-source multi-agent assistant (FastAPI backend + Next.js frontend) that orchestrates email triage (Gmail via Composio) and reminders. An **Interaction Agent** (Claude 3.5 Sonnet via OpenRouter) handles user queries and delegates to **Execution Agents** for tasks like drafting emails and scheduling reminders.

## Commands

### Backend (Python 3.10+)
```bash
# Setup
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt

# Run (dev with auto-reload)
python -m server.server --reload

# Run (production)
python -m server.server
# Options: --host (default 0.0.0.0), --port (default 8001)
```

### Frontend (Node.js)
```bash
npm install --prefix web
npm run dev --prefix web      # Dev server on port 3000
npm run build --prefix web    # Production build
npm run lint --prefix web     # ESLint
```

### Environment
Single `.env` file at repo root shared by both backend and frontend. Copy `.env.example` to `.env` and fill in:
- `OPENROUTER_API_KEY` (required) — LLM access
- `COMPOSIO_API_KEY` + `COMPOSIO_GMAIL_AUTH_CONFIG_ID` (required) — Gmail integration

## Architecture

### Communication Flow
1. User message → Next.js → `POST /api/v1/chat/send` (returns 202)
2. FastAPI routes to Interaction Agent (main LLM)
3. Interaction Agent delegates to Execution Agents via `send_message_to_agent` tool calls
4. Frontend polls `GET /api/v1/chat/history` (1.5s interval, 30s timeout) until response appears

### Agent System
- **Interaction Agent** (`server/agents/interaction_agent/`) — orchestrator LLM with a ~10KB system prompt defining routing behavior. Single instance.
- **Execution Agents** (`server/agents/execution_agent/`) — task specialists spawned dynamically (e.g., "conversation with Keith"). Each gets a ~3.7KB system prompt.
- Agent roster and logs stored under `server/services/execution/`

### Background Services (async, started on app startup)
- **TriggerScheduler** (`server/services/trigger_scheduler.py`) — polls SQLite triggers table every 10s, spawns execution agents on schedule (iCalendar RRULE support)
- **ImportantEmailWatcher** (`server/services/gmail/`) — monitors Gmail for important emails, classifies via LLM, notifies user

### Data Storage
All runtime data lives in `server/data/` (gitignored):
- `conversation/poke_conversation.log` — plaintext conversation log
- `execution_agents/` — per-agent logs
- `triggers.db` — SQLite for reminders/schedules
- `timezone.txt` — browser-detected timezone

### Key Backend Structure
- `server/app.py` — FastAPI app setup, middleware, lifecycle hooks
- `server/config.py` — Pydantic settings loaded from `.env`
- `server/routes/` — API endpoints (chat, gmail, meta)
- `server/services/` — business logic (conversation management, gmail ops, trigger scheduling)
- `server/openrouter_client/` — OpenRouter API client wrapper
- `server/models/` — Pydantic data models

### Frontend Structure
- `web/app/page.tsx` — main chat page with polling logic
- `web/app/api/` — Next.js API routes that proxy to the FastAPI backend
- `web/components/chat/` — chat UI components (ChatMessages, ChatInput, ChatHeader)
- `web/next.config.mjs` — loads repo root `.env` for frontend access
- Path aliases: `@/components`, `@/lib` (configured in tsconfig.json)

### API Endpoints
- `POST /api/v1/chat/send` — submit user message
- `GET /api/v1/chat/history` — get conversation
- `DELETE /api/v1/chat/history` — clear all data (conversation, agents, triggers, rosters)
- `POST /api/v1/gmail/connect` — initiate Gmail OAuth
- `GET /api/v1/gmail/status` — check Gmail auth status
- `POST /api/v1/gmail/disconnect` — revoke Gmail access
- `POST /api/v1/timezone` — store browser timezone
