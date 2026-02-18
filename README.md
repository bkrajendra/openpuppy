# 🐕 OpenPuppy : Intelligent Autonomous AI Agent

LangGraph-based agent with multi-LLM support, supervisor pattern, 11+ tools, SQLite + ChromaDB memory, and multiple interfaces: **CLI**, **Telegram bot**, **Web API** (chat + admin dashboard).

This project uses **uv** for dependency management (https://github.com/astral-sh/uv). You can also use `pip` and `python` if you prefer.

---

## Get started

### 1. Install dependencies

**With uv:**

```bash
uv sync
```

With dev dependencies (tests):

```bash
uv sync --all-extras
```

**With pip:**

```bash
pip install -e .
```

### 2. Configure environment

Copy the example env file and set at least your OpenAI key:

```bash
cp .env.example .env
```

Edit `.env` and set:

- **`OPENAI_API_KEY`** — required for the default LLM (e.g. `gpt-4o-mini`).

Optional:

- **`OPENAI_MODEL`** — override model (default from config: `gpt-4o-mini`).
- **`TELEGRAM_BOT_TOKEN`** — for Telegram bot (see [Telegram](#telegram-bot) below).
- **`TELEGRAM_ALLOWED_USER_IDS`** — comma-separated user IDs to restrict who can use the bot.
- **`AGENT_MEMORY_DB`** — path to SQLite DB (default: `./data/agent_memory.db`).
- **`AGENT_API_PORT`** — port for Web API (default: `8000`).

### 3. Config file (optional)

Main config is in **`config/agent_config.yaml`**. It controls:

- **Agent:** name, `max_iterations`, `timeout_seconds`.
- **LLM:** primary provider/model, temperature, max_tokens; optional fallback (e.g. Ollama).
- **Tools:** which tools are enabled, sandboxing for code executor.
- **Memory:** DB path, ChromaDB vector store.
- **Interfaces:** enable/disable CLI, Telegram, API; API port.
- **Scheduler:** optional cron jobs for autonomous tasks.

You can edit this file directly or use the **Admin UI** (see [Web API](#web-api--admin)) to edit config from the browser.

### 4. Initialize database (optional)

For conversation and long-term memory:

```bash
uv run python scripts/setup_db.py
```

### 5. Run the agent

Pick one of the modes below. Easiest first run: **CLI**.

---

## Usage

### CLI (interactive chat)

Terminal conversation loop; no extra setup.

```bash
uv run agent-cli
```

Or:

```bash
uv run python scripts/run_agent.py
```

Type your message, press Enter; the agent uses tools (search, calculator, files, etc.) and replies. Exit with your shell’s EOF (e.g. Ctrl+D) or type a command if the CLI supports one.

---

### Telegram bot

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token.
2. In `.env` set:
   - **`TELEGRAM_BOT_TOKEN=your_token`**
   - Optional: **`TELEGRAM_ALLOWED_USER_IDS=123,456`** (comma-separated Telegram user IDs; only these users can use the bot).
3. Start the bot:

   ```bash
   uv run agent-telegram
   ```

4. Open your bot in Telegram and send a message.

Optional: expose Prometheus metrics with **`PROMETHEUS_METRICS_PORT=9090`** (or another port) in `.env` when running the bot.

---

### Web API (chat + admin)

HTTP API with a built-in chat UI and an admin dashboard (tools, agents, config).

1. Start the server:

   ```bash
   uv run agent-api
   ```

   Or with uvicorn directly:

   ```bash
   uvicorn src.interfaces.api:app --host 0.0.0.0 --port 8000
   ```

2. Open in a browser:
   - **Chat:** http://localhost:8000/
   - **Admin dashboard:** http://localhost:8000/admin (status, tools, agents, config).

3. API endpoints:
   - **`GET /health`** — health check.
   - **`GET /tools`** — list tool names and schemas.
   - **`POST /chat`** — send a message; body: `{"message": "...", "thread_id": "optional_id"}`.
   - **`GET /admin/status`** — dashboard status (tool counts, teams).
   - **`GET /admin/agents`** — supervisor teams and their tools.
   - **`GET /admin/tools`** — all tools (built-in + custom) with source.
   - **`POST /admin/tools`** — add a custom HTTP tool (name, description, url, method, parameters_schema).
   - **`DELETE /admin/tools/{name}`** — remove a custom tool.
   - **`GET /admin/config`** — current config (JSON).
   - **`PATCH /admin/config`** — save config (JSON body).

Port can be overridden with **`AGENT_API_PORT`** in `.env` or the `--port` argument to uvicorn.

---

### One-off in code

```python
import asyncio
from src.agent.executor import run_agent

async def main():
    state = await run_agent("What is 2+2?")
    print(state["final_response"])

asyncio.run(main())
```

Run: `uv run python -c "..."` or `uv run python your_script.py`.

---

## Config summary

| What            | Where                         |
|-----------------|-------------------------------|
| API keys, ports | `.env` (see `.env.example`)   |
| Agent/LLM/tools | `config/agent_config.yaml`    |
| Custom tools    | Added via Admin UI → `data/custom_tools.json` |

Config file is loaded at startup; edits in **Admin → Config** are written to `config/agent_config.yaml`. Some changes (e.g. enabled tools) may require a restart.

---

## Features (overview)

- **Graph:** Supervisor → Router → Tool executor (loop) → Synthesizer. Supervisor routes to research / code / general tool sets.
- **LLMs:** OpenAI (primary), Anthropic Claude, Ollama; configurable with fallback.
- **Tools:** web_search, code_executor, read_file, write_file, list_directory, wikipedia_lookup, calculator, weather, store_memory, retrieve_memory, plus custom HTTP tools added via Admin UI.
- **Memory:** SQLite conversations; ChromaDB vector store for semantic memory.
- **Interfaces:** CLI (`agent-cli`), Telegram (`agent-telegram`), Web API + Admin (`agent-api`).
- **MCP:** `src/mcp/adapter.py` — export tools and handle MCP tool calls.
- **Scheduler:** `uv run python scripts/run_scheduler.py` — cron-based agent runs (configure in `scheduler.jobs`).
- **Docker:** `docker compose up -d telegram` or `docker compose run --rm agent-cli` (see repo for compose file).

---

## Tests

```bash
uv run pytest
```

---

## Project layout

See **`SPEC.md`** for full architecture. Key directories:

- **`src/agent`** — graph, nodes, supervisor, executor.
- **`src/llm`** — LLM providers (OpenAI, Anthropic, Ollama).
- **`src/tools`** — tool registry, built-in tools, custom tools, plugins.
- **`src/memory`** — SQLite + ChromaDB.
- **`src/interfaces`** — CLI, Telegram bot, FastAPI (chat + admin).
- **`config/`** — `agent_config.yaml`.
