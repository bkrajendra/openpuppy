# Intelligent Autonomous AI Agent

Phase 4: LangGraph, multi-LLM, supervisor, 11+ tools, SQLite + ChromaDB, CLI + Telegram + Web API, MCP, rate limiting, Prometheus, Docker, scheduler, plugins, A/B prompts.

This project uses **uv** for dependency management and running (https://github.com/astral-sh/uv).

## Setup

1. **Install uv** (if needed): https://docs.astral.sh/uv/getting-started/installation/

2. **Sync environment** (creates `.venv` and installs dependencies):

   ```bash
   uv sync
   ```

   With dev dependencies (tests):

   ```bash
   uv sync --all-extras
   ```

3. **Configure environment:** copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

4. **Optional — initialize DB:**

   ```bash
   uv run python scripts/setup_db.py
   ```

## Run

- **CLI (conversation loop):**

  ```bash
  uv run agent-cli
  ```

  Or run the script directly:

  ```bash
  uv run python scripts/run_agent.py
  ```

- **Tests:**

  ```bash
  uv run pytest
  ```

- **One-off in code:**

  ```python
  import asyncio
  from src.agent.executor import run_agent

  async def main():
      state = await run_agent("What is 2+2?")
      print(state["final_response"])

  asyncio.run(main())
  ```

  Run it: `uv run python -c "..."` or from a file: `uv run python your_script.py`

## Phase 2 Features

- **Graph:** Router → Tool Executor (max 5 iterations) → Synthesizer → END
- **LLMs:** OpenAI (primary), Anthropic Claude, Ollama (local); configurable with fallback
- **Tools:** `web_search`, `code_executor`, `read_file`, `write_file`, `list_directory`, `wikipedia_lookup`, `calculator`, `weather`, `store_memory`, `retrieve_memory`
- **Memory:** SQLite conversations; ChromaDB vector store for semantic long-term memory
- **Interfaces:** CLI (`uv run agent-cli`), Telegram (`uv run agent-telegram`)

## Run Telegram bot

1. Create a bot with [@BotFather](https://t.me/BotFather), get the token.
2. Set `TELEGRAM_BOT_TOKEN=...` in `.env`.
3. Optional: restrict to specific users: `TELEGRAM_ALLOWED_USER_IDS=123,456` in `.env`.
4. Run: `uv run agent-telegram`. Optional: `PROMETHEUS_METRICS_PORT=9090` to expose metrics.

## Phase 3 (Production)

- **MCP:** `src/mcp/adapter.py` – export tool manifest and handle MCP tool calls.
- **Rate limiting:** Telegram per-user limits via `TELEGRAM_RATE_LIMIT_MAX` / `TELEGRAM_RATE_LIMIT_WINDOW_SECONDS`.
- **Metrics:** Prometheus counters/histogram; set `PROMETHEUS_METRICS_PORT` when running the bot.
- **Docker:** `docker compose up -d telegram` or `docker compose run --rm agent-cli`. Data in volume `agent-data`.
- **Backup:** `uv run python scripts/backup_restore.py backup -o data/backups` and `restore -i data/backups`.

## Phase 4 (Advanced)

- **Supervisor:** Routes to research/code/general tool sets. Disable with `create_agent(use_supervisor=False)`.
- **Scheduler:** `uv run python scripts/run_scheduler.py` – runs agent on cron (config `scheduler.jobs`).
- **Tool composition:** `run_tool` meta-tool; `execute_tool(..., _call_depth, _max_call_depth=2)`.
- **Web API:** `uv run agent-api` or `uvicorn src.interfaces.api:app --port 8000`. Open `/` for chat UI (Tailwind). `POST /chat`, `GET /tools`, `GET /health`.
- **Plugins:** Add `tools.plugins: [my.module]` in config; modules register tools with `tool_registry`.
- **Memory:** Episodic = `EpisodicMemory` (recent turns); semantic = ChromaDB `VectorStore`.
- **A/B prompts:** `config.prompts.router: [ "prompt A", "prompt B" ]` and `get_prompt_variant()` in `src/utils/prompts_ab.py`.

## Project layout

See `SPEC.md` for full architecture. Key dirs: `src/agent`, `src/llm`, `src/tools`, `src/memory`, `src/interfaces`, `config/`.
