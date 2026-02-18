Phase 2 summary
1. Dependencies (pyproject.toml)
anthropic, python-telegram-bot, chromadb, requests
2. LLM providers
src/llm/ollama.py – Ollama (OpenAI-compatible), configurable base_url and model.
src/llm/anthropic.py – Anthropic Claude with tool use (OpenAI-style messages converted to Anthropic format).
src/utils/llm_factory.py – Builds provider from config; FallbackLLMProvider uses fallback when primary fails.
3. New tools
File: read_file, write_file, list_directory (restricted to ALLOWED_FILE_PATH, default data/).
src/tools/wikipedia.py – wikipedia_lookup (MediaWiki API).
src/tools/calculator.py – calculator (safe math expressions).
src/tools/weather.py – weather (Open-Meteo, no key).
src/tools/memory_tools.py – store_memory, retrieve_memory (ChromaDB).
4. ChromaDB
src/memory/vector_store.py – VectorStore with add() and search(); used by the memory tools.
5. Telegram
src/interfaces/telegram_bot.py – Handles text messages, runs agent, sends final reply; typing indicator while processing.
Entry point: uv run agent-telegram (or agent-telegram after install).
Requires TELEGRAM_BOT_TOKEN in .env.
6. Config and CLI
config/agent_config.yaml – llm.primary + llm.fallback, full tool list, memory.vector_store, interfaces.telegram.
CLI uses get_llm_from_config() so primary/fallback and model come from config.
7. Optional
Streaming: Existing stream_agent + CLI /stream still used; token-level streaming can be added later.
Parallel tool execution: Already in place in the tool executor node.

---

Phase 3 summary (Production Hardening)
1. Telegram safeguard (hardened)
- Allowlist: allowed user IDs from TELEGRAM_ALLOWED_USER_IDS (comma-separated) or config interfaces.telegram.allowed_user_ids (list or string). Comparison fixed to use int(user_id) so allowlist works.
- No PII in logs: log only user_id for received/unauthorized; full update object removed.
- If allowlist is empty, all users allowed (backward compatible). If set, only those IDs are served.
2. MCP adapter
- src/mcp/adapter.py – MCPAdapter(tool_registry): export_mcp_manifest() returns MCP-format tool list; handle_mcp_request(request) handles tools/call with name + arguments and returns content or error.
3. Rate limiting
- src/utils/rate_limit.py – RateLimiter (in-memory, per-key). Telegram uses get_telegram_rate_limiter() (TELEGRAM_RATE_LIMIT_MAX=30, TELEGRAM_RATE_LIMIT_WINDOW_SECONDS=60). Rejects with “Rate limited. Please try again later.”
4. Monitoring (Prometheus)
- src/utils/monitoring.py – Counters: agent_tool_executions_total (tool_name, status), agent_invocations_total (interface). Histogram: agent_llm_latency_seconds (provider). Tool registry and Telegram handler record metrics. Optional HTTP server: set PROMETHEUS_METRICS_PORT (e.g. 9090) to expose /metrics.
5. Docker
- Dockerfile: Python 3.11-slim, uv, install from pyproject.toml, VOLUME /data, ENTRYPOINT agent-cli.
- docker-compose.yml: services agent-cli (interactive) and telegram (background, port 9090 for metrics). Uses .env and agent-data volume.
6. Backup/restore
- scripts/backup_restore.py – backup: copies data/agent_memory.db and data/vector_store to output dir (default data/backups). restore: copies from input dir back to data/. CLI: python scripts/backup_restore.py backup -o data/backups; restore -i data/backups.
7. Dependencies
- prometheus-client added in pyproject.toml.

---

Phase 4 summary (Advanced Features)
1. Multi-agent supervisor pattern
- src/agent/supervisor.py: supervisor_node() chooses team (research | code | general). TEAM_TOOLS maps teams to tool name lists; tool_executor filters schemas by state.team. Graph: START -> supervisor -> router -> tool_executor | synthesizer -> END. create_agent(use_supervisor=True) by default.
2. Scheduled/cron autonomous tasks
- src/scheduler/runner.py: APScheduler (AsyncIOScheduler). add_agent_job(job_id, prompt, cron) runs run_agent(prompt) on schedule. start_scheduler() loads jobs from config.scheduler.jobs (id, prompt, cron). scripts/run_scheduler.py runs the scheduler (uv run python scripts/run_scheduler.py).
3. Tool composition
- ToolRegistry.execute_tool(..., _call_depth=0, _max_call_depth=2). Tools can call other tools up to depth limit. src/tools/compose.py: run_tool(tool_name, tool_arguments) meta-tool so the agent can invoke one tool from another (one level).
4. Web UI (FastAPI + Tailwind)
- src/interfaces/api.py: FastAPI app with GET /health, GET /tools, POST /chat (ChatRequest message, thread_id). run_api(host, port); AGENT_API_PORT env. static/index.html: minimal chat UI with Tailwind CDN. Entry point: agent-api (uv run agent-api or uvicorn src.interfaces.api:app --port 8000).
5. Plugin system
- src/tools/plugins.py: load_plugin_module(path or dotted name), load_plugins_from_config(). Config tools.plugins: list of module paths. Plugins register with tool_registry on import. Executor calls load_plugins_from_config() on startup.
6. Advanced memory: episodic vs semantic
- Episodic = recent conversation turns (time-ordered). src/memory/episodic.py: EpisodicMemory(memory_manager, conversation_id, limit) with get_recent_turns(), get_recent_text(). Semantic = long-term vector store (existing VectorStore/ChromaDB). Clear separation for “recent context” vs “stored facts”.
7. A/B testing for prompts
- src/utils/prompts_ab.py: get_prompt_variant(key, variants, variant_id) picks a variant and logs for analysis. get_router_prompt(config) returns router prompt from config.prompts.router list if set. Framework for testing prompt variants per conversation.
8. Dependencies
- fastapi, uvicorn[standard], apscheduler in pyproject.toml.

