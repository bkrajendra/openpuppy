"""FastAPI Web API for the agent (Phase 4) + Admin dashboard API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.executor import run_agent, create_agent, _initial_state
from src.utils.llm_factory import get_llm_from_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")

app = FastAPI(title="Intelligent Agent API", version="0.1.0")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "web_default"


class ChatResponse(BaseModel):
    response: str
    thread_id: str


# --- Admin: custom tool add request ---
class AddToolRequest(BaseModel):
    name: str
    description: str = ""
    type: str = "http"
    url: str = ""
    method: str = "POST"
    parameters_schema: dict[str, Any] = Field(default_factory=lambda: {"properties": {}, "required": []})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
async def list_tools() -> dict[str, Any]:
    from src.tools.registry import tool_registry
    schemas = tool_registry.get_tool_schemas()
    return {"tools": [s["function"]["name"] for s in schemas], "schemas": schemas}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        from langgraph.checkpoint.memory import MemorySaver
        llm = get_llm_from_config()
        agent = create_agent(llm=llm, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": req.thread_id}}
        initial = _initial_state(req.message, metadata={"platform": "web"})
        state = await agent.ainvoke(initial, config=config)
        response = state.get("final_response", "").strip() or "No response."
        return ChatResponse(response=response, thread_id=req.thread_id)
    except Exception as e:
        logger.exception("api_chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# --- Admin API (dashboard: status, tools, agents, config) ---

@app.get("/admin/status")
async def admin_status() -> dict[str, Any]:
    """Status overview: health, tool count, agents (teams)."""
    from src.tools.registry import tool_registry
    from src.agent import supervisor as sup
    schemas = tool_registry.get_tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    custom = [n for n in names if tool_registry.is_custom(n)]
    return {
        "status": "ok",
        "tools_total": len(names),
        "tools_builtin": len(names) - len(custom),
        "tools_custom": len(custom),
        "agents_teams": list(sup.TEAM_TOOLS.keys()),
    }


@app.get("/admin/agents")
async def admin_agents() -> dict[str, Any]:
    """List agent teams (supervisor) and which tools each can use."""
    from src.agent import supervisor as sup
    return {"teams": sup.TEAM_TOOLS}


@app.get("/admin/tools")
async def admin_tools_list() -> dict[str, Any]:
    """List all tools with source (builtin vs custom)."""
    from src.tools.registry import tool_registry
    from src.tools.custom_tools import load_custom_tools
    schemas = tool_registry.get_tool_schemas()
    custom_defs = {t["name"]: t for t in load_custom_tools()}
    tools = []
    for s in schemas:
        name = s["function"]["name"]
        tools.append({
            "name": name,
            "description": s["function"].get("description", ""),
            "source": "custom" if tool_registry.is_custom(name) else "builtin",
            "definition": custom_defs.get(name),
        })
    return {"tools": tools}


@app.post("/admin/tools")
async def admin_tools_add(req: AddToolRequest) -> dict[str, Any]:
    """Add a custom tool (HTTP type) from the UI. Non-technical users can add tools here."""
    from src.tools.custom_tools import add_custom_tool
    if req.type != "http" or not req.url.strip():
        raise HTTPException(status_code=400, detail="type must be 'http' and url is required")
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        definition = add_custom_tool({
            "name": name,
            "description": req.description or "",
            "type": "http",
            "url": req.url.strip(),
            "method": (req.method or "POST").upper(),
            "parameters_schema": req.parameters_schema or {"properties": {}, "required": []},
        })
        return {"ok": True, "tool": definition}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/admin/tools/{name}")
async def admin_tools_remove(name: str) -> dict[str, Any]:
    """Remove a custom tool by name. Built-in tools cannot be removed."""
    from src.tools.custom_tools import remove_custom_tool
    if remove_custom_tool(name):
        return {"ok": True, "removed": name}
    raise HTTPException(status_code=404, detail="Tool not found or not removable (built-in)")


@app.get("/admin/config")
async def admin_config_get() -> dict[str, Any]:
    """Return full agent config (YAML as dict) for editing in UI."""
    from src.utils.config import load_config
    return load_config()


@app.patch("/admin/config")
async def admin_config_patch(body: dict[str, Any]) -> dict[str, Any]:
    """Persist config from the UI. Body is the full config object; file is replaced so the saved file matches the editor."""
    from src.utils.config import save_config
    save_config(body)
    return body


# Serve minimal frontend from same process if static dir exists
_static = _project_root / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_html = _static / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        return HTMLResponse("<p>Agent API. Use POST /chat or see /docs.</p>")

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_dashboard():
        admin_html = _static / "admin.html"
        if admin_html.exists():
            return FileResponse(admin_html)
        return HTMLResponse("<p>Admin dashboard not found. Add static/admin.html</p>")


def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI app with uvicorn. Entry point for agent-api script."""
    import uvicorn
    port = int(os.getenv("AGENT_API_PORT", port))
    uvicorn.run(app, host=host, port=port)
