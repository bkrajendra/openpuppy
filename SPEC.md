# Intelligent Autonomous AI Agent: System Design & Implementation Prompt

## Executive Summary

Design and build a production-grade autonomous AI agent system with deterministic orchestration, explicit tool control, and phase-wise implementation starting with a minimum viable agent.

---

## System Architecture

### Core Technology Stack

**Runtime Environment:**
- Python 3.11+ (leverage modern async/await, type hints, pattern matching)
- LangGraph for deterministic state machine orchestration
- Direct SDK integration: OpenAI Python SDK, Anthropic Python SDK
- Local LLM: Ollama with OpenAI-compatible API endpoint

**Orchestration Layer:**
- LangGraph StateGraph for workflow definition
- Explicit node definitions (no magic routing)
- Checkpointing for execution resumption
- Graph compilation with deterministic edge routing

**Memory Architecture:**
- SQLite for structured conversation/task memory (JSONB columns for flexibility)
- Optional ChromaDB/FAISS for semantic search (phase 2+)
- Session management with persistent storage
- Memory schemas: `conversations`, `tool_executions`, `agent_states`, `user_preferences`

**Tools Layer:**
- Explicit tool registry pattern (decorator-based registration)
- Pydantic models for tool input/output validation
- Tool execution sandbox with timeout/retry logic
- Structured tool response format (success/failure/partial)

**Execution Model:**
- Async-first throughout (asyncio event loop)
- Concurrent tool execution where safe
- Streaming LLM responses
- Rate limiting and backoff strategies

---

## Detailed Component Design

### 1. LangGraph Orchestration Model

```
StateGraph Structure:
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Router     │ (Analyze user intent)
└──────┬──────┘
       │
       ├─────────────┬──────────────┬───────────────┐
       ▼             ▼              ▼               ▼
  ┌────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
  │ Direct │   │  Tool   │   │  Multi   │   │Clarify   │
  │Response│   │Executor │   │  Step    │   │Request   │
  └────┬───┘   └────┬────┘   └────┬─────┘   └────┬─────┘
       │            │              │              │
       │            ▼              │              │
       │      ┌──────────┐         │              │
       │      │Tool Loop │         │              │
       │      │(max 5)   │         │              │
       │      └────┬─────┘         │              │
       │           │               │              │
       └───────────┴───────────────┴──────────────┘
                   │
                   ▼
            ┌─────────────┐
            │  Synthesize │
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐
            │     END     │
            └─────────────┘
```

**State Schema:**
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    intent: str  # "direct", "tool_use", "multi_step", "clarification"
    tools_invoked: list[ToolExecution]
    iteration_count: int
    max_iterations: int  # Hard limit (default: 5)
    final_response: str
    metadata: dict
```

**Deterministic Control Flow:**
- Maximum iteration limits enforced at graph level
- Explicit conditional edges (no while-loops in graph)
- Tool execution results directly influence next node routing
- Observable execution trace via checkpointer

---

### 2. Tool Architecture Model

**Tool Definition Pattern:**
```python
@tool_registry.register(
    name="web_search",
    description="Search the web for current information",
    category="information_retrieval"
)
async def web_search(
    query: str = Field(..., description="Search query"),
    max_results: int = Field(5, ge=1, le=10)
) -> ToolResult:
    """
    Pydantic-validated tool function.
    Returns structured ToolResult.
    """
    pass

class ToolResult(BaseModel):
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    execution_time_ms: float
```

**Tool Registry:**
```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
    
    def register(self, name: str, description: str, category: str):
        """Decorator for tool registration"""
        
    def get_tool_schemas(self) -> list[dict]:
        """Return OpenAI/Anthropic function calling schemas"""
        
    async def execute_tool(
        self, 
        name: str, 
        arguments: dict,
        timeout: float = 30.0
    ) -> ToolResult:
        """Execute with timeout, retries, error handling"""
```

**Built-in Tool Categories (Phase 1):**
1. **Information Retrieval:** web_search, wikipedia_lookup
2. **File Operations:** read_file, write_file, list_directory
3. **Code Execution:** run_python_code (sandboxed)
4. **Communication:** send_telegram_message
5. **Memory:** store_memory, retrieve_memory

---

### 3. LLM Integration Layer

**Provider Abstraction:**
```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        self.client = AsyncAnthropic(api_key=api_key)

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")
```

**Reasoning Loop (Inside LangGraph Node):**
```python
async def tool_executor_node(state: AgentState) -> AgentState:
    """
    Node that handles tool execution reasoning.
    Max iterations controlled by state.
    """
    if state["iteration_count"] >= state["max_iterations"]:
        return {
            **state,
            "intent": "synthesize",
            "final_response": "Max iterations reached. Providing partial results."
        }
    
    # LLM decides which tools to call
    response = await llm_provider.generate(
        messages=state["messages"],
        tools=tool_registry.get_tool_schemas()
    )
    
    if response.tool_calls:
        # Execute tools in parallel
        results = await asyncio.gather(*[
            tool_registry.execute_tool(tc.name, tc.arguments)
            for tc in response.tool_calls
        ])
        
        # Add results to messages
        state["messages"].extend(format_tool_results(results))
        state["tools_invoked"].extend(results)
        state["iteration_count"] += 1
        
        # Loop back for another iteration or synthesize
        return state
    else:
        # No more tools needed, move to synthesis
        state["intent"] = "synthesize"
        return state
```

---

### 4. Memory System

**SQLite Schema:**
```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT REFERENCES conversations(id),
    role TEXT CHECK(role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT,
    tool_calls JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tool_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT REFERENCES conversations(id),
    tool_name TEXT NOT NULL,
    arguments JSON,
    result JSON,
    success BOOLEAN,
    execution_time_ms REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE agent_checkpoints (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    state_snapshot JSON,
    graph_position TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Memory Manager:**
```python
class MemoryManager:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db = await aiosqlite.connect(db_path)
        
    async def save_conversation_turn(
        self,
        conversation_id: str,
        messages: list[dict],
        tool_executions: list[ToolResult]
    ):
        """Persist conversation state"""
        
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> list[dict]:
        """Retrieve conversation context"""
        
    async def checkpoint_state(
        self,
        checkpoint_id: str,
        state: AgentState
    ):
        """Save graph execution state for resumption"""
```

---

### 5. Telegram Integration

**Bot Handler:**
```python
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

class TelegramInterface:
    def __init__(self, bot_token: str, agent_executor):
        self.app = Application.builder().token(bot_token).build()
        self.agent = agent_executor
        
    async def handle_message(self, update: Update, context):
        user_message = update.message.text
        user_id = str(update.effective_user.id)
        
        # Create or retrieve conversation
        conversation_id = f"telegram_{user_id}_{int(time.time())}"
        
        # Stream agent response
        async for chunk in self.agent.astream({
            "user_input": user_message,
            "messages": [{"role": "user", "content": user_message}],
            "iteration_count": 0,
            "max_iterations": 5,
            "tools_invoked": [],
            "metadata": {"platform": "telegram", "user_id": user_id}
        }):
            # Send typing indicator or partial responses
            pass
            
        # Send final response
        await update.message.reply_text(chunk["final_response"])
        
    def run(self):
        self.app.add_handler(MessageHandler(filters.TEXT, self.handle_message))
        self.app.run_polling()
```

---

### 6. MCP (Model Context Protocol) Compatibility Layer

**Future Integration Design:**
```python
class MCPAdapter:
    """
    Adapter to make tools MCP-compatible.
    Phase 3+ implementation.
    """
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry
        
    def export_mcp_manifest(self) -> dict:
        """Export tools in MCP format"""
        return {
            "version": "1.0",
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.schema.model_json_schema()
                }
                for tool in self.registry._tools.values()
            ]
        }
        
    async def handle_mcp_request(self, request: dict) -> dict:
        """Handle incoming MCP tool execution requests"""
        pass
```

---

## Phase-Wise Implementation Plan

### **Phase 1: Minimum Working Agent (Weeks 1-2)**

**Deliverables:**
- LangGraph orchestration with 3 nodes: Router → Tool Executor → Synthesizer
- 2 essential tools: `web_search` (DuckDuckGo), `code_executor` (restricted Python)
- SQLite memory for conversation persistence
- OpenAI GPT-4 integration (primary LLM)
- CLI interface for testing
- Maximum 5 iterations enforced
- Basic error handling and logging

**Success Criteria:**
- Agent can answer questions requiring 1-2 tool invocations
- Conversations persist across sessions
- Execution graph is deterministic and observable
- No infinite loops possible

**Implementation Steps:**
1. Set up project structure with Poetry/UV
2. Implement `AgentState` and basic LangGraph workflow
3. Create `ToolRegistry` with 2 tools
4. Build `MemoryManager` with SQLite
5. Integrate OpenAI SDK with function calling
6. Create CLI runner with conversation management
7. Add comprehensive logging and error handling

---

### **Phase 2: Enhanced Capabilities (Weeks 3-4)**

**Deliverables:**
- Telegram bot interface
- Ollama local LLM support
- 5 additional tools: file operations, Wikipedia, calculator, weather API
- Anthropic Claude integration
- Vector database for semantic memory (ChromaDB)
- Streaming responses
- Parallel tool execution
- Configuration management (YAML/environment variables)

**Success Criteria:**
- Agent accessible via Telegram
- Works fully offline with Ollama
- Can handle multi-tool workflows (3-5 tools)
- Sub-second response start with streaming

**Implementation Steps:**
1. Add Telegram bot with `python-telegram-bot`
2. Implement `OllamaProvider` with model management
3. Build additional tools with strong schemas
4. Add ChromaDB for long-term memory search
5. Implement streaming throughout the pipeline
6. Create configuration system
7. Add provider fallback logic

---

### **Phase 3: Production Hardening (Weeks 5-6)**

**Deliverables:**
- MCP adapter for tool exposure
- Advanced graph patterns: human-in-the-loop, approval gates
- Monitoring and observability (Prometheus metrics)
- Rate limiting per user/conversation
- Tool sandboxing and security
- Multi-conversation management
- Backup/restore functionality
- Docker deployment setup

**Success Criteria:**
- Can expose tools via MCP to other systems
- Handles 100+ concurrent conversations
- Graceful degradation when LLMs unavailable
- Full execution audit trail

---

### **Phase 4: Advanced Features (Weeks 7-8)**

**Deliverables:**
- Multi-agent collaboration (supervisor pattern)
- Scheduled/cron-based autonomous tasks
- Tool composition (tools calling tools)
- Web UI (FastAPI + Angular + Tailwind CSS)
- Plugin system for third-party tools
- Advanced memory: episodic vs semantic separation
- A/B testing framework for prompts

**Success Criteria:**
- Multiple specialized agents working together
- Agent can execute recurring tasks autonomously
- Non-technical users can add tools via UI

---

## Implementation Guidelines

### Project Structure
```
intelligent-agent/
├── pyproject.toml
├── README.md
├── .env.example
├── config/
│   ├── agent_config.yaml
│   └── tools_config.yaml
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py          # LangGraph definition
│   │   ├── nodes.py          # Individual node functions
│   │   ├── state.py          # AgentState definitions
│   │   └── executor.py       # Main agent executor
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py           # LLMProvider ABC
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── ollama.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py       # ToolRegistry
│   │   ├── base.py           # ToolResult, base classes
│   │   ├── web_search.py
│   │   ├── code_executor.py
│   │   └── file_operations.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── manager.py        # MemoryManager
│   │   ├── schemas.sql
│   │   └── vector_store.py   # Optional ChromaDB
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── telegram_bot.py
│   │   └── api.py            # Future FastAPI
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── config.py
│       └── monitoring.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
│   ├── setup_db.py
│   └── run_agent.py
└── data/
    ├── agent_memory.db
    └── vector_store/
```

### Key Design Principles

1. **Determinism First:** Every execution path is traceable and reproducible
2. **Explicit Over Implicit:** No magic; every tool call is intentional and logged
3. **Async Throughout:** Never block the event loop
4. **Schema Validation:** Pydantic models everywhere for type safety
5. **Fail Gracefully:** Timeouts, retries, and fallbacks at every layer
6. **Observable:** Logs, metrics, and execution graphs for debugging
7. **Local-First:** Works offline, cloud is an enhancement
8. **Extensible:** Add tools/LLMs/interfaces without core rewrites

### Error Handling Strategy

- **LLM failures:** Retry with exponential backoff, fallback to alternate provider
- **Tool failures:** Return structured error in ToolResult, allow agent to adapt
- **Timeout violations:** Hard kill at node level, return partial results
- **Invalid schemas:** Reject at Pydantic validation, request correction
- **Database errors:** Queue writes, retry on connection issues

### Observability

```python
# Structured logging
import structlog

logger = structlog.get_logger()
logger.info(
    "tool_executed",
    tool_name=tool_name,
    execution_time_ms=result.execution_time_ms,
    success=result.success
)

# Prometheus metrics
from prometheus_client import Counter, Histogram

tool_executions = Counter('agent_tool_executions_total', 'Total tool executions', ['tool_name', 'status'])
llm_latency = Histogram('agent_llm_latency_seconds', 'LLM request duration')
```

---

## Configuration Example

**config/agent_config.yaml:**
```yaml
agent:
  name: "IntelligentAgent"
  max_iterations: 5
  timeout_seconds: 120
  
llm:
  primary:
    provider: "openai"
    model: "gpt-4-turbo-preview"
    temperature: 0.7
    max_tokens: 2000
  fallback:
    provider: "ollama"
    model: "mistral:7b"
    
tools:
  enabled:
    - web_search
    - code_executor
    - file_operations
  sandboxing:
    code_executor:
      max_execution_time: 10
      allowed_imports: ["math", "statistics", "datetime"]
      
memory:
  database_path: "./data/agent_memory.db"
  vector_store:
    enabled: false
    path: "./data/vector_store"
    
interfaces:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
  api:
    enabled: false
    port: 8000
```

---

## Testing Strategy

**Unit Tests:**
- Individual tool execution with mocked dependencies
- LangGraph node functions with fixed state inputs
- Memory operations with in-memory SQLite

**Integration Tests:**
- Full agent execution on predefined scenarios
- Multi-tool workflows with real LLM calls (cached)
- Telegram bot message handling

**Benchmark Tests:**
- Agent response latency (p50, p95, p99)
- Concurrent conversation handling
- Memory query performance

---

## Security Considerations

1. **Code Execution Sandbox:** Use `RestrictedPython` or Docker containers
2. **Input Validation:** Sanitize all user inputs and tool arguments
3. **API Key Management:** Environment variables, never in code
4. **Rate Limiting:** Per-user limits on tool executions and LLM calls
5. **Audit Logging:** All tool executions and LLM interactions logged
6. **File Access Controls:** Restrict file operations to designated directories

---

## Success Metrics

**Phase 1:**
- Agent successfully completes 90%+ of single-tool tasks
- Zero infinite loops in 1000 test runs
- Conversation persistence 100% reliable

**Phase 2:**
- Telegram bot uptime >99%
- Local LLM response quality within 10% of cloud models
- Streaming response start <500ms

**Phase 3:**
- 100 concurrent users without degradation
- Full execution trace for every run
- MCP compatibility verified with external systems

**Phase 4:**
- Multi-agent workflows complete complex tasks
- Non-developers successfully add custom tools
- Plugin ecosystem with 10+ community tools

---

## Next Steps

1. **Initialize Project:** Set up Python 3.11+ environment with Poetry
2. **Spike LangGraph:** Build minimal 3-node graph with hardcoded logic
3. **Prototype Tool Registry:** Implement registration and one working tool
4. **Database Setup:** Create SQLite schema and basic CRUD
5. **LLM Integration:** Connect OpenAI SDK with function calling
6. **End-to-End Test:** Run first complete conversation with one tool call

**First Milestone Target:** Working CLI agent that can search the web and answer questions (1-2 weeks).

---

This prompt provides a complete blueprint for building a production-ready autonomous agent system with:
- ✅ Deterministic orchestration (LangGraph state machines)
- ✅ Explicit tool control (no magic, maximum iteration limits)
- ✅ Strong typing and validation (Pydantic throughout)
- ✅ Async-first architecture (non-blocking I/O)
- ✅ Observable execution (logs, metrics, graph visualization)
- ✅ Local-first design (Ollama support, SQLite)
- ✅ Extensible architecture (tool registry, provider abstraction)
- ✅ Phase-wise implementation (MVP in 2 weeks)
- ✅ Telegram integration roadmap
- ✅ MCP compatibility path

Begin with Phase 1 to validate core architecture, then iterate based on real-world usage patterns.