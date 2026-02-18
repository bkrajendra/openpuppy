# Quick Start for Cursor

## 1. Read These Files First
- [ ] SPEC.md - Complete system specification
- [ ] .cursorrules - Project conventions

## 2. Initial Setup
Run this prompt in Cursor Composer:

"Read SPEC.md and set up Phase 1 project structure.
Create: project directories, pyproject.toml, config files.
Follow .cursorrules conventions."

## 3. Phase 1 Implementation Order
1. AgentState and base types (src/agent/state.py)
2. ToolRegistry skeleton (src/tools/registry.py)
3. LLMProvider interface (src/llm/base.py)
4. MemoryManager (src/memory/manager.py)
5. LangGraph workflow (src/agent/graph.py)
6. First two tools (src/tools/)
7. CLI interface (src/interfaces/cli.py)

## 4. Prompt Template for Each Component
"@SPEC.md Implement [component] following section [X].
Create in [filepath]. Include tests."

## 5. Testing as You Go
After each component: "Create unit tests for [component]"

## 6. Review Checklist
- Type hints on all functions?
- Async/await used properly?
- Pydantic validation added?
- Error handling included?
- Logging statements added?
- Follows .cursorrules style?
```

---

## Summary: Best Approach

1. **Create `.cursorrules`** - Cursor automatically reads this ⭐
2. **Create `SPEC.md`** - Your complete specification
3. **Use explicit prompts** - Reference files with `@filename`
4. **Work incrementally** - One phase/component at a time
5. **Review frequently** - Ask Cursor to validate against spec

**Winning prompt pattern:**
```
@SPEC.md @.cursorrules

Implement [specific component] from SPEC.md section [name].
Place in [filepath].
Follow [specific principle] from .cursorrules.
Include [tests/docs/error handling].