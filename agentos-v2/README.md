# 🧠 AgentOS v2: Hybrid Edge-to-Kernel Multi-Agent Developer Platform

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)
![Database](https://img.shields.io/badge/SQLAlchemy-Async_SQLite_%2F_Postgres-4CAF50.svg)

AgentOS v2 is a stateful, multi-agent developer productivity platform combining a **Tier-0 Edge Reflex Router** with a **Senior–Junior LangGraph Kernel**, persistent workspace tools, a browser-based Monaco IDE, an analytics dashboard, and a self-contained local GGUF inference engine (**Codex Engine**).

## 🏗️ Architecture Flow

```mermaid
flowchart LR
    User([Web IDE / VS Code]) -->|HTTP + Bearer JWT| API[FastAPI Gateway]

    API -->|Tier-0 Check| Edge{Edge Reflex Router}
    Edge -->|Single-Hop Tool| FastExec[Direct Tool Dispatch]
    Edge -->|Escalate| Supervisor{Senior Supervisor}

    subgraph Kernel [LangGraph Multi-Agent Kernel]
        direction TB
        Supervisor -->|Route: Code & FS| Codex[Codex Agent]
        Supervisor -->|Route: Web & Docs| Research[Research Agent]
        Supervisor -->|Route: Snippets & Logs| DBAgent[DB Agent]

        Codex --> Supervisor
        Research --> Supervisor
        DBAgent --> Supervisor
    end

    subgraph Persistence [Storage & Execution]
        direction TB
        Codex -->|Isolated Workspace| FS[File System / Pytest / AST]
        Research -->|DuckDuckGo + BS4| Web[Web Search & Scraper]
        DBAgent -->|SQLAlchemy Async| DB[(SQLite / PostgreSQL)]
    end
```

## ✨ Key Capabilities

- **Tier-0 Edge Reflex Router (`edge_router.py`)**: Model-agnostic fast path supporting deterministic rules, local embeddings, ONNX token taggers, or micro-LLMs with margin & span-entropy gating before escalating to the LangGraph kernel.
- **Senior–Junior Model Hierarchy (`agent.py`)**: Configurable cloud providers (`OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL`, `SENIOR_MODEL_PROVIDER`), free web fallback (`g4f`), and local GGUF inference (`codex_engine/`).
- **Per-User Auth, Token Budgets & Context Pruning (`database.py`, `settings_manager.py`)**: PBKDF2-HMAC-SHA256 authentication, atomic SQL token budget tracking with automatic local fallback, and LangGraph checkpoint pruning.
- **Self-Contained Local Codex Engine (`codex_engine/`)**: Hardware-aware (`hardware.py`) GGUF model loader with GPU layer offloading, SSD `mmap` overflow, and idle VRAM unloading.
