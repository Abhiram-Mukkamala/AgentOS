# 🧠 AgentOS: Contextual Multi-Agent Productivity Hub

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)
![Database](https://img.shields.io/badge/Supabase-asyncpg-4CAF50.svg)
![LLM](https://img.shields.io/badge/Gemini-1.5_Flash-8b5cf6.svg)

AgentOS is a stateful, multi-agent orchestrator designed to eliminate the "Parrot Effect" in AI assistants. Instead of returning raw data dumps, specialized agents execute tools and perform a **Synthesis Pass** to provide actionable, context-aware insights.

Built specifically for AI developers managing complex, multi-year AIML roadmaps and high-powered home lab environments.

## 🏗️ Architecture Flow

```mermaid
flowchart LR
    User([User Client]) -->|JSON Payload| API[FastAPI Gateway]

    subgraph Core_Engine [AgentOS Backend - LangGraph]
        direction TB
        API -->|Thread ID| Supervisor{Generalist Supervisor}

        Supervisor -->|Route: Logic| Codex[Codex Engine]
        Supervisor -->|Route: Data| Research[Research Agent]
        Supervisor -->|Route: State| DBAgent[DB Agent]

        Codex -.-> Synthesis[Anti-Parrot Synthesis Loop]
        Research -.-> Synthesis
        DBAgent -.-> Synthesis

        Synthesis -.->|Conversational Output| Supervisor
    end

    subgraph External_Services [External Infrastructure]
        direction TB
        Codex -->|Execute| REPL[Python REPL]
        Research -->|Fetch| Wiki[Wikipedia API]
        DBAgent -->|asyncpg| DB[(Supabase PostgreSQL)]
    end

    Supervisor ===|Reasoning| Gemini((Gemini 1.5 Flash))
    Synthesis ===|Formatting| Gemini

    style Supervisor fill:#d97706,stroke:#b45309,color:#fff
    style Synthesis fill:#059669,stroke:#047857,color:#fff
    style Gemini fill:#2563eb,stroke:#1d4ed8,color:#fff
    style Core_Engine fill:#1e293b,stroke:#475569,color:#e2e8f0
    style DB fill:#0f766e,stroke:#115e59,color:#fff