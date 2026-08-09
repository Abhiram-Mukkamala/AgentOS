# 🧠 AgentOS: LangGraph Multi-Agent Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-8b5cf6.svg)

AgentOS is a stateful multi-agent system built with FastAPI and LangGraph. It routes user requests from a central Supervisor agent to specialized sub-agents that execute tools for Python execution, Wikipedia research, and task database storage.

## 🏗️ Architecture & How It Works

```mermaid
flowchart LR
    User([User / Browser]) -->|HTTP / JSON| API[FastAPI Gateway]

    subgraph Core [LangGraph Core]
        API -->|Prompt & Thread ID| Supervisor{Generalist Supervisor}
        Supervisor -->|Route: Code| Codex[CodexEngine]
        Supervisor -->|Route: Research| Research[ResearchAgent]
        Supervisor -->|Route: Database| DBAgent[DBAgent]

        Codex -.->|Tool Output| Supervisor
        Research -.->|Tool Output| Supervisor
        DBAgent -.->|Tool Output| Supervisor
    end

    subgraph Tools [Tools & Infrastructure]
        Codex --> REPL[Python REPL]
        Research --> Wiki[Wikipedia API]
        DBAgent --> DB[(SQLAlchemy Database)]
    end

    Supervisor ===|LLM| Gemini((Gemini 2.0 Flash))
```

### Agents & Tools Currently Implemented

- **Generalist Supervisor**: Analyzes incoming messages and routes requests to the appropriate specialist (`CodexEngine`, `ResearchAgent`, `DBAgent`) or produces final responses (`FINISH`).
- **CodexEngine**: Uses `python_execution_tool` (LangChain PythonREPL) to run Python code for math and logic validation.
- **ResearchAgent**: Uses `research_tool` (Wikipedia API) to fetch documentation and summary snippets.
- **DBAgent**: Uses `add_database_task` to insert tasks (description, status, due date) into PostgreSQL or SQLite databases via SQLAlchemy async sessions.

---

## 🛠️ Installation & Running

### Prerequisites

- Python 3.11+
- Google Gemini API key (`GEMINI_API_KEY`)

### Quickstart (Local)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abhiram-Mukkamala/AgentOS.git
   cd AgentOS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   # Optional: default database is local SQLite (sqlite+aiosqlite:///./local_dev.db)
   # export DATABASE_URL="postgresql://user:password@host:port/dbname"
   ```

4. **Run the FastAPI server**:
   ```bash
   python main.py
   # or
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```

5. **Access the application**:
   - Web UI: Open `http://localhost:8080` in your browser.
   - API Endpoint: `POST http://localhost:8080/chat` with JSON body:
     ```json
     {
       "prompt": "Calculate 15 * 87 using python"
     }
     ```

### Running with Docker

1. **Build the image**:
   ```bash
   docker build -t agentos .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8080:8080 -e GEMINI_API_KEY="your-gemini-api-key" agentos
   ```

---

## ⚠️ Current Limitations

- **LLM Model**: Configured for `gemini-2.0-flash`. Requires a valid `GEMINI_API_KEY`.
- **In-Memory Checkpoints**: Conversation state uses `MemorySaver` in-memory checkpointer; state is reset when the server restarts.
- **REPL Execution Sandbox**: `PythonREPL` runs Python code in the local execution context without isolated container sandboxing.
- **Fixed Thread ID Default**: The web interface uses generated thread IDs per request session; hard reset (`reset` / `clear`) clears the active thread session.

---

## 🚧 In Progress

- **[In Progress] Frontend Task Sidebar**: The single-page UI contains a task memory sidebar. Backend endpoints (`GET /tasks` or `/demo/tasks`) to list and retrieve saved database tasks for the frontend are currently in progress.
- **[In Progress] Advanced Checkpoint Persistence**: Replacing in-memory session saving with persistent database checkpointers across server restarts.
- **[In Progress] Multi-Step Task Management**: Querying, updating, and deleting stored tasks via conversational agent commands.