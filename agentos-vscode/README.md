# AgentOS VS Code Extension

AgentOS is a multi-agent orchestration framework designed for developer productivity. This VS Code extension embeds the AgentOS developer interface directly inside your sidebar.

## Features

- **Sidebar Agent Chat**: Chat with the supervisor agent and worker nodes (Codex, Research, DB) directly inside your VS Code window.
- **Auto-spawning Backend**: Spawns the FastAPI Python server automatically inside a dedicated VS Code Terminal when you launch.
- **DB-Backed Settings & Deep Fallback Routing**: Manage API mode (`paid`, `free`, `local`), provider selection (`openai`, `anthropic`, `gemini`), API keys, and token budgets in the AgentOS Settings UI (persisted in SQLite/PostgreSQL with `.env` fallback). Automatically falls back to free providers or the local Codex GGUF engine when configured.
- **Dynamic Port Configuration**: Configure `agentos.backendPort` in VS Code Settings to dynamically update both the backend server and the sidebar webview target.
- **Click-to-Open Files**: Click on file links in the chat response to open them directly in your VS Code workspace editor.

## Getting Started

### 1. Requirements
Ensure you have the Python environment configured and all backend dependencies installed in the `agentos-v2` directory:
```bash
cd agentos-v2
pip install -r requirements.txt
```

### 2. Configure API Keys & Mode
You can configure API keys and model routing either via the **AgentOS Web UI Settings modal** (`http://localhost:8080`) or by copying `agentos-v2/.env.example` to `agentos-v2/.env` and setting `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`.

### 3. Debugging / Running in Development
1. Open this workspace (`c:\Projects\agentos`) in VS Code.
2. Open `agentos-vscode/extension.js` or `agentos-vscode/package.json`.
3. Press `F5` (or click **Run and Debug** -> **Launch Extension**).
4. A new VS Code Window (Extension Development Host) will open.
5. In the new window, click the **AgentOS** icon in the Activity Bar (left sidebar) to open the chat interface.

## Commands

- `AgentOS: Start Backend Server`: Manually start the AgentOS Python server.
- `AgentOS: Stop Backend Server`: Manually stop the running AgentOS server terminal.
- `AgentOS: Configure Backend & Settings`: Open VS Code settings for AgentOS (`agentos.backendPort`, `agentos.autoStartBackend`).
- `AgentOS: Open Chat View`: Focus the AgentOS sidebar chat view.
