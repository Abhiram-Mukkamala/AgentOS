# AgentOS VS Code Extension

AgentOS is a multi-agent orchestration framework designed for developer productivity. This VS Code extension embeds the AgentOS developer interface directly inside your sidebar.

## Features

- **Sidebar Agent Chat**: Chat with the supervisor agent and worker nodes (Codex, Research, DB) directly inside your VS Code window.
- **Auto-spawning Backend**: Spawns the FastAPI python server automatically inside a dedicated VS Code Terminal when you launch.
- **Deep Fallback Routing**: Intelligently routes calls between Gemini, Claude, and OpenAI depending on active API keys in `.env`, falling back to free g4f versions and local model (Qwen) if cloud providers are unreachable.
- **Click-to-Open Files**: Click on file links in the chat response to open them directly in your VS Code workspace editor.

## Getting Started

### 1. Requirements
Ensure you have the Python environment configured and all backend dependencies installed in the `agentos-v2` directory:
```bash
cd agentos-v2
pip install -r requirements.txt
```

### 2. Configure Keys
Open the `agentos-v2/.env` file and set your desired API keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). If no keys are set, AgentOS will automatically fallback to free keyless models or launch the local embedded Codex engine.

### 3. Debugging / Running in Development
1. Open this workspace (`c:\Projects\agentos`) in VS Code.
2. Open `extension.js` or `package.json`.
3. Press `F5` (or click **Run and Debug** -> **Launch Extension**).
4. A new VS Code Window (Extension Development Host) will open.
5. In the new window, click the **AgentOS** icon in the Activity Bar (left sidebar) to open the chat interface. The FastAPI backend will start running automatically in the background terminal!

## Commands

- `AgentOS: Start Backend Server`: Manually start the AgentOS python server.
- `AgentOS: Stop Backend Server`: Manually stop the running AgentOS server terminal.
- `AgentOS: Configure API Keys`: Open VS Code settings for AgentOS.
