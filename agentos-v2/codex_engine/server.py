"""
server.py — OpenAI-compatible API server for AgentOS Codex Engine.

Endpoints:
    POST /v1/chat/completions   → chat with tool calling
    GET  /v1/models             → list loaded model
    GET  /health                → engine health + hardware stats

Can run:
    - Standalone:  python -m codex_engine.server
    - Embedded:    start_codex_server(background=True)  (inside main.py)
"""

import logging
import os
import threading
import time
import uuid
from typing import Any, Optional

log = logging.getLogger("codex_engine.server")

# Lazy-initialised global engine instance
_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    """Get or create the singleton CodexEngine instance."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from codex_engine.engine import CodexEngine
                force_size = os.getenv("LOCAL_CODEX_FORCE_SIZE", "").strip() or None
                model_path = os.getenv("LOCAL_CODEX_MODEL_PATH", "").strip() or None
                _engine = CodexEngine(model_path=model_path, force_size=force_size)
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════════


def _create_app():
    """Create the FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="AgentOS Codex Engine",
        version="1.0.0",
        description="Self-contained local LLM with hardware auto-scaling and SSD overflow.",
    )

    # ── Request / Response models ───────────────────────────────────────────

    class ChatMessage(BaseModel):
        role: str
        content: Optional[str] = None
        tool_calls: Optional[list[dict]] = None
        tool_call_id: Optional[str] = None
        name: Optional[str] = None

    class ChatCompletionRequest(BaseModel):
        model: str = "local"
        messages: list[dict]
        tools: Optional[list[dict]] = None
        tool_choice: Optional[Any] = "auto"
        temperature: float = 0.1
        max_tokens: int = 4096
        stream: bool = False

    # ── Routes ──────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        """Engine health check with hardware stats."""
        try:
            engine = _get_engine()
            return engine.health()
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "detail": str(e)},
            )

    @app.get("/v1/models")
    async def list_models():
        """List loaded models (OpenAI-compatible)."""
        try:
            engine = _get_engine()
            h = engine.health()
            return {
                "object": "list",
                "data": [
                    {
                        "id": h.get("model", "local-codex"),
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "agentos-local",
                    }
                ],
            }
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"error": str(e)},
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completion with tool calling."""
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not yet supported by the local engine.",
            )

        try:
            engine = _get_engine()
            response = engine.chat_completion(
                messages=request.messages,
                tools=request.tools,
                tool_choice=request.tool_choice,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            return response

        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            log.error("Chat completion error: %s", repr(e))
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# Server Launchers
# ═══════════════════════════════════════════════════════════════════════════════


def start_codex_server(
    host: str = "127.0.0.1",
    port: int = 8090,
    background: bool = False,
):
    """
    Start the Codex Engine API server.

    Args:
        host:       Bind address (default: 127.0.0.1)
        port:       Port number (default: 8090)
        background: If True, starts in a daemon thread and returns immediately.
                    If False, blocks until the server is stopped.
    """
    import uvicorn

    port = int(os.getenv("LOCAL_CODEX_PORT", str(port)))
    host = os.getenv("LOCAL_CODEX_HOST", host)

    app = _create_app()

    if background:
        # Pre-load the engine in the main thread so errors are caught early
        log.info("Pre-loading Codex Engine...")
        try:
            _get_engine()
        except Exception as e:
            log.error("Failed to start Codex Engine: %s", e)
            raise

        def _run():
            config = uvicorn.Config(
                app, host=host, port=port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            server.run()

        thread = threading.Thread(target=_run, daemon=True, name="codex-engine")
        thread.start()

        # Wait for the server to be ready
        import httpx
        for _ in range(30):
            try:
                resp = httpx.get(f"http://{host}:{port}/health", timeout=1.0)
                if resp.status_code == 200:
                    log.info(
                        "Codex Engine server running at http://%s:%d",
                        host, port,
                    )
                    return
            except Exception:
                time.sleep(0.5)

        log.warning("Codex Engine server started but health check timed out.")

    else:
        # Blocking (standalone) mode
        print()
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║   AgentOS Codex Engine — Standalone Server       ║")
        print(f"  ║   Listening on http://{host}:{port}           ║")
        print("  ╚══════════════════════════════════════════════════╝")
        print()
        print("  Endpoints:")
        print(f"    POST http://{host}:{port}/v1/chat/completions")
        print(f"    GET  http://{host}:{port}/v1/models")
        print(f"    GET  http://{host}:{port}/health")
        print()

        uvicorn.run(app, host=host, port=port, log_level="info")


# ═══════════════════════════════════════════════════════════════════════════════
# __main__ — standalone entry point
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    start_codex_server(background=False)
