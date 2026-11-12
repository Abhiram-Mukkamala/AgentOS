"""
main.py — AgentOS FastAPI gateway.

Endpoints:
  POST /chat                 → multi-agent conversation
  POST /run                  → direct code execution (no agent overhead)
  GET  /snippets             → list saved code snippets
  GET  /snippets/{id}        → get single snippet (full code)
  DELETE /snippets/{id}      → delete a snippet
  GET  /improvements         → list logged improvements
  GET  /api/metrics          → live system & user metrics for dashboard
  GET  /health               → health check
  GET  /                     → IDE frontend (index.html)
  GET  /dashboard            → admin dashboard (dashboard.html)
"""

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from agent import AgentState, graph, prune_messages_to_limit
from database import (
    AsyncSessionLocal,
    CodeSnippet,
    Improvement,
    User,
    get_or_create_local_dev_user,
    hash_password,
    init_db,
    verify_password,
)
from memory_manager import save_memory
from settings_manager import estimate_tokens, get_user_settings, update_user_settings
from tools import execute_code

BASE_DIR = Path(__file__).parent

# ── Runtime Metrics Counter ─────────────────────────────────────────────────────

_metrics_date: date = date.today()
_agent_calls_today: dict[int, int] = {}


def _record_agent_call(user_id: int) -> None:
    global _metrics_date, _agent_calls_today
    today = date.today()
    if today != _metrics_date:
        _metrics_date = today
        _agent_calls_today = {}
    _agent_calls_today[user_id] = _agent_calls_today.get(user_id, 0) + 1


def _get_agent_calls_today(user_id: int) -> int:
    global _metrics_date, _agent_calls_today
    if date.today() != _metrics_date:
        return 0
    return _agent_calls_today.get(user_id, 0)


# ── Lifecycle ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="AgentOS", version="2.0.0", lifespan=lifespan)

# ── Agent & Checkpointer ────────────────────────────────────────────────────────

memory = MemorySaver()
multi_agent_system = graph.compile(checkpointer=memory)


def _purge_thread_checkpoints(thread_id_prefix: str) -> int:
    """Remove all in-memory checkpoints matching `thread_id_prefix` from `MemorySaver`."""
    removed_count = 0
    if hasattr(memory, "storage"):
        keys_to_delete = [
            k
            for k in list(memory.storage.keys())
            if (isinstance(k, tuple) and k and k[0] == thread_id_prefix) or k == thread_id_prefix
        ]
        for k in keys_to_delete:
            del memory.storage[k]
            removed_count += 1
    return removed_count


# ── Authentication & JWT Helpers ────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "agentos-jwt-secret-local-only-fallback-12345")


def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def base64url_decode(data: str) -> bytes:
    padding = "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_jwt(payload: dict) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if "exp" not in payload:
        payload["exp"] = int(time.time()) + 86400
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    header_b64 = base64url_encode(header_json)
    payload_b64 = base64url_encode(payload_json)

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_jwt(token: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts

        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
        expected_sig_b64 = base64url_encode(expected_sig)

        if not hmac.compare_digest(signature_b64, expected_sig_b64):
            return None

        payload = json.loads(base64url_decode(payload_b64).decode("utf-8"))
        if "exp" in payload and payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None


async def get_current_user_id(authorization: str | None = Header(None)) -> int:
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()

    if token:
        payload = decode_jwt(token)
        if payload and "sub" in payload:
            return int(payload["sub"])

    is_local_dev = (JWT_SECRET == "agentos-jwt-secret-local-only-fallback-12345") or (
        "sqlite" in os.getenv("DATABASE_URL", "sqlite")
    )
    if is_local_dev:
        user = await get_or_create_local_dev_user()
        return user.id

    raise HTTPException(status_code=401, detail="Unauthorized: authentication token required")


# ── Request / Response Models ───────────────────────────────────────────────────


class SignupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str


class TestKeyRequest(BaseModel):
    provider: str
    key: str


class ChatRequest(BaseModel):
    prompt: str
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class RunRequest(BaseModel):
    code: str


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class RunResponse(BaseModel):
    output: str


class SettingsUpdateRequest(BaseModel):
    active_mode: Optional[str] = None
    selected_providers: Optional[list[str]] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    token_budget: Optional[int] = None
    token_used: Optional[int] = None
    token_warning_threshold: Optional[float] = None
    token_auto_fallback: Optional[bool] = None
    local_max_context_tokens: Optional[int] = None
    local_auto_delete: Optional[bool] = None
    language: Optional[str] = None


# ── Auth Endpoints ──────────────────────────────────────────────────────────────


@app.post("/api/auth/signup", response_model=AuthResponse)
async def signup(req: SignupRequest):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.username == req.username))
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")

        hashed = hash_password(req.password)
        new_user = User(username=req.username, hashed_password=hashed)
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        wp_path = BASE_DIR / "workspaces" / f"user_{new_user.id}"
        wp_path.mkdir(parents=True, exist_ok=True)

        token = create_jwt({"sub": str(new_user.id), "username": new_user.username})
        return AuthResponse(token=token, username=new_user.username)


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.username == req.username))
        user = result.scalar_one_or_none()
        if not user or not verify_password(req.password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Invalid username or password")

        token = create_jwt({"sub": str(user.id), "username": user.username})
        return AuthResponse(token=token, username=user.username)


@app.get("/api/auth/me")
async def me(user_id: int = Depends(get_current_user_id)):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"id": user.id, "username": user.username, "language": user.language}


# ── Locales ─────────────────────────────────────────────────────────────────────


@app.get("/locales/{lang}.json")
async def serve_locale(lang: str):
    path = BASE_DIR / "locales" / f"{lang}.json"
    if not path.exists():
        path = BASE_DIR / "locales" / "en.json"
    if not path.exists():
        return JSONResponse(content={}, status_code=404)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Settings ────────────────────────────────────────────────────────────────────


@app.get("/api/settings")
async def get_settings(user_id: int = Depends(get_current_user_id)):
    res = await get_user_settings(user_id)
    if res.get("openai_api_key"):
        res["openai_api_key"] = "••••••••••••"
    if res.get("anthropic_api_key"):
        res["anthropic_api_key"] = "••••••••••••"
    if res.get("gemini_api_key"):
        res["gemini_api_key"] = "••••••••••••"
    return res


@app.post("/api/settings")
async def update_settings(req: SettingsUpdateRequest, user_id: int = Depends(get_current_user_id)):
    updates = {}
    for k, v in req.model_dump(exclude_unset=True).items():
        if k in ("openai_api_key", "anthropic_api_key", "gemini_api_key") and v == "••••••••••••":
            continue
        updates[k] = v

    await update_user_settings(user_id, updates)
    return {"status": "ok"}


@app.post("/api/settings/test-key")
async def test_key(request: TestKeyRequest, user_id: int = Depends(get_current_user_id)):
    provider = request.provider.lower()
    key = request.key.strip()

    if key == "••••••••••••":
        settings = await get_user_settings(user_id)
        if provider == "openai":
            key = settings.get("openai_api_key", "")
        elif provider == "anthropic":
            key = settings.get("anthropic_api_key", "")
        elif provider == "gemini":
            key = settings.get("gemini_api_key", "")

    if not key or "your_" in key.lower() or "placeholder" in key.lower():
        return {"valid": False, "error": "Invalid API key format."}

    try:
        if provider == "openai":
            resp = httpx.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                return {"valid": True}
            return {"valid": False, "error": resp.json().get("error", {}).get("message", "API request failed.")}
        elif provider == "anthropic":
            headers = {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            data = {
                "model": os.getenv("ANTHROPIC_FAST_MODEL", "claude-3-5-haiku-20241022"),
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}],
            }
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=5.0,
            )
            if resp.status_code == 200:
                return {"valid": True}
            elif resp.status_code == 401:
                return {"valid": False, "error": "Invalid API key."}
            else:
                try:
                    detail = resp.json().get("error", {}).get("message", "API request failed.")
                except Exception:
                    detail = f"Status code {resp.status_code}"
                return {"valid": False, "error": detail}
        elif provider == "gemini":
            resp = httpx.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
                timeout=5.0,
            )
            if resp.status_code == 200:
                return {"valid": True}
            try:
                detail = resp.json().get("error", {}).get("message", "API request failed.")
            except Exception:
                detail = f"Status code {resp.status_code}"
            return {"valid": False, "error": detail}
        return {"valid": False, "error": f"Unknown provider: {provider}"}
    except Exception as e:
        return {"valid": False, "error": f"Connection error: {str(e)}"}


# ── Local Context Management ────────────────────────────────────────────────────


@app.get("/api/local-context/size")
async def get_context_size(thread_id: str, user_id: int = Depends(get_current_user_id)):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    state = await multi_agent_system.aget_state(config)
    messages = state.values.get("messages", [])

    text = "".join(getattr(m, "content", "") or "" for m in messages)
    tokens = estimate_tokens(text)
    return {"tokens": tokens}


@app.post("/api/local-context/prune")
async def force_prune_context(thread_id: str, user_id: int = Depends(get_current_user_id)):
    settings = await get_user_settings(user_id)
    limit = settings.get("local_max_context_tokens", 2048)

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    state = await multi_agent_system.aget_state(config)
    messages = list(state.values.get("messages", []))

    pruned_messages, removed = prune_messages_to_limit(messages, limit, user_id)

    _purge_thread_checkpoints(thread_id)
    if pruned_messages:
        await multi_agent_system.aupdate_state(
            config,
            {"messages": pruned_messages, "next": "Supervisor", "iterations": 0},
        )

    return {"pruned": len(removed), "remaining": len(pruned_messages)}


@app.delete("/api/local-context")
async def delete_local_context(thread_id: str, user_id: int = Depends(get_current_user_id)):
    _purge_thread_checkpoints(thread_id)

    save_memory({
        "files_edited": [],
        "tests": {"passed": 0, "failed": 0, "failing_tests": []},
        "plan_remaining": [],
        "key_decisions": [],
        "errors_seen": [],
        "context_summary": "No prior context.",
    })

    filename = f"pruned_history_{user_id}.json"
    filepath = Path(os.getcwd()) / filename
    if filepath.exists():
        try:
            filepath.unlink()
        except Exception:
            pass

    return {"status": "cleared"}


# ── Dashboard Metrics ───────────────────────────────────────────────────────────


@app.get("/api/metrics")
async def get_metrics(user_id: int = Depends(get_current_user_id)):
    """Return real database and runtime metrics for the current user."""
    async with AsyncSessionLocal() as db:
        snippets_count = (
            await db.scalar(select(func.count(CodeSnippet.id)).where(CodeSnippet.user_id == user_id))
            or 0
        )
        improvements_count = (
            await db.scalar(select(func.count(Improvement.id)).where(Improvement.user_id == user_id))
            or 0
        )
    settings = await get_user_settings(user_id)
    return {
        "snippets_count": int(snippets_count),
        "improvements_count": int(improvements_count),
        "agent_calls_today": _get_agent_calls_today(user_id),
        "token_used": settings["token_used"],
        "token_budget": settings["token_budget"],
        "active_mode": settings["active_mode"],
    }


# ── Routes ──────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: int = Depends(get_current_user_id)):
    if request.prompt.strip().lower() in ("reset", "clear", "new session"):
        return ChatResponse(response="Session cleared.", thread_id=str(uuid.uuid4()))

    _record_agent_call(user_id)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": user_id,
        }
    }
    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.prompt)],
        "next": "Supervisor",
        "iterations": 0,
    }

    try:
        final_state = await multi_agent_system.ainvoke(initial_state, config=config)
        last = final_state["messages"][-1]

        updated_settings = await get_user_settings(user_id)

        warning = None
        if updated_settings["token_used"] >= updated_settings["token_budget"]:
            warning = "⚠️ System Alert: Paid token budget exhausted. Falling back to Local Codex model."
        elif updated_settings["token_used"] >= (
            updated_settings["token_budget"] * updated_settings["token_warning_threshold"]
        ):
            pct = int((updated_settings["token_used"] / updated_settings["token_budget"]) * 100)
            warning = f"⚠️ Token Warning: Budget is {pct}% full. Falling back to local model soon."

        response_text = str(last.content)
        if warning:
            response_text = warning + "\n\n" + response_text

        return ChatResponse(response=response_text, thread_id=request.thread_id)
    except Exception as e:
        err_msg = str(e).lower()
        if "local_context_window_exceeded" in err_msg:
            return ChatResponse(
                response=(
                    "⚠️ **Local Context Window Full**: The model has run out of token context memory. "
                    "Please click 'Prune' in Settings or enable 'Auto-Pruning' to free up space."
                ),
                thread_id=request.thread_id,
            )
        if "paid_api_keys_missing_or_invalid" in err_msg:
            friendly = (
                "⚠️ **API Keys Missing**: Your account is set to **Paid Cloud API** mode in Settings, "
                "but no valid API keys were configured for your selected providers.\n\n"
                "Please configure valid keys or switch API Mode in the **Settings panel** to proceed."
            )
            return ChatResponse(response=friendly, thread_id=request.thread_id)

        if any(k in err_msg for k in ("api key not valid", "api_key_invalid", "invalid api key", "authentication")):
            friendly = (
                "⚠️ **API Key Error**: The API key in your settings is invalid or expired.\n\n"
                "Please configure valid keys in the **Settings panel** to proceed."
            )
            return ChatResponse(response=friendly, thread_id=request.thread_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run", response_model=RunResponse)
async def run_code(request: RunRequest, user_id: int = Depends(get_current_user_id)):
    """Direct Python execution — no agent loop, instant output."""
    _record_agent_call(user_id)
    try:
        result = await execute_code.ainvoke({"code": request.code})
        return RunResponse(output=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snippets")
async def list_snippets(query: str = "", language: str = "", user_id: int = Depends(get_current_user_id)):
    async with AsyncSessionLocal() as db:
        stmt = (
            select(CodeSnippet)
            .where(CodeSnippet.user_id == user_id)
            .order_by(CodeSnippet.created_at.desc())
        )
        result = await db.execute(stmt)
        items = result.scalars().all()

    if query:
        q = query.lower()
        items = [s for s in items if q in s.title.lower() or (s.tags and q in s.tags.lower())]
    if language:
        items = [s for s in items if s.language.lower() == language.lower()]

    return [
        {
            "id": s.id,
            "title": s.title,
            "language": s.language,
            "tags": s.tags,
            "notes": s.notes,
            "preview": s.code[:300],
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in items
    ]


@app.get("/snippets/{snippet_id}")
async def get_snippet(snippet_id: int, user_id: int = Depends(get_current_user_id)):
    """Return a single snippet with full code (not truncated)."""
    async with AsyncSessionLocal() as db:
        stmt = select(CodeSnippet).where(CodeSnippet.id == snippet_id, CodeSnippet.user_id == user_id)
        result = await db.execute(stmt)
        item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(status_code=404, detail="Snippet not found")

    return {
        "id": item.id,
        "title": item.title,
        "language": item.language,
        "code": item.code,
        "tags": item.tags,
        "notes": item.notes,
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }


@app.delete("/snippets/{snippet_id}")
async def delete_snippet(snippet_id: int, user_id: int = Depends(get_current_user_id)):
    async with AsyncSessionLocal() as db:
        stmt = select(CodeSnippet).where(CodeSnippet.id == snippet_id, CodeSnippet.user_id == user_id)
        res = await db.execute(stmt)
        item = res.scalar_one_or_none()
        if not item:
            raise HTTPException(status_code=404, detail="Snippet not found")
        await db.execute(delete(CodeSnippet).where(CodeSnippet.id == snippet_id))
        await db.commit()
    return {"deleted": snippet_id}


@app.get("/improvements")
async def list_improvements(user_id: int = Depends(get_current_user_id)):
    async with AsyncSessionLocal() as db:
        stmt = (
            select(Improvement)
            .where(Improvement.user_id == user_id)
            .order_by(Improvement.created_at.desc())
        )
        result = await db.execute(stmt)
        items = result.scalars().all()

    return [
        {
            "id": i.id,
            "description": i.description,
            "category": i.category,
            "impact": i.impact,
            "created_at": i.created_at.isoformat() if i.created_at else None,
        }
        for i in items
    ]


# ── HTML Pages ──────────────────────────────────────────────────────────────────


def _serve_html(filename: str) -> HTMLResponse:
    path = BASE_DIR / filename
    if not path.exists():
        return HTMLResponse(content=f"<h2>{filename} not found</h2>", status_code=404)
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return _serve_html("index.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    return _serve_html("dashboard.html")


# ── Entry ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
