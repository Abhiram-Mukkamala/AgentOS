import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import delete

# Ensure agentos-v2 directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database import AsyncSessionLocal, User, init_db
from main import app, multi_agent_system
from settings_manager import add_user_tokens


async def _cleanup_test_user(username: str) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(delete(User).where(User.username == username))
        await session.commit()


async def run_tests():
    await init_db()
    test_username = "test_user_unique_123"
    await _cleanup_test_user(test_username)

    client = TestClient(app)

    # 1. Test Signup
    signup_data = {
        "username": test_username,
        "password": "test_password_123",
    }
    resp = client.post("/api/auth/signup", json=signup_data)
    assert resp.status_code == 200, f"Signup failed: {resp.text}"
    data = resp.json()
    assert "token" in data
    assert data["username"] == test_username
    token = data["token"]

    # 2. Test Signup with duplicate username (should fail with 400)
    resp = client.post("/api/auth/signup", json=signup_data)
    assert resp.status_code == 400, "Expected duplicate signup to fail with 400"

    # 3. Test Login
    resp = client.post("/api/auth/login", json=signup_data)
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    data = resp.json()
    assert "token" in data
    assert data["username"] == test_username

    # 4. Test /api/auth/me with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.get("/api/auth/me", headers=headers)
    assert resp.status_code == 200
    me_data = resp.json()
    assert me_data["username"] == test_username
    user_id = me_data["id"]

    # 5. Test settings retrieval
    resp = client.get("/api/settings", headers=headers)
    assert resp.status_code == 200
    settings = resp.json()
    assert settings["active_mode"] == "free"

    # 6. Test settings update & persistence
    update_data = {
        "active_mode": "paid",
        "selected_providers": ["openai", "gemini"],
        "token_budget": 1000,
        "local_max_context_tokens": 50,
    }
    resp = client.post("/api/settings", json=update_data, headers=headers)
    assert resp.status_code == 200

    resp = client.get("/api/settings", headers=headers)
    settings = resp.json()
    assert settings["active_mode"] == "paid"
    assert settings["selected_providers"] == ["openai", "gemini"]
    assert settings["token_budget"] == 1000
    assert settings["local_max_context_tokens"] == 50

    # 7. Test atomic single-query add_user_tokens & threshold evaluation
    res1 = await add_user_tokens(user_id, 920)
    assert res1["token_used"] == 920
    assert res1["token_budget"] == 1000
    assert res1["warning_reached"] is True
    assert res1["budget_exhausted"] is False
    assert res1["active_mode"] == "paid"

    res2 = await add_user_tokens(user_id, 130)
    assert res2["token_used"] == 1050
    assert res2["budget_exhausted"] is True
    assert res2["active_mode"] == "local"  # auto-fallback triggered atomically

    # 8. Test /api/metrics endpoint
    resp = client.get("/api/metrics", headers=headers)
    assert resp.status_code == 200
    metrics = resp.json()
    assert "agent_calls_today" in metrics
    assert "snippets_count" in metrics
    assert "improvements_count" in metrics
    assert metrics["token_used"] == 1050
    assert metrics["token_budget"] == 1000

    # 9. Test local context size & force_prune_context (via aupdate_state, no LLM invocation)
    thread_id = "test-thread-id-999"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    await multi_agent_system.aupdate_state(
        config,
        {
            "messages": [
                HumanMessage(content=" ".join(["alpha"] * 60), id="msg-1"),
                AIMessage(content=" ".join(["beta"] * 60), id="msg-2"),
                HumanMessage(content=" ".join(["gamma"] * 10), id="msg-3"),
            ]
        },
    )

    resp = client.get(f"/api/local-context/size?thread_id={thread_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["tokens"] > 50

    resp = client.post(f"/api/local-context/prune?thread_id={thread_id}", headers=headers)
    assert resp.status_code == 200
    prune_data = resp.json()
    assert prune_data["pruned"] >= 1
    assert prune_data["remaining"] >= 1

    # Clean up pruned history file & checkpoints
    resp = client.delete(f"/api/local-context?thread_id={thread_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"

    mem_file = Path("agentos_memory.json")
    if mem_file.exists():
        mem_file.unlink()

    await _cleanup_test_user(test_username)


def test_auth_and_settings_suite():
    asyncio.run(run_tests())


if __name__ == "__main__":
    asyncio.run(run_tests())
