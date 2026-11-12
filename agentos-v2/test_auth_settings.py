import asyncio
import httpx
import os
import sys
from fastapi.testclient import TestClient

# Add current dir to path to import app and database
sys.path.append(os.getcwd())

from main import app
from database import init_db

async def run_tests():
    # Force initialize db
    await init_db()
    
    # We will use FastAPI TestClient
    client = TestClient(app)
    
    print("\n--- Running FastAPI Auth & Settings Integration Tests ---")
    
    # 1. Test Signup
    signup_data = {
        "username": "test_user_unique_123",
        "password": "test_password_123"
    }
    resp = client.post("/api/auth/signup", json=signup_data)
    print(f"Signup response status: {resp.status_code}")
    assert resp.status_code == 200, f"Signup failed: {resp.text}"
    data = resp.json()
    assert "token" in data
    assert data["username"] == "test_user_unique_123"
    token = data["token"]
    print("SUCCESS: Signup endpoint verified!")
    
    # 2. Test Signup with duplicate username (should fail)
    resp = client.post("/api/auth/signup", json=signup_data)
    print(f"Duplicate signup response status: {resp.status_code}")
    assert resp.status_code == 400, "Expected duplicate signup to fail with 400"
    print("SUCCESS: Duplicate username signup prevention verified!")
    
    # 3. Test Login
    login_data = {
        "username": "test_user_unique_123",
        "password": "test_password_123"
    }
    resp = client.post("/api/auth/login", json=login_data)
    print(f"Login response status: {resp.status_code}")
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    data = resp.json()
    assert "token" in data
    assert data["username"] == "test_user_unique_123"
    print("SUCCESS: Login endpoint verified!")
    
    # 4. Test me endpoint with auth header
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.get("/api/auth/me", headers=headers)
    print(f"Get /me response status: {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "test_user_unique_123"
    print("SUCCESS: Authenticated /me endpoint verified!")
    
    # 5. Test settings retrieval
    resp = client.get("/api/settings", headers=headers)
    print(f"Get settings response status: {resp.status_code}")
    assert resp.status_code == 200
    settings = resp.json()
    assert settings["active_mode"] == "free"
    print("SUCCESS: Get settings verified!")
    
    # 6. Test settings update
    update_data = {
        "active_mode": "paid",
        "selected_providers": ["openai", "gemini"],
        "token_budget": 500000,
        "local_max_context_tokens": 4096
    }
    resp = client.post("/api/settings", json=update_data, headers=headers)
    print(f"Update settings response status: {resp.status_code}")
    assert resp.status_code == 200
    print("SUCCESS: Post settings update verified!")
    
    # Verify updates persisted
    resp = client.get("/api/settings", headers=headers)
    settings = resp.json()
    assert settings["active_mode"] == "paid"
    assert settings["selected_providers"] == ["openai", "gemini"]
    assert settings["token_budget"] == 500000
    assert settings["local_max_context_tokens"] == 4096
    print("SUCCESS: Settings persistence verified!")
    
    # 7. Test local context size endpoint
    thread_id = "test-thread-id-999"
    resp = client.get(f"/api/local-context/size?thread_id={thread_id}", headers=headers)
    print(f"Get local context size response status: {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    assert "tokens" in data
    print(f"Context size: {data['tokens']} tokens")
    print("SUCCESS: Local context size verified!")
    
    print("\nAll auth and settings integration tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(run_tests())
