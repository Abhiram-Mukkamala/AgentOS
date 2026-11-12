"""
settings_manager.py — Manages token estimation and database-backed configuration per user.
"""

import logging
from typing import Any, Dict, List

log = logging.getLogger("agentos.settings")

DEFAULT_SETTINGS = {
    "active_mode": "free",
    "selected_providers": ["openai", "gemini", "anthropic"],
    "openai_api_key": "",
    "anthropic_api_key": "",
    "gemini_api_key": "",
    "token_budget": 100000,
    "token_used": 0,
    "token_warning_threshold": 0.90,
    "token_warning_sent": False,
    "token_auto_fallback": True,
    "local_max_context_tokens": 2048,
    "local_auto_delete": True,
    "language": "en"
}

def estimate_tokens(text: str) -> int:
    """
    Surgically fast token estimator based on character count and word length.
    Avoids native tiktoken compilation issues.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    if words == 0:
        return chars // 4
    # 1.35 tokens per word is a very reliable average for normal dev context.
    return int(words * 1.35)

async def get_user_settings(user_id: int) -> Dict[str, Any]:
    """Retrieve user settings dictionary from the database."""
    from database import AsyncSessionLocal, User
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                return DEFAULT_SETTINGS.copy()
            
            providers = [p.strip() for p in user.selected_providers.split(",")] if user.selected_providers else []
            return {
                "active_mode": user.active_mode,
                "selected_providers": providers,
                "openai_api_key": user.openai_api_key or "",
                "anthropic_api_key": user.anthropic_api_key or "",
                "gemini_api_key": user.gemini_api_key or "",
                "token_budget": user.token_budget,
                "token_used": user.token_used,
                "token_warning_threshold": user.token_warning_threshold,
                "token_warning_sent": user.token_warning_sent,
                "token_auto_fallback": user.token_auto_fallback,
                "local_max_context_tokens": user.local_max_context_tokens,
                "local_auto_delete": user.local_auto_delete,
                "language": user.language or "en",
            }
        except Exception as e:
            log.error("Failed to fetch settings for user %s: %s", user_id, e)
            return DEFAULT_SETTINGS.copy()

async def update_user_settings(user_id: int, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Update user settings in database and return the merged settings."""
    from database import AsyncSessionLocal, User
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                raise ValueError(f"User {user_id} not found")

            if "active_mode" in settings:
                user.active_mode = str(settings["active_mode"])
            if "selected_providers" in settings:
                user.selected_providers = ",".join(settings["selected_providers"])
            if "openai_api_key" in settings:
                user.openai_api_key = str(settings["openai_api_key"])
            if "anthropic_api_key" in settings:
                user.anthropic_api_key = str(settings["anthropic_api_key"])
            if "gemini_api_key" in settings:
                user.gemini_api_key = str(settings["gemini_api_key"])
            if "token_budget" in settings:
                user.token_budget = int(settings["token_budget"])
            if "token_used" in settings:
                user.token_used = int(settings["token_used"])
            if "token_warning_threshold" in settings:
                user.token_warning_threshold = float(settings["token_warning_threshold"])
            if "token_warning_sent" in settings:
                user.token_warning_sent = bool(settings["token_warning_sent"])
            if "token_auto_fallback" in settings:
                user.token_auto_fallback = bool(settings["token_auto_fallback"])
            if "local_max_context_tokens" in settings:
                user.local_max_context_tokens = int(settings["local_max_context_tokens"])
            if "local_auto_delete" in settings:
                user.local_auto_delete = bool(settings["local_auto_delete"])
            if "language" in settings:
                user.language = str(settings["language"])

            await session.commit()
            return await get_user_settings(user_id)
        except Exception as e:
            log.error("Failed to update settings for user %s: %s", user_id, e)
            await session.rollback()
            raise e

async def add_user_tokens(user_id: int, count: int) -> dict:
    """Add tokens to user's used token tally and check fallback rules."""
    settings = await get_user_settings(user_id)
    new_used = settings["token_used"] + count
    updates = {"token_used": new_used}
    
    # Auto fallback check
    if settings["token_auto_fallback"] and settings["active_mode"] != "local":
        if new_used >= settings["token_budget"]:
            updates["active_mode"] = "local"
            log.info("User %s token budget exhausted (%s/%s). Falling back to local mode.", 
                     user_id, new_used, settings["token_budget"])
            
    await update_user_settings(user_id, updates)
    return await get_user_settings(user_id)
