"""
database.py — AgentOS persistent storage layer.

Supports:
  • SQLite (local dev, default)        DATABASE_URL=sqlite+aiosqlite:///./agentos.db
  • Supabase / Postgres (production)   DATABASE_URL=postgresql://user:pass@host/db
"""

import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

load_dotenv()

# ── Connection URL ──────────────────────────────────────────────────────────────

_raw_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./agentos.db")

# Normalise Supabase/Heroku postgres:// → postgresql+asyncpg://
if _raw_url.startswith("postgresql://") or _raw_url.startswith("postgres://"):
    _raw_url = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    _raw_url = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)

DATABASE_URL = _raw_url

# ── Engine & Session ────────────────────────────────────────────────────────────
# BUG FIX: original code defined engine + AsyncSessionLocal twice — the first pair
# was silently overwritten by the second, making the NullPool setting unreachable.

engine = create_async_engine(DATABASE_URL, poolclass=NullPool, echo=False)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

# ── Models ──────────────────────────────────────────────────────────────────────


class User(Base):
    """User account credentials and configuration state."""
    __tablename__ = "users"

    id                       = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username                 = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password          = Column(String(255), nullable=False)
    active_mode              = Column(String(50), default="free")
    selected_providers       = Column(String(255), default="openai,gemini,anthropic")
    openai_api_key           = Column(String(500), nullable=True)
    anthropic_api_key        = Column(String(500), nullable=True)
    gemini_api_key           = Column(String(500), nullable=True)
    token_budget             = Column(Integer, default=100000)
    token_used               = Column(Integer, default=0)
    token_warning_threshold  = Column(Float, default=0.90)
    token_warning_sent       = Column(Boolean, default=False)
    token_auto_fallback      = Column(Boolean, default=True)
    local_max_context_tokens = Column(Integer, default=2048)
    local_auto_delete        = Column(Boolean, default=True)
    language                 = Column(String(10), default="en")
    created_at               = Column(DateTime, default=datetime.utcnow)


class CodeSnippet(Base):
    """Reusable code saved by the developer for future reference."""
    __tablename__ = "code_snippets"

    id          = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id     = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title       = Column(String(255), nullable=False)
    code        = Column(Text, nullable=False)
    language    = Column(String(50), default="python")
    tags        = Column(String(500), nullable=True)   # comma-separated keywords
    notes       = Column(Text, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)


class Improvement(Base):
    """Logged code improvements, refactoring decisions, and bug fixes."""
    __tablename__ = "improvements"

    id          = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id     = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    description = Column(Text, nullable=False)
    category    = Column(String(50), default="general")   # bug-fix|refactor|performance|security|feature|general
    impact      = Column(String(20), default="medium")    # low|medium|high
    created_at  = Column(DateTime, default=datetime.utcnow)


class Task(Base):
    """General task / note storage (kept for backwards compatibility)."""
    __tablename__ = "tasks"

    id          = Column(Integer, primary_key=True, index=True, autoincrement=True)
    description = Column(Text, nullable=False)
    status      = Column(String, default="pending")
    due_date    = Column(String, nullable=True)


# ── Initialisation ──────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables. Called once at app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
