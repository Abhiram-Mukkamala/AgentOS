"""
database.py — AgentOS persistent storage layer and shared auth utilities.

Supports:
  • SQLite (local dev, default)        DATABASE_URL=sqlite+aiosqlite:///./agentos.db
  • Supabase / Postgres (production)   DATABASE_URL=postgresql://user:pass@host/db
"""

import hashlib
import hmac
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

load_dotenv()

log = logging.getLogger("agentos.database")

# ── Connection URL ──────────────────────────────────────────────────────────────

_raw_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./agentos.db")

# Normalise Supabase/Heroku postgres:// → postgresql+asyncpg://
if _raw_url.startswith("postgresql://") or _raw_url.startswith("postgres://"):
    _raw_url = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    _raw_url = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)

DATABASE_URL = _raw_url

# ── Engine & Session ────────────────────────────────────────────────────────────

engine = create_async_engine(DATABASE_URL, poolclass=NullPool, echo=False)


@event.listens_for(engine.sync_engine, "connect")
def _enable_sqlite_fks(dbapi_conn, _connection_record):
    """Ensure ON DELETE CASCADE works in SQLite connections."""
    if DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


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


class ContextArtifact(Base):
    """
    Pass-by-reference offloaded context artifact (Phase 4.1).
    Stores large tool outputs, research digests, AST slices, patch diffs, and
    archived conversation turns out-of-band so AgentState.messages carries only
    compact pointers.
    """
    __tablename__ = "context_artifacts"

    artifact_id    = Column(String(32), primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    thread_id      = Column(String(128), index=True, nullable=False, default="default")
    producer_agent = Column(String(64), nullable=False)
    artifact_type  = Column(String(64), nullable=False)
    summary        = Column(Text, nullable=False)
    payload_json   = Column(Text, nullable=False)
    token_count    = Column(Integer, nullable=False, default=0)
    created_at     = Column(DateTime, default=datetime.utcnow)


class ASTSymbol(Base):
    """
    Indexed Python AST symbol node (function, method, or class) for the
    deterministic relational codebase graph (Phase 4.2 & Phase 7).
    """
    __tablename__ = "ast_symbols"

    symbol_id      = Column(String(64), primary_key=True, index=True)
    user_id        = Column(Integer, index=True, nullable=False)
    file_path      = Column(String(512), index=True, nullable=False)
    qualified_name = Column(String(512), index=True, nullable=False)
    short_name     = Column(String(128), index=True, nullable=False)
    symbol_kind    = Column(String(32), nullable=False)   # function|method|class
    start_line     = Column(Integer, nullable=False)
    end_line       = Column(Integer, nullable=False)
    signature_hash = Column(String(64), nullable=False)
    docstring      = Column(Text, nullable=True, default="")
    embedding_blob = Column(LargeBinary, nullable=True)


class ASTEdge(Base):
    """
    Directed dependency edge between an `ASTSymbol` caller and a target symbol
    (`CALLS`, `IMPORTS`, `INHERITS`) for 2-hop blast-radius queries (Phase 4.2).
    """
    __tablename__ = "ast_edges"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    user_id               = Column(Integer, index=True, nullable=False)
    caller_id             = Column(String(64), ForeignKey("ast_symbols.symbol_id", ondelete="CASCADE"), index=True, nullable=False)
    callee_qualified_name = Column(String(512), index=True, nullable=False)
    callee_short_name     = Column(String(128), index=True, nullable=False)
    edge_kind             = Column(String(32), nullable=False)   # CALLS|IMPORTS|INHERITS
    line_number           = Column(Integer, nullable=False)


# ── Shared Cryptographic & Local Dev User Helpers ───────────────────────────────


def hash_password(password: str) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with a 16-byte random salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"{salt.hex()}${key.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash."""
    try:
        salt_hex, key_hex = hashed.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return hmac.compare_digest(key, new_key)
    except Exception:
        return False


async def get_or_create_local_dev_user(session: AsyncSession | None = None) -> User:
    """Fetch or lazily create the default `local_dev_user` account."""
    async def _run(db: AsyncSession) -> User:
        result = await db.execute(select(User).where(User.username == "local_dev_user"))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                username="local_dev_user",
                hashed_password=hash_password("local_dev_password"),
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user

    if session is not None:
        return await _run(session)

    async with AsyncSessionLocal() as db:
        return await _run(db)


# ── Initialisation ──────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables. Called once at app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
