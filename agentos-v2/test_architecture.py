"""
test_architecture.py — Isolated unit & integration tests for Phase 4 and Phase 5
(`ContextArtifact` pass-by-reference offloading, sliding-window compaction,
`ASTGraphIndexer` incremental SHA-256 diffing & 2-hop blast radius, and
`_execute_and_record_tools` self-healing circuit breaker).
"""

import asyncio
import re
import sys
import tempfile
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import ast_graph
import database
import memory_manager
import tools as agentos_tools
from agent import _execute_and_record_tools
from ast_graph import ASTGraphIndexer, query_blast_radius
from database import Base, User
from memory_manager import (
    compact_conversation_window,
    fetch_context_artifact,
    store_context_artifact,
)


async def _create_isolated_session_factory():
    """Create an isolated in-memory SQLite engine + session factory (`sqlite+aiosqlite:///:memory:`)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        session.add(User(id=1, username="local_dev_user", hashed_password="pw"))
        await session.commit()
    return engine, factory


async def _run_artifact_offloading_and_selective_retrieval():
    engine, session_factory = await _create_isolated_session_factory()
    # Patch AsyncSessionLocal temporarily so tool calls use the in-memory DB
    orig_db_factory = database.AsyncSessionLocal
    orig_tools_factory = agentos_tools.AsyncSessionLocal
    orig_mem_factory = memory_manager.AsyncSessionLocal
    orig_ast_factory = ast_graph.AsyncSessionLocal
    database.AsyncSessionLocal = session_factory
    agentos_tools.AsyncSessionLocal = session_factory
    memory_manager.AsyncSessionLocal = session_factory
    ast_graph.AsyncSessionLocal = session_factory

    try:
        async with session_factory() as session:
            # 1. Verify direct structured ContextArtifact storage & selective key retrieval
            structured_payload = {
                "architecture": "Hybrid Edge-to-Kernel v2.5",
                "modules": ["edge_router.py", "ast_graph.py", "memory_manager.py", "agent.py"],
                "raw_trace": "TRACE_LINE_" * 600,
                "benchmark_target": {"recall_at_2_hops": 1.0, "token_reduction_pct": 85.0},
            }
            art_meta = await store_context_artifact(
                session=session,
                user_id=1,
                thread_id="thread-arch-1",
                producer_agent="ResearchAgent",
                artifact_type="research_digest",
                payload=structured_payload,
            )
            assert art_meta["artifact_id"].startswith("art_")
            assert art_meta["token_count"] > 1000
            assert art_meta["tokens_saved"] > int(art_meta["token_count"] * 0.60)

            # Selective key retrieval via fetch_context_artifact
            partial_fetch = await fetch_context_artifact(
                session=session,
                user_id=1,
                artifact_id=art_meta["artifact_id"],
                keys=["architecture", "benchmark_target"],
            )
            assert partial_fetch["data"] == {
                "architecture": "Hybrid Edge-to-Kernel v2.5",
                "benchmark_target": {"recall_at_2_hops": 1.0, "token_reduction_pct": 85.0},
            }
            assert "raw_trace" not in partial_fetch["data"]

            # Selective key retrieval via the @tool `read_context_artifact`
            tool_out = await agentos_tools.read_context_artifact.ainvoke(
                {"artifact_id": art_meta["artifact_id"], "keys": "architecture,modules"},
                config={"configurable": {"user_id": 1, "thread_id": "thread-arch-1"}},
            )
            assert "Hybrid Edge-to-Kernel v2.5" in tool_out
            assert "edge_router.py" in tool_out
            assert "TRACE_LINE_" not in tool_out

            # 2. Verify automatic >250 token offloading in `_execute_and_record_tools`
            #    achieves >60% token reduction on ToolMessage.content
            state = {"messages": [], "failure_signatures": []}
            raw_code = "print('\\n'.join(f'row_{i}: ' + 'X'*80 for i in range(40)))"
            msgs = await _execute_and_record_tools(
                [{"id": "call_big_1", "name": "execute_code", "args": {"code": raw_code}}],
                state,
                user_id=1,
                thread_id="thread-arch-1",
                session=session,
            )
            assert len(msgs) == 1
            tool_msg = msgs[0]
            assert isinstance(tool_msg, ToolMessage)
            assert "[Artifact ref: art_" in tool_msg.content

            m = re.search(r"Artifact ref: (art_[a-f0-9]+)\s*\|\s*Tokens:\s*(\d+)", tool_msg.content)
            assert m is not None
            offloaded_art_id = m.group(1)
            original_tokens = int(m.group(2))
            pointer_tokens = max(1, len(tool_msg.content) // 4)
            reduction_ratio = (original_tokens - pointer_tokens) / original_tokens
            assert reduction_ratio > 0.60, f"Expected >60% reduction, got {reduction_ratio:.2%}"

            # Verify line-slice retrieval from the offloaded tool output
            slice_fetch = await fetch_context_artifact(
                session=session,
                user_id=1,
                artifact_id=offloaded_art_id,
                keys=["lines:1-3"],
            )
            assert "row_0:" in slice_fetch["data"]["lines:1-3"]
            assert "row_2:" in slice_fetch["data"]["lines:1-3"]
            assert "row_10:" not in slice_fetch["data"]["lines:1-3"]

            # 3. Verify Sliding-Window Semantic Context Compaction
            long_history = [SystemMessage(content="Pinned system prompt")] + [
                HumanMessage(content=f"Turn {i}: " + ("context_payload " * 30))
                for i in range(14)
            ]
            compacted, ckpt_art = await compact_conversation_window(
                long_history,
                session=session,
                user_id=1,
                thread_id="thread-arch-1",
                token_budget=512,
                max_messages=12,
                keep_recent=6,
            )
            assert ckpt_art is not None
            assert ckpt_art["artifact_type"] == "conversation_checkpoint"
            # 1 pinned SystemMessage + 1 checkpoint SystemMessage + 6 verbatim recent turns = 8
            assert len(compacted) == 8
            assert "CONVERSATION CHECKPOINT" in compacted[1].content
    finally:
        database.AsyncSessionLocal = orig_db_factory
        agentos_tools.AsyncSessionLocal = orig_tools_factory
        memory_manager.AsyncSessionLocal = orig_mem_factory
        ast_graph.AsyncSessionLocal = orig_ast_factory
        await engine.dispose()


async def _run_ast_graph_indexer_and_blast_radius():
    engine, session_factory = await _create_isolated_session_factory()
    try:
        async with session_factory() as session:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)

                # File 1: low-level token verifier
                (root / "crypto_core.py").write_text(
                    "def verify_hmac_signature(payload: str, sig: str) -> bool:\n"
                    "    return len(payload) > 0 and sig.startswith('hmac_')\n\n"
                    "def unused_helper() -> int:\n"
                    "    return 123\n",
                    encoding="utf-8",
                )

                # File 2: Hop-1 service importing verify_hmac_signature under an alias
                (root / "auth_service.py").write_text(
                    "from crypto_core import verify_hmac_signature as check_sig\n\n"
                    "class SessionGuard:\n"
                    "    def authenticate_request(self, token: str) -> bool:\n"
                    "        return check_sig(token, 'hmac_valid')\n\n"
                    "def validate_api_webhook(body: str, header_sig: str) -> bool:\n"
                    "    return check_sig(body, header_sig)\n",
                    encoding="utf-8",
                )

                # File 3: Hop-2 gateway calling SessionGuard.authenticate_request and validate_api_webhook
                (root / "gateway_routes.py").write_text(
                    "from auth_service import SessionGuard, validate_api_webhook\n\n"
                    "def handle_protected_route(req_token: str) -> str:\n"
                    "    guard = SessionGuard()\n"
                    "    if guard.authenticate_request(req_token):\n"
                    "        return '200 OK'\n"
                    "    return '401 Unauthorized'\n\n"
                    "def handle_webhook_post(payload: str, sig: str) -> str:\n"
                    "    if validate_api_webhook(payload, sig):\n"
                    "        return 'ACK'\n"
                    "    return 'NACK'\n",
                    encoding="utf-8",
                )

                # File 4: Unrelated file mentioning verify_hmac_signature only in a comment/string
                (root / "docs_notes.py").write_text(
                    "NOTE = 'verify_hmac_signature is documented in the security wiki'\n"
                    "def render_help() -> str:\n"
                    "    return NOTE\n",
                    encoding="utf-8",
                )

                indexer = ASTGraphIndexer()

                # Initial full workspace index
                stats1 = await indexer.index_workspace(session, user_id=1, workspace_root=root)
                assert stats1["files_indexed"] == 4
                assert stats1["files_skipped"] == 0
                assert stats1["symbols_indexed"] >= 7
                assert stats1["edges_indexed"] >= 4

                # Second run without file changes -> 100% skipped via signature_hash
                stats2 = await indexer.index_workspace(session, user_id=1, workspace_root=root)
                assert stats2["files_indexed"] == 0
                assert stats2["files_skipped"] == 4
                assert stats2["symbols_unchanged"] == stats1["symbols_indexed"]

                # Modify only 1 file -> only 1 file re-indexed, 3 skipped
                (root / "crypto_core.py").write_text(
                    "def verify_hmac_signature(payload: str, sig: str, strict: bool = True) -> bool:\n"
                    "    return len(payload) > 0 and sig.startswith('hmac_')\n\n"
                    "def unused_helper() -> int:\n"
                    "    return 123\n",
                    encoding="utf-8",
                )
                stats3 = await indexer.index_workspace(session, user_id=1, workspace_root=root)
                assert stats3["files_indexed"] == 1
                assert stats3["files_skipped"] == 3

                # Query 2-hop blast radius for `verify_hmac_signature`
                t0 = time.perf_counter()
                impacts = await query_blast_radius(
                    session,
                    user_id=1,
                    symbol_name="verify_hmac_signature",
                    max_hops=2,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                assert elapsed_ms < 25.0  # typically < 2ms in SQLite

                by_qual = {item["qualified_name"]: item for item in impacts}
                # Hop 1 callers (even though imported via `as check_sig`!)
                assert "auth_service.SessionGuard.authenticate_request" in by_qual
                assert by_qual["auth_service.SessionGuard.authenticate_request"]["hop_depth"] == 1
                assert "auth_service.validate_api_webhook" in by_qual
                assert by_qual["auth_service.validate_api_webhook"]["hop_depth"] == 1

                # Hop 2 callers across files
                assert "gateway_routes.handle_protected_route" in by_qual
                assert by_qual["gateway_routes.handle_protected_route"]["hop_depth"] == 2
                assert "gateway_routes.handle_webhook_post" in by_qual
                assert by_qual["gateway_routes.handle_webhook_post"]["hop_depth"] == 2

                # String-only mention in docs_notes.py must NOT be a false positive
                assert "docs_notes.render_help" not in by_qual
    finally:
        await engine.dispose()


async def _run_circuit_breaker_tests():
    engine, session_factory = await _create_isolated_session_factory()
    try:
        async with session_factory() as session:
            # Case A: 2 consecutive identical failures on `execute_code` trips circuit breaker
            state_a = {"messages": [], "failure_signatures": []}
            failing_exec = [
                {
                    "id": "exec_fail_1",
                    "name": "execute_code",
                    "args": {"code": "def broken_calc():\n    return 1 / 0\nbroken_calc()"},
                }
            ]
            res1 = await _execute_and_record_tools(
                failing_exec, state_a, user_id=1, thread_id="cb-1", session=session
            )
            assert state_a.get("circuit_breaker_tripped") is not True
            assert len(state_a["failure_signatures"]) == 1

            res2 = await _execute_and_record_tools(
                failing_exec, state_a, user_id=1, thread_id="cb-1", session=session
            )
            assert state_a.get("circuit_breaker_tripped") is True
            assert len(state_a["failure_signatures"]) == 2
            sys_directives = [m for m in res2 if isinstance(m, SystemMessage)]
            assert len(sys_directives) == 1
            assert "CIRCUIT BREAKER TRIPPED" in sys_directives[0].content
            assert "get_symbol_blast_radius" in sys_directives[0].content

            # Case B: 3 distinct consecutive failures trip `max_consecutive_failures = 3`
            state_b = {"messages": [], "failure_signatures": []}
            for i in range(1, 4):
                call = [
                    {
                        "id": f"patch_fail_{i}",
                        "name": "patch_file",
                        "args": {
                            "path": f"missing_file_{i}.py",
                            "old_lines": "foo",
                            "new_lines": "bar",
                        },
                    }
                ]
                out = await _execute_and_record_tools(
                    call, state_b, user_id=1, thread_id="cb-2", session=session
                )
                if i < 3:
                    assert state_b.get("circuit_breaker_tripped") is not True
                else:
                    assert state_b.get("circuit_breaker_tripped") is True
                    assert any(
                        isinstance(m, SystemMessage) and "CIRCUIT BREAKER TRIPPED" in m.content
                        for m in out
                    )

            # Case C: Successful call resets failure_signatures and circuit_breaker_tripped
            ok_call = [
                {
                    "id": "exec_ok_1",
                    "name": "execute_code",
                    "args": {"code": "print('recovered_ok')"},
                }
            ]
            await _execute_and_record_tools(
                ok_call, state_b, user_id=1, thread_id="cb-2", session=session
            )
            assert state_b["failure_signatures"] == []
            assert state_b["circuit_breaker_tripped"] is False
    finally:
        await engine.dispose()


def test_context_artifact_offloading_and_compaction():
    asyncio.run(_run_artifact_offloading_and_selective_retrieval())


def test_relational_ast_graph_and_2hop_blast_radius():
    asyncio.run(_run_ast_graph_indexer_and_blast_radius())


def test_self_healing_circuit_breaker():
    asyncio.run(_run_circuit_breaker_tests())


if __name__ == "__main__":
    test_context_artifact_offloading_and_compaction()
    test_relational_ast_graph_and_2hop_blast_radius()
    test_self_healing_circuit_breaker()
    print("ALL_ARCHITECTURE_TESTS_PASSED")
