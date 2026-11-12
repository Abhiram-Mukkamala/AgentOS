"""
memory_manager.py — Persistent Memory Tokens, Pass-by-Reference ContextArtifact
Insulator, and Sliding-Window Semantic Context Compaction (Phase 4.1).
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal, ContextArtifact

MEMORY_FILE = Path(os.getcwd()) / "agentos_memory.json"
MAX_MEMORY_TOKENS = int(os.getenv("MEMORY_TOKEN_BUDGET", "512"))

# Runtime telemetry for pass-by-reference artifact offloading
_ARTIFACTS_STORED_COUNT = 0
_ARTIFACT_TOKENS_SAVED = 0


def get_memory_insulation_metrics() -> Dict[str, int]:
    """Return cumulative pass-by-reference artifact telemetry."""
    return {
        "artifacts_stored": _ARTIFACTS_STORED_COUNT,
        "tokens_saved": _ARTIFACT_TOKENS_SAVED,
    }


COMPRESSION_PROMPT = """You are a memory compressor. Given the action taken and its result, \
update the memory tokens to preserve the most important state for future steps. \
Be extremely concise. Use structured notation. Preserve: file paths edited, \
test results, errors encountered, plan remaining, key decisions made. \
Output ONLY the updated memory block in JSON format.

CURRENT MEMORY:
{current_memory_json}

NEW ACTION:
Tool called: {tool_name}
Input: {tool_input_summary}
Result: {tool_result_summary}

OUTPUT FORMAT (JSON only):
{{
  "files_edited": ["path:lines_changed"],
  "tests": {{"passed": 0, "failed": 0, "failing_tests": []}},
  "plan_remaining": ["step1", "step2"],
  "key_decisions": ["decision1"],
  "errors_seen": ["error_type: brief_description"],
  "context_summary": "one sentence about overall task state"
}}"""


def load_memory() -> dict:
    """Load persistent memory state from disk, or return empty schema."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "files_edited": [],
        "tests": {"passed": 0, "failed": 0, "failing_tests": []},
        "plan_remaining": [],
        "key_decisions": [],
        "errors_seen": [],
        "context_summary": "No prior context.",
    }


def save_memory(memory: dict) -> None:
    """Save persistent memory state to disk."""
    MEMORY_FILE.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def memory_to_system_message(memory: dict) -> SystemMessage:
    """Convert memory dictionary to a SystemMessage prepended to every LLM call."""
    tests = memory.get("tests", {})
    content = f"""
[PERSISTENT MEMORY — {MAX_MEMORY_TOKENS} token budget]
Files edited this session: {', '.join(memory.get('files_edited', [])) or 'none'}
Tests: {tests.get('passed', 0)} passed / {tests.get('failed', 0)} failed
Failing: {', '.join(tests.get('failing_tests', [])) or 'none'}
Plan remaining: {' → '.join(memory.get('plan_remaining', [])) or 'task complete'}
Key decisions: {'; '.join(memory.get('key_decisions', [])) or 'none'}
Errors seen: {'; '.join(memory.get('errors_seen', [])) or 'none'}
State: {memory.get('context_summary', 'No context')}
[END MEMORY]
""".strip()
    return SystemMessage(content=content)


async def update_memory(
    current: dict,
    tool_name: str,
    tool_input: str,
    tool_result: str,
    compressor_llm,
) -> dict:
    """
    Run the compression step after a tool call.
    Uses the compressor LLM to merge the new tool result into the existing memory dict.
    """
    prompt = COMPRESSION_PROMPT.format(
        current_memory_json=json.dumps(current, indent=2),
        tool_name=tool_name,
        tool_input_summary=str(tool_input)[:300],
        tool_result_summary=str(tool_result)[:500],
    )
    try:
        response = await compressor_llm.ainvoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", str(response)).strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()

        updated = json.loads(content)
        save_memory(updated)
        return updated
    except Exception:
        return current


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4.1: Pass-by-Reference ContextArtifact Store & Retrieval
# ═══════════════════════════════════════════════════════════════════════════════


def _build_compact_summary(payload: Any, summary: Optional[str] = None, max_tokens: int = 150) -> str:
    """Ensure the summary is concise and capped at `<= 150` tokens (auto-summaries target ~45 tokens)."""
    max_chars = max_tokens * 4
    if summary and summary.strip():
        clean = " ".join(summary.strip().split())
        return clean[: max_chars - 3] + "..." if len(clean) > max_chars else clean

    if isinstance(payload, dict):
        keys_preview = ", ".join(list(payload.keys())[:6])
        preview_str = json.dumps(payload, ensure_ascii=False)
        snippet = " ".join(preview_str.split())[:140]
        text = f"Keys=[{keys_preview}] | Preview: {snippet}"
    elif isinstance(payload, list):
        preview_str = json.dumps(payload[:2], ensure_ascii=False)
        snippet = " ".join(preview_str.split())[:140]
        text = f"List[{len(payload)} items] | Preview: {snippet}"
    else:
        raw_str = str(payload).strip()
        lines = [ln.strip() for ln in raw_str.splitlines() if ln.strip()]
        head = " | ".join(lines[:3])[:140]
        text = f"{len(lines)} lines ({len(raw_str)} chars): {head}"

    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def _serialize_payload(payload: Any) -> Tuple[str, int]:
    """
    Serialize `payload` into a JSON string and estimate its original token count
    using `len(serialized) // 4`.
    """
    if isinstance(payload, str):
        raw_token_count = max(1, len(payload) // 4)
        # If it is already a valid JSON dict or list, keep it structured
        stripped = payload.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                parsed = json.loads(stripped)
                serialized = json.dumps(parsed, ensure_ascii=False)
                return serialized, max(raw_token_count, len(serialized) // 4)
            except Exception:
                pass
        lines = payload.splitlines()
        structured = {
            "content": payload,
            "line_count": len(lines),
            "head": "\n".join(lines[:20]),
            "tail": "\n".join(lines[-20:]) if len(lines) > 20 else "\n".join(lines),
        }
        serialized = json.dumps(structured, ensure_ascii=False)
        return serialized, raw_token_count

    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    return serialized, max(1, len(serialized) // 4)


def format_artifact_pointer(
    artifact_id: str,
    summary: str,
    token_count: int,
    artifact_type: str = "tool_output",
) -> str:
    """Format a compact pass-by-reference pointer suitable for `ToolMessage.content`."""
    return (
        f"[Artifact ref: {artifact_id} | Type: {artifact_type} | "
        f"Tokens: {token_count} | Summary: {summary}] "
        f"(Call read_context_artifact('{artifact_id}') to inspect specific keys or full content.)"
    )


async def store_context_artifact(
    session: Optional[AsyncSession],
    user_id: int,
    thread_id: str,
    producer_agent: str,
    artifact_type: str,
    payload: Any,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persist a large payload in `context_artifacts` and return its compact pointer
    metadata along with token savings.
    """
    global _ARTIFACTS_STORED_COUNT, _ARTIFACT_TOKENS_SAVED

    payload_json, token_count = _serialize_payload(payload)
    compact_summary = _build_compact_summary(payload, summary=summary, max_tokens=150)

    digest = hashlib.sha256(
        f"{user_id}:{thread_id}:{producer_agent}:{artifact_type}:{payload_json}".encode("utf-8")
    ).hexdigest()[:12]
    artifact_id = f"art_{digest}"

    async def _persist(db: AsyncSession) -> None:
        existing = await db.execute(
            select(ContextArtifact).where(ContextArtifact.artifact_id == artifact_id)
        )
        row = existing.scalar_one_or_none()
        if row is None:
            row = ContextArtifact(
                artifact_id=artifact_id,
                user_id=int(user_id),
                thread_id=str(thread_id or "default"),
                producer_agent=str(producer_agent),
                artifact_type=str(artifact_type),
                summary=compact_summary,
                payload_json=payload_json,
                token_count=token_count,
                created_at=datetime.utcnow(),
            )
            db.add(row)
        else:
            row.summary = compact_summary
            row.payload_json = payload_json
            row.token_count = token_count
        await db.commit()

    if session is not None:
        await _persist(session)
    else:
        async with AsyncSessionLocal() as OwnSession:
            await _persist(OwnSession)

    pointer = format_artifact_pointer(
        artifact_id=artifact_id,
        summary=compact_summary,
        token_count=token_count,
        artifact_type=artifact_type,
    )
    pointer_tokens = max(1, len(pointer) // 4)
    tokens_saved = max(0, token_count - pointer_tokens)

    _ARTIFACTS_STORED_COUNT += 1
    _ARTIFACT_TOKENS_SAVED += tokens_saved

    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "producer_agent": producer_agent,
        "summary": compact_summary,
        "token_count": token_count,
        "pointer_tokens": pointer_tokens,
        "tokens_saved": tokens_saved,
        "pointer": pointer,
    }


async def fetch_context_artifact(
    session: Optional[AsyncSession],
    user_id: int,
    artifact_id: str,
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Retrieve a stored `ContextArtifact` scoped to `user_id`.
    When `keys` is specified, returns only the requested dictionary fields or slices.
    """
    async def _fetch(db: AsyncSession) -> Optional[ContextArtifact]:
        result = await db.execute(
            select(ContextArtifact).where(
                ContextArtifact.artifact_id == artifact_id.strip(),
                ContextArtifact.user_id == int(user_id),
            )
        )
        return result.scalar_one_or_none()

    if session is not None:
        row = await _fetch(session)
    else:
        async with AsyncSessionLocal() as OwnSession:
            row = await _fetch(OwnSession)

    if row is None:
        return {"error": f"Artifact '{artifact_id}' not found for user {user_id}."}

    try:
        parsed = json.loads(row.payload_json)
    except Exception:
        parsed = {"content": row.payload_json}

    clean_keys = [k.strip() for k in (keys or []) if k and k.strip()]
    if clean_keys and isinstance(parsed, dict):
        selected: Dict[str, Any] = {}
        for k in clean_keys:
            if k in parsed:
                selected[k] = parsed[k]
            elif k.startswith("lines:") and "content" in parsed:
                try:
                    span = k.split(":", 1)[1]
                    start_s, end_s = span.split("-", 1)
                    start_i = max(1, int(start_s))
                    end_i = int(end_s)
                    all_lines = str(parsed["content"]).splitlines()
                    selected[k] = "\n".join(all_lines[start_i - 1 : end_i])
                except Exception:
                    pass
        data_out: Any = selected
    else:
        data_out = parsed

    return {
        "artifact_id": row.artifact_id,
        "artifact_type": row.artifact_type,
        "producer_agent": row.producer_agent,
        "thread_id": row.thread_id,
        "summary": row.summary,
        "token_count": row.token_count,
        "data": data_out,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4.1: Sliding-Window Semantic Context Compaction
# ═══════════════════════════════════════════════════════════════════════════════


async def compact_conversation_window(
    messages: Sequence[BaseMessage],
    session: Optional[AsyncSession] = None,
    user_id: int = 1,
    thread_id: str = "default",
    token_budget: Optional[int] = None,
    max_messages: int = 12,
    keep_recent: int = 6,
) -> Tuple[List[BaseMessage], Optional[Dict[str, Any]]]:
    """
    When active message history exceeds `token_budget` (default `MEMORY_TOKEN_BUDGET`)
    or `max_messages` (default `12`), preserve leading `SystemMessage`s + pinned facts
    + the most recent `keep_recent` (`K = 6`) messages verbatim, archive older turns
    into a `ContextArtifact` (`artifact_type="conversation_checkpoint"`), and replace
    them in-state with a single compact checkpoint pointer message.
    """
    msg_list = list(messages)
    if not msg_list:
        return [], None

    budget = int(token_budget if token_budget is not None else MAX_MEMORY_TOKENS)
    total_tokens = sum(max(1, len(str(getattr(m, "content", "") or "")) // 4) for m in msg_list)

    if len(msg_list) <= max_messages and total_tokens <= budget:
        return msg_list, None

    pinned_system: List[BaseMessage] = []
    conversational: List[BaseMessage] = []
    for m in msg_list:
        if isinstance(m, SystemMessage) and not conversational:
            pinned_system.append(m)
        else:
            conversational.append(m)

    if len(conversational) <= keep_recent:
        return msg_list, None

    older_turns = conversational[:-keep_recent]
    recent_turns = conversational[-keep_recent:]

    serialized_turns = [
        {
            "role": getattr(m, "type", m.__class__.__name__),
            "name": getattr(m, "name", None),
            "content": str(getattr(m, "content", "") or ""),
        }
        for m in older_turns
    ]
    digest_parts = []
    for t in serialized_turns[:6]:
        snippet = " ".join(t["content"].split())[:80]
        digest_parts.append(f"{t['role']}: {snippet}")
    summary_text = f"Archived {len(older_turns)} turns — " + " | ".join(digest_parts)

    art = await store_context_artifact(
        session=session,
        user_id=user_id,
        thread_id=thread_id,
        producer_agent="MemoryCompactor",
        artifact_type="conversation_checkpoint",
        payload={"archived_turns": serialized_turns, "turn_count": len(older_turns)},
        summary=summary_text,
    )

    checkpoint_msg = SystemMessage(
        content=(
            f"[CONVERSATION CHECKPOINT: {art['artifact_id']} | "
            f"Archived {len(older_turns)} older messages ({art['token_count']} tokens) | "
            f"Summary: {art['summary']}] "
            f"Use read_context_artifact('{art['artifact_id']}') if verbatim earlier history is needed."
        )
    )
    compacted = pinned_system + [checkpoint_msg] + recent_turns
    return compacted, art
