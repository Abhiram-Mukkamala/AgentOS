"""
agent.py — AgentOS multi-agent orchestration via LangGraph.

Architecture:
  Commenter Supervisor ─┬─► CodexEngine    (ReAct loop: plan → act → observe → repeat)
                        ├─► ResearchAgent  (web search, scraping, algorithm lookup)
                        └─► DBAgent        (snippets, changelog)
"""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import operator
import os
import re
import socket
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Annotated, Any, List, Literal, Optional, Sequence, Tuple, TypedDict
from urllib.parse import urlparse

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from memory_manager import (
    compact_conversation_window,
    load_memory,
    memory_to_system_message,
    store_context_artifact,
    update_memory,
)
from settings_manager import add_user_tokens, estimate_tokens, get_user_settings
from tools import (
    analyze_code,
    execute_code,
    get_snippets,
    get_symbol_blast_radius,
    index_codebase_graph,
    index_repo,
    list_directory,
    log_improvement,
    patch_file,
    read_context_artifact,
    read_file,
    run_shell_command,
    run_tests,
    save_code_snippet,
    scrape_page,
    search_algorithms,
    search_codebase,
    web_search,
    write_file,
)

log = logging.getLogger("agentos")


# ── Async / Sync Delegation Helper ──────────────────────────────────────────────


def _run_sync(coro):
    """Run an async coroutine from a synchronous `.invoke()` context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ── LLM Factory & Key Validation ────────────────────────────────────────────────


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_valid_key(key: str | None) -> bool:
    """Validate API key format using placeholder rejection, length, and Shannon entropy."""
    if not key:
        return False
    k = key.strip()
    if len(k) < 16:
        return False
    lower = k.lower()
    invalid_markers = ("your_", "your-", "placeholder", "dummy", "changeme", "example", "test_key")
    if any(m in lower for m in invalid_markers):
        return False
    if _shannon_entropy(k) < 3.0:
        return False
    return True


class MissingAPIKeysModel:
    async def ainvoke(self, *args, **kwargs):
        raise ValueError("PAID_API_KEYS_MISSING_OR_INVALID")

    def invoke(self, *args, **kwargs):
        return _run_sync(self.ainvoke(*args, **kwargs))

    def bind_tools(self, *args, **kwargs):
        return self

    def with_structured_output(self, *args, **kwargs):
        return self


# ── Embedded Local Server Watchdog with TTL Cache ───────────────────────────────

_last_server_check_ts: float = 0.0
_SERVER_CHECK_TTL_SECONDS: float = 30.0


def ensure_local_server(runnable) -> None:
    """
    Ensure the local Codex Engine server is listening when a local model is invoked.
    Uses a 30-second TTL cache so repeated calls do not block on socket/HTTP checks.
    """
    global _last_server_check_ts
    now = time.monotonic()
    if now - _last_server_check_ts < _SERVER_CHECK_TTL_SECONDS:
        return

    underlying = runnable
    while True:
        if isinstance(underlying, BaseChatModel):
            break
        if hasattr(underlying, "bound"):
            underlying = underlying.bound
        elif hasattr(underlying, "bound_model"):
            underlying = underlying.bound_model
        elif hasattr(underlying, "model") and not isinstance(underlying.model, str):
            underlying = underlying.model
        else:
            break

    api_base = ""
    if hasattr(underlying, "openai_api_base") and underlying.openai_api_base:
        api_base = str(underlying.openai_api_base)
    elif hasattr(underlying, "base_url") and underlying.base_url:
        api_base = str(underlying.base_url)

    local_url = os.getenv("LOCAL_CODEX_URL", "http://localhost:8090/v1")
    parsed = urlparse(api_base or "")
    target_port = parsed.port or int(os.getenv("LOCAL_CODEX_PORT", "8090"))
    target_host = parsed.hostname or "127.0.0.1"

    if str(target_port) not in api_base and api_base != local_url:
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect((target_host, target_port))
            _last_server_check_ts = now
            return
    except Exception:
        pass

    if os.getenv("LOCAL_CODEX_EMBEDDED", "true").lower() == "false":
        return

    try:
        from codex_engine import start_codex_server

        log.info("Starting embedded Codex Engine on port %d...", target_port)
        start_codex_server(port=target_port, background=True)
        for _ in range(5):
            try:
                resp = httpx.get(f"http://{target_host}:{target_port}/health", timeout=1.0)
                if resp.status_code == 200:
                    _last_server_check_ts = time.monotonic()
                    log.info("Embedded Codex Engine started successfully!")
                    break
            except Exception:
                time.sleep(1.0)
    except Exception as e:
        log.error("Failed to start embedded Codex Engine: %s", e)


class FallbackRunnable:
    def __init__(self, runnables: list):
        self.runnables = runnables

    async def ainvoke(self, input, config=None, **kwargs):
        last_exception = None
        for r in self.runnables:
            try:
                ensure_local_server(r)
                return await r.ainvoke(input, config=config, **kwargs)
            except Exception as e:
                log.warning("Fallback triggered: failed with %s. Trying next...", e)
                last_exception = e
        if last_exception:
            raise last_exception
        raise ValueError("No runnables in fallback chain")

    def invoke(self, input, config=None, **kwargs):
        return _run_sync(self.ainvoke(input, config=config, **kwargs))


# ── G4F Free Web Model Adapter (Deduplicated Sync/Async) ────────────────────────


class G4FChatModel(BaseChatModel):
    model_name: str
    tools: Optional[List[Any]] = None

    def __init__(self, model_name: str, tools: Optional[List[Any]] = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        self.tools = tools

    def bind_tools(self, tools, **kwargs):
        return G4FChatModel(model_name=self.model_name, tools=tools)

    def _prepare_g4f_payload(self, messages: List[BaseMessage]) -> tuple[str, list[dict], Optional[list], list]:
        """Shared payload preparation for both sync and async g4f execution."""
        from g4f.Provider import OperaAria, Yqcloud

        g4f_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            g4f_messages.append({"role": role, "content": msg.content})

        lower_name = self.model_name.lower()
        if "gemini" in lower_name:
            g4f_model = "gemini-2.5-flash"
        elif "claude" in lower_name:
            g4f_model = "claude-3.5-sonnet"
        elif "gpt-4" in lower_name and "mini" not in lower_name:
            g4f_model = "gpt-4o"
        elif "gpt-3.5" in lower_name or "mini" in lower_name:
            g4f_model = "gpt-4o-mini"
        else:
            g4f_model = self.model_name

        tools_arg = [convert_to_openai_tool(t) for t in self.tools] if self.tools else None
        providers = [Yqcloud, OperaAria, None] if "gpt" in g4f_model.lower() else [Yqcloud, None]
        return g4f_model, g4f_messages, tools_arg, providers

    @staticmethod
    def _parse_g4f_response(response: Any) -> ChatResult:
        """Shared parser converting a g4f completion response into a LangChain ChatResult."""
        choice = response.choices[0]
        content = choice.message.content or ""

        tool_calls = []
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    raw_args = tc.function.arguments
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    tool_calls.append({
                        "name": tc.function.name,
                        "args": args,
                        "id": tc.id or f"call_{uuid.uuid4().hex}",
                    })
                except Exception:
                    pass

        ai_message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from g4f.client import Client

        g4f_model, g4f_messages, tools_arg, providers = self._prepare_g4f_payload(messages)
        response = None
        last_exception = None

        for prov in providers:
            try:
                client = Client()
                call_kwargs = {
                    "model": g4f_model,
                    "messages": g4f_messages,
                    "tools": tools_arg,
                }
                if prov is not None:
                    call_kwargs["provider"] = prov
                call_kwargs.update(kwargs)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(client.chat.completions.create, **call_kwargs)
                    response = future.result(timeout=15.0)
                if response and response.choices and response.choices[0].message.content:
                    break
            except Exception as e:
                prov_name = prov.__name__ if prov else "default"
                log.warning("g4f _generate provider %s failed: %s", prov_name, e)
                last_exception = e

        if not response:
            if last_exception:
                raise last_exception
            raise ValueError("All g4f providers failed")

        return self._parse_g4f_response(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from g4f.client import AsyncClient

        g4f_model, g4f_messages, tools_arg, providers = self._prepare_g4f_payload(messages)
        response = None
        last_exception = None

        for prov in providers:
            try:
                client = AsyncClient()
                call_kwargs = {
                    "model": g4f_model,
                    "messages": g4f_messages,
                    "tools": tools_arg,
                }
                if prov is not None:
                    call_kwargs["provider"] = prov
                call_kwargs.update(kwargs)

                response = await asyncio.wait_for(
                    client.chat.completions.create(**call_kwargs),
                    timeout=15.0,
                )
                if response and response.choices and response.choices[0].message.content:
                    break
            except Exception as e:
                prov_name = prov.__name__ if prov else "default"
                log.warning("g4f _agenerate provider %s failed: %s", prov_name, e)
                last_exception = e

        if not response:
            if last_exception:
                raise last_exception
            raise ValueError("All g4f async providers failed")

        return self._parse_g4f_response(response)

    @property
    def _llm_type(self) -> str:
        return "g4f"


# ── Dynamic Configuration, Token Tracking & Context Pruning ──────────────────────


def get_local_env_settings() -> dict:
    """Helper to build settings from environment variables for local/offline execution."""
    providers = ["openai", "gemini", "anthropic"]
    preferred = os.getenv("SENIOR_MODEL_PROVIDER", "").strip().lower()
    if preferred in providers:
        providers = [preferred] + [p for p in providers if p != preferred]

    return {
        "active_mode": (
            "paid"
            if os.getenv("USE_PAID_API", "false").lower() == "true"
            else ("free" if os.getenv("USE_FREE_VERSIONS", "true").lower() == "true" else "local")
        ),
        "selected_providers": providers,
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "token_budget": int(os.getenv("TOKEN_BUDGET", "100000")),
        "token_used": int(os.getenv("TOKEN_USED", "0")),
        "token_warning_threshold": 0.90,
        "token_auto_fallback": True,
        "local_max_context_tokens": int(os.getenv("LOCAL_MAX_CONTEXT_TOKENS", "2048")),
        "local_auto_delete": True,
        "language": "en",
    }


def prune_messages_to_limit(messages: list, limit: int, user_id: int | None) -> tuple[list, list]:
    """Prunes oldest messages if total tokens exceed limit, saving pruned logs locally."""
    system_msgs = []
    chat_msgs = []
    for m in messages:
        if isinstance(m, SystemMessage) or (hasattr(m, "type") and m.type == "system"):
            system_msgs.append(m)
        else:
            chat_msgs.append(m)

    def total_tokens(sys_m, chat_m):
        return estimate_tokens("".join(getattr(m, "content", "") or "" for m in sys_m + chat_m))

    pruned = []
    while total_tokens(system_msgs, chat_msgs) > limit and len(chat_msgs) > 0:
        pruned.append(chat_msgs.pop(0))

    if pruned:
        pruned_dicts = []
        for m in pruned:
            role = "user"
            if isinstance(m, AIMessage) or (hasattr(m, "type") and m.type == "assistant"):
                role = "assistant"
            elif isinstance(m, ToolMessage) or (hasattr(m, "type") and m.type == "tool"):
                role = "tool"
            pruned_dicts.append({
                "role": role,
                "content": getattr(m, "content", "") or "",
                "name": getattr(m, "name", None),
                "tool_calls": getattr(m, "tool_calls", None),
            })

        filename = f"pruned_history_{user_id}.json" if user_id else "pruned_history.json"
        filepath = Path(os.getcwd()) / filename

        existing = []
        if filepath.exists():
            try:
                existing = json.loads(filepath.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.extend(pruned_dicts)
        filepath.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        log.info(
            "Local model context limit reached. Pruned %s oldest messages and archived to %s",
            len(pruned),
            filename,
        )

    return system_msgs + chat_msgs, pruned


class TokenTrackingModel:
    """Wrapper that estimates tokens consumed, logs them to settings, and truncates local context."""

    def __init__(
        self,
        bound_model,
        user_id: int | None,
        is_local: bool = False,
        max_context: int = 2048,
        auto_delete: bool = True,
    ):
        self.bound_model = bound_model
        self.user_id = user_id
        self.is_local = is_local
        self.max_context = max_context
        self.auto_delete = auto_delete

    async def ainvoke(self, messages, config=None, **kwargs):
        if self.is_local:
            messages_to_use, pruned = prune_messages_to_limit(messages, self.max_context, self.user_id)
            if not self.auto_delete and pruned:
                raise ValueError("LOCAL_CONTEXT_WINDOW_EXCEEDED")
            messages = messages_to_use

        if isinstance(messages, list):
            input_text = "".join(getattr(m, "content", "") or "" for m in messages)
        else:
            input_text = str(messages)
        input_tokens = estimate_tokens(input_text)

        res = await self.bound_model.ainvoke(messages, config=config, **kwargs)

        output_text = getattr(res, "content", "") or ""
        output_tokens = estimate_tokens(output_text)

        total_tokens = input_tokens + output_tokens
        if self.user_id:
            await add_user_tokens(self.user_id, total_tokens)
        else:
            log.info("Local session consumed %s tokens.", total_tokens)

        return res

    def invoke(self, messages, config=None, **kwargs):
        return _run_sync(self.ainvoke(messages, config=config, **kwargs))


# ── Unified Structured Output Wrapper ───────────────────────────────────────────


def wrap_robust_structured_output(model, schema):
    """
    Unified structured-output wrapper used across all LLM backends (including G4FChatModel).
    Includes heuristic `Route` recovery when local/free models emit raw tool names instead of JSON.
    """

    def parse_output(response_message) -> Any:
        content = (getattr(response_message, "content", "") or "").strip()

        if schema.__name__ == "Route":
            content_lower = content.lower()
            reasoning = f"Heuristic override based on model content: {content[:100]}"

            db_keywords = ("save_code_snippet", "get_snippets", "log_improvement", "save_snippet")
            if any(kw in content_lower for kw in db_keywords):
                log.info("Heuristically routing to DBAgent because model outputted: %s", content)
                return schema(next="DBAgent", reasoning=reasoning)

            research_keywords = ("web_search", "scrape_page", "search_algorithms")
            if any(kw in content_lower for kw in research_keywords):
                log.info("Heuristically routing to ResearchAgent because model outputted: %s", content)
                return schema(next="ResearchAgent", reasoning=reasoning)

            codex_keywords = (
                "write_file",
                "patch_file",
                "read_file",
                "execute_code",
                "run_tests",
                "list_directory",
                "search_codebase",
                "run_shell_command",
                "index_repo",
                "delete_file",
                "append_to_file",
            )
            if "functions." in content_lower or any(kw in content_lower for kw in codex_keywords):
                log.info("Heuristically routing to CodexEngine because model outputted: %s", content)
                return schema(next="CodexEngine", reasoning=reasoning)

        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    json_str = content[start : end + 1]
                else:
                    json_str = content

        try:
            data = json.loads(json_str)
            if issubclass(schema, BaseModel):
                return schema(**data)
            return data
        except Exception as e:
            log.warning("Structured output parsing failed: %s, raw content: %s", e, content)
            if issubclass(schema, BaseModel):
                defaults = {}
                for field_name, field in schema.model_fields.items():
                    if field.default is not None and field.default is not PydanticUndefined:
                        defaults[field_name] = field.default
                    elif field.default_factory is not None:
                        defaults[field_name] = field.default_factory()
                    else:
                        if getattr(field.annotation, "__origin__", None) is Literal:
                            defaults[field_name] = field.annotation.__args__[0]
                        else:
                            defaults[field_name] = "FINISH"
                return schema(**defaults)
            return {}

    schema_repr = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema_json()
    system_instruction = f"Output must be a valid JSON object matching the schema: {schema_repr}"

    class RobustStructuredModel:
        def __init__(self, bound_model):
            self.bound_model = bound_model

        async def ainvoke(self, messages, config=None, **kwargs):
            new_messages = [SystemMessage(content=system_instruction)] + list(messages)
            res = await self.bound_model.ainvoke(new_messages, config=config, **kwargs)
            return parse_output(res)

        def invoke(self, messages, config=None, **kwargs):
            return _run_sync(self.ainvoke(messages, config=config, **kwargs))

    return RobustStructuredModel(model)


# ── Dynamic Model Retrieval ─────────────────────────────────────────────────────


def _resolve_model_names(agent_name: str) -> dict[str, str]:
    """Resolve model names per provider from environment variables with sensible defaults."""
    is_senior = agent_name in ("codex", "research", "supervisor")
    return {
        "openai": (
            os.getenv("OPENAI_MODEL", "gpt-4o")
            if is_senior
            else os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")
        ),
        "anthropic": (
            os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
            if is_senior
            else os.getenv("ANTHROPIC_FAST_MODEL", "claude-3-5-haiku-20241022")
        ),
        "gemini": (
            os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            if is_senior
            else os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash")
        ),
    }


def _order_providers(providers: list[str]) -> list[str]:
    """Order selected providers so `SENIOR_MODEL_PROVIDER` is attempted first if enabled."""
    preferred = os.getenv("SENIOR_MODEL_PROVIDER", "").strip().lower()
    if preferred and preferred in providers:
        return [preferred] + [p for p in providers if p != preferred]
    return list(providers)


async def get_agent_llm(agent_name: str, config, tools=None, structured_schema=None):
    """
    Builds the model fallback chain dynamically based on the current user's settings
    and environment variables. Integrates TokenTrackingModel wrapper per runnable.
    """
    configurable = config.get("configurable", {}) if config else {}
    user_id = configurable.get("user_id")

    if user_id:
        settings = await get_user_settings(user_id)
    else:
        settings = get_local_env_settings()

    active_mode = settings["active_mode"]
    providers = _order_providers(settings["selected_providers"])
    max_context = settings.get("local_max_context_tokens", 2048)
    auto_delete = settings.get("local_auto_delete", True)
    model_names = _resolve_model_names(agent_name)
    local_codex_enabled = os.getenv("LOCAL_CODEX", "true").lower() != "false"

    chain = []

    def _prepare(model, is_local_model: bool = False):
        if tools:
            model = model.bind_tools(tools)
        if structured_schema:
            model = wrap_robust_structured_output(model, structured_schema)
        return TokenTrackingModel(
            model,
            user_id,
            is_local=is_local_model or (active_mode == "local"),
            max_context=max_context,
            auto_delete=auto_delete,
        )

    def _build_local_model():
        local_url = os.getenv("LOCAL_CODEX_URL", "http://localhost:8090/v1")
        return ChatOpenAI(
            base_url=local_url,
            model=os.getenv("LOCAL_CODEX_MODEL", "qwen2.5-coder-7b-instruct"),
            api_key="not-needed",
            temperature=0.1,
            max_tokens=4096,
        )

    # 1. Local Codex mode
    if active_mode == "local":
        model = _build_local_model()
        ensure_local_server(model)
        chain.append(_prepare(model, is_local_model=True))

    # 2. Paid API mode
    elif active_mode == "paid":
        temp = 0.0 if agent_name == "routing" else 0.1
        for prov in providers:
            if prov == "openai" and _is_valid_key(settings["openai_api_key"]):
                if ChatOpenAI is None:
                    log.warning("ChatOpenAI requested but langchain-openai is not installed.")
                else:
                    chain.append(
                        _prepare(
                            ChatOpenAI(
                                model=model_names["openai"],
                                api_key=settings["openai_api_key"],
                                temperature=temp,
                            )
                        )
                    )
            elif prov == "anthropic" and _is_valid_key(settings["anthropic_api_key"]):
                if ChatAnthropic is None:
                    log.warning("ChatAnthropic requested but langchain-anthropic is not installed.")
                else:
                    chain.append(
                        _prepare(
                            ChatAnthropic(
                                model=model_names["anthropic"],
                                api_key=settings["anthropic_api_key"],
                                temperature=temp,
                            )
                        )
                    )
            elif prov == "gemini" and _is_valid_key(settings["gemini_api_key"]):
                if ChatGoogleGenerativeAI is None:
                    log.warning("ChatGoogleGenerativeAI requested but langchain-google-genai is not installed.")
                else:
                    chain.append(
                        _prepare(
                            ChatGoogleGenerativeAI(
                                model=model_names["gemini"],
                                api_key=settings["gemini_api_key"],
                                temperature=temp,
                            )
                        )
                    )

        if not chain:
            chain.append(TokenTrackingModel(MissingAPIKeysModel(), user_id))

    # 3. Free API mode (g4f web models + optional local Codex fallback)
    else:
        import importlib.util

        if importlib.util.find_spec("g4f") is not None:
            for prov in providers:
                if prov in model_names:
                    chain.append(_prepare(G4FChatModel(model_name=model_names[prov])))

        if local_codex_enabled and ChatOpenAI is not None:
            chain.append(_prepare(_build_local_model(), is_local_model=True))
        elif not chain:
            chain.append(TokenTrackingModel(MissingAPIKeysModel(), user_id))

    if len(chain) > 1:
        return FallbackRunnable(chain)
    return chain[0]


# ── State & Routing Schema ──────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    iterations: int  # supervisor routing budget (resets per ainvoke)
    failure_signatures: List[Tuple[str, str, str]]
    circuit_breaker_tripped: bool


class Route(BaseModel):
    next: Literal["FINISH", "CodexEngine", "ResearchAgent", "DBAgent"] = Field(
        description="The specialist to call next, or FINISH to write the final response."
    )
    reasoning: str = Field(description="One-sentence explanation for this routing decision.")


# ── System Prompts ──────────────────────────────────────────────────────────────

COMMENTER_ROUTING_PROMPT = """You are the Commenter — a senior software engineer \
orchestrating a developer productivity assistant.

Your three specialists:

  CodexEngine    A Codex-level hybrid coding agent with two modes:
                 CONSULTATIVE: reviews code, proposes solutions in chat.
                 AUTONOMOUS: reads files, patches them, runs tests, iterates until green.
                 Also handles: static analysis, code generation, debugging, refactoring.

  ResearchAgent  Web search for bug fixes, docs, Stack Overflow, GitHub repos,
                 algorithm implementations with complexity analysis.

  DBAgent        Save/retrieve code snippets, log improvements to project changelog.

Routing rules:
  • Code to review / fix / improve / build        → CodexEngine
  • "build", "implement", "create", "fix this file" → CodexEngine (autonomous)
  • "what do you think", "review", "should I"       → CodexEngine (consultative)
  • Run tests / shell commands                       → CodexEngine
  • "how do I…", "best way to…", algorithm question → ResearchAgent
  • Save / retrieve code or log a decision           → DBAgent
  • Task fully answered by prior specialist output   → FINISH
  Never route the same request to the same specialist twice in one turn."""

COMMENTER_SYNTHESIS_PROMPT = """You are the Commenter — a senior software engineer \
writing the final response to the developer.

Specialists have completed their work. Write a comprehensive response:
  1. Direct Answer      — address the developer's request immediately
  2. Code Commentary    — review clarity, edge cases, efficiency, best practices
  3. Key Findings       — summarise what research / execution revealed
  4. Actionable Steps   — specific, ordered next steps
  5. Code Examples      — show improved code in fenced blocks where useful

Tone: senior engineer doing a code review — specific, constructive, educational.
Format: clean markdown with sections and code blocks."""

CODEX_SYSTEM_PROMPT = """You are Codex — a world-class autonomous coding agent, \
on par with OpenAI Codex and Claude Code.

═══ TWO MODES ═══════════════════════════════════════════════════════════════

MODE 1: CONSULTATIVE (default for ambiguous requests)
  Triggered by: "what do you think", "review this", "how should I", vague ideas
  Behaviour:
    • Read files to understand context (read_file, list_directory, search_codebase)
    • Analyse code (analyze_code)
    • Ask one clarifying question if intent is unclear
    • Propose complete, production-ready code IN THE CHAT (fenced blocks)
    • Explain trade-offs. The developer copies what they like.
    • Do NOT use patch_file, write_file, or run_shell_command in this mode.
    • Execute code to VERIFY your suggestions (execute_code is always fine).

MODE 2: AUTONOMOUS (for explicit implementation tasks)
  Triggered by: "build this", "implement", "fix this file", "go ahead", "do it",
                "run the tests", "create X", "update Y"
  Behaviour — follow this exact workflow:

  STEP 1 — ORIENT
    Always start with: index_repo() to understand the whole project.
    Then:              list_directory() and read_file() on relevant files.
    Then:              search_codebase() to find related code / imports / usages.
    Do NOT skip orientation — blind edits break things.

  STEP 2 — PLAN (internal, before acting)
    Before touching any file, state your plan:
      "I will: 1) patch X in file A, 2) add Y to file B, 3) run tests to verify."
    This keeps you from going in circles.

  STEP 3 — ACT
    • patch_file to surgically edit existing files (preferred over write_file).
    • write_file to create new files.
    • execute_code to validate logic in a sandbox before writing to disk.
    • run_shell_command for installs, builds, git operations.

  STEP 4 — VERIFY
    • run_tests after every change. Read failures carefully.
    • If tests fail: search_codebase for the failing symbol, read_file the
      relevant lines, patch_file to fix, run_tests again.
    • Repeat until green or until you've exhausted 3 fix attempts, then
      report exactly what failed and why.

  STEP 5 — REPORT
    Summarise: what you changed, what files, what the tests showed.

═══ TOOL SELECTION GUIDE ════════════════════════════════════════════════════

  index_repo        → start of any autonomous task (full project map)
  list_directory    → explore a specific subdirectory
  read_file         → read a specific file with line numbers
  search_codebase   → find a symbol/pattern across all files (grep)
  analyze_code      → AST-based bug/quality analysis of a code string
  execute_code      → run Python in a sandbox (always verify before patching)
  patch_file        → surgically replace a block of lines in an existing file
  write_file        → create a new file (or full overwrite when needed)
  run_tests         → run pytest, get structured pass/fail/error output
  run_shell_command → pip, npm, git, make, cargo, or any other shell command

═══ GOLDEN RULES ════════════════════════════════════════════════════════════

  • ALWAYS read a file before patching it.
  • ALWAYS run tests after patching.
  • ALWAYS execute_code to verify logic before writing to disk.
  • NEVER guess what a file contains — use read_file or search_codebase.
  • If ambiguous intent: ask one question, default to consultative mode.
  • Explain the WHY behind every decision, not just the WHAT."""

RESEARCH_SYSTEM_PROMPT = """You are the Research Agent — expert at surfacing software \
engineering knowledge from the internet.

Use your tools to:
  • Search for bug fixes, library docs, tutorials, and error explanations
  • Look up algorithm implementations with time/space complexity analysis
  • Scrape Stack Overflow, GitHub READMEs, or docs pages for full content
  • Find best practices and known pitfalls for the technology in question

Synthesise findings into concise, actionable insights. Always include source URLs."""

DB_SYSTEM_PROMPT = """You are the DB Agent — persistent memory for the developer's workspace.

  • Save code snippets with clear titles and searchable tags
  • Retrieve previously saved code when the developer asks
  • Log improvements, bug fixes, and architectural decisions to the changelog

Always confirm what was saved/retrieved/logged with the relevant details."""


# ── Safety Limits & Tool Maps ───────────────────────────────────────────────────

MAX_SUPERVISOR_ITERATIONS = 6
CODEX_MAX_STEPS = 12
DEFAULT_MAX_STEPS = 6

codex_tools = [
    execute_code,
    analyze_code,
    read_file,
    list_directory,
    write_file,
    patch_file,
    search_codebase,
    run_tests,
    run_shell_command,
    index_repo,
    index_codebase_graph,
    get_symbol_blast_radius,
    read_context_artifact,
]
research_tools = [web_search, scrape_page, search_algorithms, read_context_artifact]
db_tools = [save_code_snippet, get_snippets, log_improvement, read_context_artifact]

_ALL_TOOLS = {t.name: t for t in (codex_tools + research_tools + db_tools)}


# ══════════════════════════════════════════════════════════════════════════════
# CORE: ReAct worker node
# ══════════════════════════════════════════════════════════════════════════════

PLANNING_INJECTION = """Before you call any tool, output a short PLAN as your first \
response.  The plan MUST:
  1. State what the user wants (one sentence).
  2. List 3-7 concrete steps you will take, in order.
  3. Identify which files you expect to touch.
  4. State your success criteria (e.g. "tests pass", "file created").
Do NOT call any tools in this response.  Just output the plan in plain text.
After this, you will proceed to execute the plan step by step."""

ARTIFACT_OFFLOAD_CHAR_THRESHOLD = 1000  # ~250 tokens
MAX_CONSECUTIVE_TOOL_FAILURES = 3

_TOOL_ARTIFACT_TYPE_MAP = {
    "web_search": "research_digest",
    "scrape_page": "research_digest",
    "search_algorithms": "research_digest",
    "get_symbol_blast_radius": "ast_slice",
    "index_codebase_graph": "ast_slice",
    "index_repo": "ast_slice",
    "analyze_code": "ast_slice",
    "patch_file": "patch_diff",
}

_PRODUCER_DISPLAY_MAP = {
    "codex": "CodexAgent",
    "research": "ResearchAgent",
    "db": "DBAgent",
}


def _is_tool_failure(tool_name: str, output: str) -> bool:
    """Detect whether a tool execution output represents an error/failure."""
    text = (output or "").strip()
    if not text:
        return False
    failure_prefixes = (
        "Execution Error:",
        "patch_file ERROR:",
        "Error:",
        "Error ",
        "Syntax Error",
        "Security Error:",
        "Tool error",
        "run_tests ERROR:",
        "Failed to scrape",
        "Search error:",
        "Algorithm search error:",
        "DB Error:",
    )
    if text.startswith(failure_prefixes):
        return True
    if tool_name in ("execute_code", "patch_file") and (
        "Traceback (most recent call last):" in text
        or re.search(r"\b\w*(?:Error|Exception)\s*[:(]", text)
    ):
        return True
    return False


def _extract_failing_symbol_hint(tool_calls: list[dict], error_text: str) -> str:
    """Best-effort extraction of a symbol or file identifier for blast-radius diagnosis."""
    m = re.search(r"in (\w+)\b", error_text)
    if m and m.group(1) not in {"the", "file", "line", "module"}:
        return m.group(1)
    for tc in tool_calls:
        args = tc.get("args") or {}
        if isinstance(args, dict):
            for k in ("symbol_name", "path", "filepath", "code"):
                val = str(args.get(k, "")).strip()
                if val:
                    fn_m = re.search(r"\bdef\s+([a-zA-Z_]\w*)", val)
                    if fn_m:
                        return fn_m.group(1)
                    return val.splitlines()[0][:40]
    return "target_symbol"


async def _execute_and_record_tools(
    tool_calls: list[dict],
    state: dict,
    user_id: int = 1,
    thread_id: str = "default",
    session: Any = None,
    *,
    config: Optional[RunnableConfig] = None,
    agent_name: str = "codex",
    use_memory: bool = False,
    memory_state: Optional[dict] = None,
    compressor_llm: Any = None,
) -> List[BaseMessage]:
    """
    Unified tool dispatcher with:
      1. Automatic Pass-by-Reference `ContextArtifact` offloading for outputs > 250 tokens (> 1000 chars).
      2. Self-Healing Circuit Breaker tracking `(tool_name, args_hash, stderr_hash)` in `state["failure_signatures"]`.
      3. Optional persistent Memory Token compression via `update_memory`.
    """
    results: List[BaseMessage] = []
    failure_sigs: List[Tuple[str, str, str]] = list(state.get("failure_signatures") or [])
    producer_agent = _PRODUCER_DISPLAY_MAP.get(agent_name, "ToolRunner")

    eff_config = config or RunnableConfig(
        configurable={"user_id": user_id, "thread_id": thread_id}
    )

    for idx, tc in enumerate(tool_calls):
        tool_name = tc.get("name", "")
        tool_args = tc.get("args") or {}
        tool_id = tc.get("id") or f"call_{idx}_{uuid.uuid4().hex[:8]}"

        tool_fn = _ALL_TOOLS.get(tool_name)
        if tool_fn is None:
            result_content = f"Error: Unknown tool '{tool_name}'"
        else:
            try:
                if hasattr(tool_fn, "ainvoke"):
                    result_content = str(await tool_fn.ainvoke(tool_args, config=eff_config))
                elif callable(tool_fn):
                    maybe_coro = tool_fn(**tool_args)
                    result_content = str(await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro)
                else:
                    result_content = f"Error: Tool '{tool_name}' is not callable"
            except Exception as e:
                result_content = f"Execution Error: Tool error in '{tool_name}': {repr(e)}"

        # ── Self-Healing Circuit Breaker Evaluation ──────────────────────────
        failed = _is_tool_failure(tool_name, result_content)
        tripped = False
        if failed:
            args_hash = hashlib.sha256(
                json.dumps(tool_args, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
            stderr_hash = hashlib.sha256(
                result_content.strip().encode("utf-8")
            ).hexdigest()[:12]
            sig = (tool_name, args_hash, stderr_hash)
            failure_sigs.append(sig)
            state["failure_signatures"] = failure_sigs

            # Condition 1: execute_code or patch_file fails twice consecutively with identical (args_hash, stderr_hash)
            identical_repeat = (
                tool_name in ("execute_code", "patch_file")
                and len(failure_sigs) >= 2
                and failure_sigs[-1] == failure_sigs[-2]
            )
            # Condition 2: consecutive failures reach MAX_CONSECUTIVE_TOOL_FAILURES (3)
            max_exceeded = len(failure_sigs) >= MAX_CONSECUTIVE_TOOL_FAILURES

            if identical_repeat or max_exceeded:
                tripped = True
                state["circuit_breaker_tripped"] = True
        else:
            failure_sigs = []
            state["failure_signatures"] = []
            state["circuit_breaker_tripped"] = False

        # ── Pass-by-Reference Artifact Offloading (> 250 tokens / 1000 chars) ─
        message_content = result_content
        if (
            len(result_content) > ARTIFACT_OFFLOAD_CHAR_THRESHOLD
            and tool_name != "read_context_artifact"
        ):
            artifact_type = _TOOL_ARTIFACT_TYPE_MAP.get(tool_name, "tool_output")
            art = await store_context_artifact(
                session=session,
                user_id=user_id,
                thread_id=thread_id,
                producer_agent=producer_agent,
                artifact_type=artifact_type,
                payload=result_content,
            )
            message_content = (
                f"[Artifact ref: {art['artifact_id']} | "
                f"Tokens: {art['token_count']} | "
                f"Summary: {art['summary']}]"
            )

        if use_memory and memory_state is not None and compressor_llm is not None:
            memory_state_updated = await update_memory(
                current=memory_state,
                tool_name=tool_name,
                tool_input=str(tool_args),
                tool_result=message_content,
                compressor_llm=compressor_llm,
            )
            memory_state.clear()
            memory_state.update(memory_state_updated)

        results.append(
            ToolMessage(
                content=message_content,
                name=tool_name,
                tool_call_id=tool_id,
            )
        )

        if tripped:
            symbol_hint = _extract_failing_symbol_hint(tool_calls, result_content)
            directive = SystemMessage(
                content=(
                    f"[CIRCUIT BREAKER TRIPPED] Tool `{tool_name}` failed repeatedly with identical "
                    f"or consecutive errors (signature={failure_sigs[-1][1]}:{failure_sigs[-1][2]}). "
                    f"Halt redundant retries immediately. Call `get_symbol_blast_radius('{symbol_hint}')` "
                    f"to inspect upstream/downstream callers before attempting a different fix, or "
                    f"escalate cleanly to the user with the root cause."
                )
            )
            results.append(directive)
            break

    return results


async def worker_node(
    state: AgentState,
    config: RunnableConfig,
    agent_name: str,
    system_prompt: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    use_memory: bool = False,
) -> dict:
    """
    ReAct (Reason + Act) loop with Pass-by-Reference Artifact Offloading,
    Sliding-Window Context Compaction, and Self-Healing Circuit Breaker.
    """
    configurable = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {}) or {}
    user_id = int(configurable.get("user_id") or 1)
    thread_id = str(configurable.get("thread_id") or "default")

    raw_conversation = list(state.get("messages", []))
    conversation, _ = await compact_conversation_window(
        raw_conversation,
        session=None,
        user_id=user_id,
        thread_id=thread_id,
    )
    working: list[BaseMessage] = []
    steps = 0

    mutable_state: dict = {
        "messages": conversation,
        "failure_signatures": list(state.get("failure_signatures") or []),
        "circuit_breaker_tripped": bool(state.get("circuit_breaker_tripped", False)),
    }

    tools = (
        codex_tools
        if agent_name == "codex"
        else (research_tools if agent_name == "research" else (db_tools if agent_name == "db" else []))
    )

    agent_llm = await get_agent_llm(agent_name, config, tools=tools)
    routing_llm = await get_agent_llm("routing", config) if use_memory else None
    memory_state = load_memory() if use_memory else None

    # ── PLANNING PASS (Codex only) ──────────────────────────────────────────
    if use_memory:
        memory_sys_msg = [memory_to_system_message(memory_state)] if memory_state else []
        plan_prompt = system_prompt + "\n\n" + PLANNING_INJECTION

        plan_response: AIMessage = await agent_llm.ainvoke(
            [SystemMessage(content=plan_prompt)] + memory_sys_msg + conversation,
            config=config,
        )
        working.append(plan_response)
        log.info("Codex planning pass completed — plan: %s", plan_response.content[:200])

        if plan_response.tool_calls:
            steps += 1
            tool_msgs = await _execute_and_record_tools(
                plan_response.tool_calls,
                mutable_state,
                user_id=user_id,
                thread_id=thread_id,
                session=None,
                config=config,
                agent_name=agent_name,
                use_memory=use_memory,
                memory_state=memory_state,
                compressor_llm=routing_llm,
            )
            working.extend(tool_msgs)

    # ── MAIN ReAct LOOP ─────────────────────────────────────────────────────
    while steps < max_steps:
        steps += 1
        memory_sys_msg = [memory_to_system_message(memory_state)] if (use_memory and memory_state) else []

        response: AIMessage = await agent_llm.ainvoke(
            [SystemMessage(content=system_prompt)] + memory_sys_msg + conversation + working,
            config=config,
        )
        working.append(response)

        if not response.tool_calls:
            break

        tool_msgs = await _execute_and_record_tools(
            response.tool_calls,
            mutable_state,
            user_id=user_id,
            thread_id=thread_id,
            session=None,
            config=config,
            agent_name=agent_name,
            use_memory=use_memory,
            memory_state=memory_state,
            compressor_llm=routing_llm,
        )
        working.extend(tool_msgs)

        if mutable_state.get("circuit_breaker_tripped"):
            log.warning("Circuit breaker tripped in %s; halting redundant retries.", agent_name)

    if steps >= max_steps and working and getattr(working[-1], "tool_calls", None):
        working.append(AIMessage(content=f"[{max_steps}-step limit reached. Reporting current state.]"))

    return {
        "messages": working,
        "failure_signatures": mutable_state.get("failure_signatures", []),
        "circuit_breaker_tripped": mutable_state.get("circuit_breaker_tripped", False),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor node
# ══════════════════════════════════════════════════════════════════════════════


async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Commenter Supervisor: routes to specialists then writes the final synthesis."""
    messages = list(state["messages"])
    iterations = state.get("iterations", 0)

    llm = await get_agent_llm("supervisor", config)
    routing_structured = await get_agent_llm("routing", config, structured_schema=Route)

    if iterations >= MAX_SUPERVISOR_ITERATIONS:
        final = await llm.ainvoke(
            [SystemMessage(content=COMMENTER_SYNTHESIS_PROMPT)] + messages,
            config=config,
        )
        return {"messages": [final], "next": "FINISH", "iterations": iterations}

    route = await routing_structured.ainvoke(
        [SystemMessage(content=COMMENTER_ROUTING_PROMPT)] + messages,
        config=config,
    )
    next_agent = route.next if isinstance(route, Route) else "FINISH"

    if next_agent == "FINISH":
        final = await llm.ainvoke(
            [SystemMessage(content=COMMENTER_SYNTHESIS_PROMPT)] + messages,
            config=config,
        )
        return {"messages": [final], "next": "FINISH", "iterations": iterations + 1}

    return {"next": next_agent, "iterations": iterations + 1}


# ── Build Graph ─────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node(
    "CodexEngine",
    partial(
        worker_node,
        agent_name="codex",
        system_prompt=CODEX_SYSTEM_PROMPT,
        max_steps=CODEX_MAX_STEPS,
        use_memory=True,
    ),
)
workflow.add_node(
    "ResearchAgent",
    partial(
        worker_node,
        agent_name="research",
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        max_steps=DEFAULT_MAX_STEPS,
        use_memory=False,
    ),
)
workflow.add_node(
    "DBAgent",
    partial(
        worker_node,
        agent_name="db",
        system_prompt=DB_SYSTEM_PROMPT,
        max_steps=DEFAULT_MAX_STEPS,
        use_memory=False,
    ),
)

workflow.add_edge("CodexEngine", "Supervisor")
workflow.add_edge("ResearchAgent", "Supervisor")
workflow.add_edge("DBAgent", "Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "CodexEngine": "CodexEngine",
        "ResearchAgent": "ResearchAgent",
        "DBAgent": "DBAgent",
        "FINISH": END,
    },
)
workflow.set_entry_point("Supervisor")

graph = workflow

__all__ = ["graph", "AgentState"]
