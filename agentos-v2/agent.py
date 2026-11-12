"""
agent.py — AgentOS multi-agent orchestration via LangGraph.

Architecture:
  Commenter Supervisor ─┬─► CodexEngine    (ReAct loop: plan → act → observe → repeat)
                        ├─► ResearchAgent  (web search, scraping, algorithm lookup)
                        └─► DBAgent        (snippets, changelog)

Key design changes vs v1:
  • worker_node is now a ReAct loop — the agent keeps calling tools until it
    decides it's done (no more tool calls), not just one round. This is what
    makes Codex feel like Codex: read file → analyse → patch → run tests →
    read failures → patch again → run tests → green → done.
  • Codex gets a separate inner iteration cap (CODEX_MAX_STEPS) independent
    of the supervisor's routing budget, so complex tasks get enough room.
  • A planning pass is injected before Codex's first tool call so it
    reasons about the full task before acting (reduces wasted steps).
  • edit_file is replaced by patch_file everywhere.
"""

import logging
import operator
import os
from functools import partial
from typing import Annotated, Literal, Sequence, TypedDict

log = logging.getLogger("agentos")

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

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

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from memory_manager import load_memory, memory_to_system_message, update_memory

from tools import (
    analyze_code,
    execute_code,
    get_snippets,
    index_repo,
    list_directory,
    log_improvement,
    patch_file,
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

# ── LLM Factory ─────────────────────────────────────────────────────────────────

def _is_valid_key(key: str | None) -> bool:
    if not key:
        return False
    k = key.strip()
    placeholders = {
        "your_anthropic_key_here",
        "your_openai_key_here",
        "your_gemini_key_here",
        "placeholder",
        ""
    }
    if k in placeholders:
        return False
    if "your_" in k.lower() or "placeholder" in k.lower():
        return False
    return True

class MissingAPIKeysModel:
    def invoke(self, *args, **kwargs):
        raise ValueError("PAID_API_KEYS_MISSING_OR_INVALID")
    async def ainvoke(self, *args, **kwargs):
        raise ValueError("PAID_API_KEYS_MISSING_OR_INVALID")
    def bind_tools(self, *args, **kwargs):
        return self
    def with_structured_output(self, *args, **kwargs):
        return self

class LocalContextExceededModel:
    def invoke(self, *args, **kwargs):
        raise ValueError("LOCAL_CONTEXT_WINDOW_EXCEEDED")
    async def ainvoke(self, *args, **kwargs):
        raise ValueError("LOCAL_CONTEXT_WINDOW_EXCEEDED")
    def bind_tools(self, *args, **kwargs):
        return self
    def with_structured_output(self, *args, **kwargs):
        return self

def ensure_local_server(runnable):
    from langchain_core.language_models.chat_models import BaseChatModel
    underlying = runnable
    while True:
        if isinstance(underlying, BaseChatModel):
            break
        if hasattr(underlying, "bound"):
            underlying = underlying.bound
        elif hasattr(underlying, "bound_model"):
            underlying = underlying.bound_model
        elif hasattr(underlying, "model"):
            if isinstance(underlying.model, str):
                break
            underlying = underlying.model
        else:
            break
    
    api_base = ""
    if hasattr(underlying, "openai_api_base") and underlying.openai_api_base:
        api_base = str(underlying.openai_api_base)
    elif hasattr(underlying, "base_url") and underlying.base_url:
        api_base = str(underlying.base_url)
    if "8090" in api_base:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(("127.0.0.1", 8090))
                # Socket is open, server is running or port is in use! Do not attempt to start it again.
                return
        except Exception:
            pass
        
        try:
            from codex_engine import start_codex_server
            log.info("Starting embedded Codex Engine on port 8090...")
            start_codex_server(port=8090, background=True)
            import time
            for _ in range(5):
                try:
                    resp = httpx.get("http://localhost:8090/health", timeout=1.0)
                    if resp.status_code == 200:
                        log.info("Embedded Codex Engine started successfully!")
                        break
                except Exception:
                    time.sleep(1.0)
        except Exception as e:
            log.error("Failed to start embedded Codex Engine: %s", e)

class FallbackRunnable:
    def __init__(self, runnables: list):
        self.runnables = runnables

    def invoke(self, input, config=None, **kwargs):
        last_exception = None
        for r in self.runnables:
            try:
                ensure_local_server(r)
                return r.invoke(input, config=config, **kwargs)
            except Exception as e:
                log.warning("Fallback triggered: failed with %s. Trying next...", e)
                last_exception = e
        if last_exception:
            raise last_exception
        raise ValueError("No runnables in fallback chain")

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

from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pathlib import Path

class G4FChatModel(BaseChatModel):
    model_name: str
    tools: Optional[List[Any]] = None
    
    def __init__(self, model_name: str, tools: Optional[List[Any]] = None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        self.tools = tools

    def bind_tools(self, tools, **kwargs):
        return G4FChatModel(model_name=self.model_name, tools=tools)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import g4f
        from g4f.client import Client
        import json
        import uuid
        
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

        g4f_model = self.model_name
        if "gemini" in self.model_name.lower():
            g4f_model = "gemini-2.5-flash"
        elif "claude" in self.model_name.lower():
            g4f_model = "claude-3.5-sonnet"
        elif "gpt-4" in self.model_name.lower():
            g4f_model = "gpt-4o"
        elif "gpt-3.5" in self.model_name.lower() or "mini" in self.model_name.lower():
            g4f_model = "gpt-4o-mini"

        tools_arg = None
        if self.tools:
            from langchain_core.utils.function_calling import convert_to_openai_tool
            tools_arg = [convert_to_openai_tool(t) for t in self.tools]

        import concurrent.futures
        import g4f
        from g4f.Provider import Yqcloud, OperaAria
        
        if "gpt" in g4f_model.lower():
            providers = [Yqcloud, OperaAria, None]
        else:
            providers = [Yqcloud, None]
            
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
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        client.chat.completions.create,
                        **call_kwargs
                    )
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
            
        try:
            choice = response.choices[0]
            content = choice.message.content or ""
            
            tool_calls = []
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        tool_calls.append({
                            "name": tc.function.name,
                            "args": args,
                            "id": tc.id or f"call_{uuid.uuid4().hex}"
                        })
                    except Exception:
                        pass
            
            ai_message = AIMessage(content=content, tool_calls=tool_calls)
        except Exception as e:
            log.error("g4f _generate parsing error: %s", e)
            raise e
            
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import g4f
        from g4f.client import AsyncClient
        import json
        import uuid
        
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

        g4f_model = self.model_name
        if "gemini" in self.model_name.lower():
            g4f_model = "gemini-2.5-flash"
        elif "claude" in self.model_name.lower():
            g4f_model = "claude-3.5-sonnet"
        elif "gpt-4" in self.model_name.lower():
            g4f_model = "gpt-4o"
        elif "gpt-3.5" in self.model_name.lower() or "mini" in self.model_name.lower():
            g4f_model = "gpt-4o-mini"

        tools_arg = None
        if self.tools:
            from langchain_core.utils.function_calling import convert_to_openai_tool
            tools_arg = [convert_to_openai_tool(t) for t in self.tools]

        import asyncio
        import g4f
        from g4f.Provider import Yqcloud, OperaAria
        
        if "gpt" in g4f_model.lower():
            providers = [Yqcloud, OperaAria, None]
        else:
            providers = [Yqcloud, None]
            
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
                    timeout=15.0
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
            
        try:
            choice = response.choices[0]
            content = choice.message.content or ""
            
            tool_calls = []
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                        tool_calls.append({
                            "name": tc.function.name,
                            "args": args,
                            "id": tc.id or f"call_{uuid.uuid4().hex}"
                        })
                    except Exception:
                        pass
            
            ai_message = AIMessage(content=content, tool_calls=tool_calls)
        except Exception as e:
            log.error("g4f _agenerate parsing error: %s", e)
            raise e
            
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def with_structured_output(self, schema, **kwargs):
        from pydantic import BaseModel
        import json
        import re

        def parse_output(response_message) -> Any:
            content = response_message.content.strip()
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1:
                        json_str = content[start:end+1]
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
                    from pydantic_core import PydanticUndefined
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

        system_instruction = f"Output must be a valid JSON object matching the schema: {schema.schema_json() if hasattr(schema, 'schema_json') else schema.model_json_schema()}"
        
        class StructuredModel:
            def __init__(self, model):
                self.model = model
            async def ainvoke(self, messages, config=None, **kwargs):
                new_messages = [SystemMessage(content=system_instruction)] + list(messages)
                res = await self.model.ainvoke(new_messages, **kwargs)
                return parse_output(res)
            def invoke(self, messages, config=None, **kwargs):
                new_messages = [SystemMessage(content=system_instruction)] + list(messages)
                res = self.model.invoke(new_messages, **kwargs)
                return parse_output(res)

        return StructuredModel(self)

    @property
    def _llm_type(self) -> str:
        return "g4f"

# ── Dynamic Configuration, Token Tracking & Context Pruning ──────────────────────────

def get_local_env_settings() -> dict:
    """Helper to build settings from environment variables for local/offline execution."""
    providers = ["openai", "gemini", "anthropic"]
    return {
        "active_mode": "paid" if os.getenv("USE_PAID_API", "false").lower() == "true" else ("free" if os.getenv("USE_FREE_VERSIONS", "true").lower() == "true" else "local"),
        "selected_providers": providers,
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "token_budget": int(os.getenv("TOKEN_BUDGET", "100000")),
        "token_used": int(os.getenv("TOKEN_USED", "0")),
        "token_warning_threshold": 0.90,
        "token_warning_sent": False,
        "token_auto_fallback": True,
        "local_max_context_tokens": int(os.getenv("LOCAL_MAX_CONTEXT_TOKENS", "2048")),
        "local_auto_delete": True,
        "language": "en"
    }

def prune_messages_to_limit(messages: list, limit: int, user_id: int | None) -> tuple[list, list]:
    """Prunes oldest messages if total tokens exceed limit, saving pruned logs locally."""
    from settings_manager import estimate_tokens
    import json
    
    system_msgs = []
    chat_msgs = []
    for m in messages:
        if isinstance(m, SystemMessage) or (hasattr(m, "type") and m.type == "system"):
            system_msgs.append(m)
        else:
            chat_msgs.append(m)
            
    def total_tokens(sys, chat):
        return estimate_tokens("".join(getattr(m, "content", "") or "" for m in sys + chat))
        
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
                "tool_calls": getattr(m, "tool_calls", None)
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
        log.info("Local model context limit reached. Pruned %s oldest messages and archived to %s", len(pruned), filename)
        
    return system_msgs + chat_msgs, pruned

class TokenTrackingModel:
    """Wrapper that estimates tokens consumed, logs them to settings, and truncates local context."""
    def __init__(self, bound_model, user_id: int | None, is_local: bool = False, max_context: int = 2048, auto_delete: bool = True):
        self.bound_model = bound_model
        self.user_id = user_id
        self.is_local = is_local
        self.max_context = max_context
        self.auto_delete = auto_delete

    async def ainvoke(self, messages, config=None, **kwargs):
        from settings_manager import estimate_tokens, add_user_tokens
        
        # 1. Truncate context if local
        if self.is_local:
            messages_to_use, pruned = prune_messages_to_limit(messages, self.max_context, self.user_id)
            if not self.auto_delete and pruned:
                raise ValueError("LOCAL_CONTEXT_WINDOW_EXCEEDED")
            messages = messages_to_use

        # 2. Count input tokens
        input_text = ""
        if isinstance(messages, list):
            for m in messages:
                input_text += getattr(m, "content", "") or ""
        else:
            input_text = str(messages)
        input_tokens = estimate_tokens(input_text)

        # 3. Call model
        res = await self.bound_model.ainvoke(messages, config=config, **kwargs)

        # 4. Count output tokens
        output_text = getattr(res, "content", "") or ""
        output_tokens = estimate_tokens(output_text)
        
        total_tokens = input_tokens + output_tokens
        if self.user_id:
            await add_user_tokens(self.user_id, total_tokens)
        else:
            log.info("Local session consumed %s tokens.", total_tokens)
            
        return res

    def invoke(self, messages, config=None, **kwargs):
        from settings_manager import estimate_tokens
        if self.is_local:
            messages_to_use, pruned = prune_messages_to_limit(messages, self.max_context, self.user_id)
            if not self.auto_delete and pruned:
                raise ValueError("LOCAL_CONTEXT_WINDOW_EXCEEDED")
            messages = messages_to_use

        input_text = ""
        if isinstance(messages, list):
            for m in messages:
                input_text += getattr(m, "content", "") or ""
        else:
            input_text = str(messages)
        input_tokens = estimate_tokens(input_text)

        res = self.bound_model.invoke(messages, config=config, **kwargs)

        output_text = getattr(res, "content", "") or ""
        output_tokens = estimate_tokens(output_text)
        
        total_tokens = input_tokens + output_tokens
        if self.user_id:
            from settings_manager import add_user_tokens
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(add_user_tokens(self.user_id, total_tokens))
                else:
                    asyncio.run(add_user_tokens(self.user_id, total_tokens))
            except Exception:
                pass
        else:
            log.info("Local session consumed %s tokens.", total_tokens)
            
        return res

# ── Dynamic Model Retrieval ───────────────────────────────────────────────────────

async def get_agent_llm(agent_name: str, config, tools=None, structured_schema=None):
    """
    Builds the model fallback chain dynamically based on the current user's settings.
    Integrates TokenTrackingModel wrapper per runnable.
    """
    from settings_manager import get_user_settings
    
    configurable = config.get("configurable", {}) if config else {}
    user_id = configurable.get("user_id")
    
    if user_id:
        settings = await get_user_settings(user_id)
    else:
        settings = get_local_env_settings()

    active_mode = settings["active_mode"]
    providers = settings["selected_providers"]
    max_context = settings.get("local_max_context_tokens", 2048)
    auto_delete = settings.get("local_auto_delete", True)

    chain = []
    
    def _prepare(model):
        if tools:
            model = model.bind_tools(tools)
        if structured_schema:
            if hasattr(model, "_llm_type") and model._llm_type == "g4f":
                model = model.with_structured_output(structured_schema)
            else:
                model = wrap_robust_structured_output(model, structured_schema)
        is_local = (active_mode == "local")
        return TokenTrackingModel(model, user_id, is_local=is_local, max_context=max_context, auto_delete=auto_delete)

    # 1. Local Codex mode
    if active_mode == "local":
        local_url = os.getenv("LOCAL_CODEX_URL", "http://localhost:8090/v1")
        model = ChatOpenAI(
            base_url=local_url,
            model=os.getenv("LOCAL_CODEX_MODEL", "qwen2.5-coder-7b-instruct"),
            api_key="not-needed",
            temperature=0.1,
            max_tokens=4096,
        )
        ensure_local_server(model)
        chain.append(_prepare(model))
        
    # 2. Paid API mode
    elif active_mode == "paid":
        if "openai" in providers and _is_valid_key(settings["openai_api_key"]):
            if ChatOpenAI is None:
                log.warning("ChatOpenAI requested but langchain-openai is not installed/working.")
            else:
                model_name = "gpt-4o" if agent_name in ("codex", "research", "supervisor") else "gpt-4o-mini"
                chain.append(_prepare(ChatOpenAI(
                    model=model_name,
                    api_key=settings["openai_api_key"],
                    temperature=0.1 if agent_name != "routing" else 0.0,
                )))
        if "anthropic" in providers and _is_valid_key(settings["anthropic_api_key"]):
            if ChatAnthropic is None:
                log.warning("ChatAnthropic requested but langchain-anthropic is not installed/working.")
            else:
                model_name = "claude-3-7-sonnet-latest" if agent_name in ("codex", "research", "supervisor") else "claude-3-5-haiku-20241022"
                chain.append(_prepare(ChatAnthropic(
                    model=model_name,
                    api_key=settings["anthropic_api_key"],
                    temperature=0.1 if agent_name != "routing" else 0.0,
                )))
        if "gemini" in providers and _is_valid_key(settings["gemini_api_key"]):
            if ChatGoogleGenerativeAI is None:
                log.warning("ChatGoogleGenerativeAI requested but langchain-google-genai is not installed/working.")
            else:
                model_name = "gemini-2.5-pro" if agent_name in ("codex", "research", "supervisor") else "gemini-2.5-flash"
                chain.append(_prepare(ChatGoogleGenerativeAI(
                    model=model_name,
                    api_key=settings["gemini_api_key"],
                    temperature=0.1 if agent_name != "routing" else 0.0,
                )))
            
        if not chain:
            chain.append(TokenTrackingModel(MissingAPIKeysModel(), user_id))

    # 3. Free API mode (g4f web scrapers)
    else:
        try:
            import g4f
            if "openai" in providers:
                model_name = "gpt-4o" if agent_name in ("codex", "research", "supervisor") else "gpt-4o-mini"
                chain.append(_prepare(G4FChatModel(model_name=model_name)))
            if "anthropic" in providers:
                chain.append(_prepare(G4FChatModel(model_name="claude-3.5-sonnet")))
            if "gemini" in providers:
                model_name = "gemini-2.5-pro" if agent_name in ("codex", "research", "supervisor") else "gemini-2.5-flash"
                chain.append(_prepare(G4FChatModel(model_name=model_name)))
        except ImportError:
            pass
            
        if ChatOpenAI is not None:
            local_url = os.getenv("LOCAL_CODEX_URL", "http://localhost:8090/v1")
            local_model = ChatOpenAI(
                base_url=local_url,
                model=os.getenv("LOCAL_CODEX_MODEL", "qwen2.5-coder-7b-instruct"),
                api_key="not-needed",
                temperature=0.1,
                max_tokens=4096,
            )
            chain.append(_prepare(local_model))
        else:
            log.warning("ChatOpenAI is not available. Local Codex model will not be added to fallbacks.")

    if len(chain) > 1:
        return FallbackRunnable(chain)
    return chain[0]

def wrap_robust_structured_output(model, schema):
    from pydantic import BaseModel
    import json
    import re
    
    def parse_output(response_message) -> Any:
        content = response_message.content.strip()
        
        # Heuristic check for local model outputting tool call format instead of Route JSON
        if schema.__name__ == "Route":
            content_lower = content.lower()
            reasoning = f"Heuristic override based on model content: {content[:100]}"
            
            # DB tools
            db_keywords = ["save_code_snippet", "get_snippets", "log_improvement", "save_snippet"]
            if any(kw in content_lower for kw in db_keywords):
                log.info("Heuristically routing to DBAgent because model outputted: %s", content)
                return schema(next="DBAgent", reasoning=reasoning)
                
            # Research tools
            research_keywords = ["web_search", "scrape_page", "search_algorithms"]
            if any(kw in content_lower for kw in research_keywords):
                log.info("Heuristically routing to ResearchAgent because model outputted: %s", content)
                return schema(next="ResearchAgent", reasoning=reasoning)
                
            # Codex tools (explicit keywords OR any functions. prefix)
            codex_keywords = [
                "write_file", "patch_file", "read_file", "execute_code", "run_tests", 
                "list_directory", "search_codebase", "run_shell_command", "index_repo", "delete_file", "append_to_file"
            ]
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
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1:
                    json_str = content[start:end+1]
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
                from pydantic_core import PydanticUndefined
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

    system_instruction = f"Output must be a valid JSON object matching the schema: {schema.schema_json() if hasattr(schema, 'schema_json') else schema.model_json_schema()}"
    
    class RobustStructuredModel:
        def __init__(self, bound_model):
            self.bound_model = bound_model
            
        async def ainvoke(self, messages, config=None, **kwargs):
            from langchain_core.messages import SystemMessage
            new_messages = [SystemMessage(content=system_instruction)] + list(messages)
            res = await self.bound_model.ainvoke(new_messages, **kwargs)
            return parse_output(res)
            
        def invoke(self, messages, config=None, **kwargs):
            from langchain_core.messages import SystemMessage
            new_messages = [SystemMessage(content=system_instruction)] + list(messages)
            res = self.bound_model.invoke(new_messages, **kwargs)
            return parse_output(res)
            
    return RobustStructuredModel(model)

# LLMs will be instantiated dynamically inside the nodes now

# ── State ───────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:   Annotated[Sequence[BaseMessage], operator.add]
    next:       str
    iterations: int   # supervisor routing budget (resets per ainvoke)


# ── Routing schema ──────────────────────────────────────────────────────────────

class Route(BaseModel):
    next: Literal["FINISH", "CodexEngine", "ResearchAgent", "DBAgent"] = Field(
        description="The specialist to call next, or FINISH to write the final response."
    )
    reasoning: str = Field(description="One-sentence explanation for this routing decision.")


# ── System prompts ──────────────────────────────────────────────────────────────

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


# ── Safety limits ───────────────────────────────────────────────────────────────

MAX_SUPERVISOR_ITERATIONS = 6   # max specialist invocations per user turn
CODEX_MAX_STEPS           = 12  # max ReAct steps inside one Codex invocation
DEFAULT_MAX_STEPS         = 6   # max ReAct steps for Research / DB agents


# ── Tool maps ────────────────────────────────────────────────────────────────────

_ALL_TOOLS = {
    # Codex
    "execute_code":      execute_code,
    "analyze_code":      analyze_code,
    "read_file":         read_file,
    "list_directory":    list_directory,
    "write_file":        write_file,
    "patch_file":        patch_file,
    "search_codebase":   search_codebase,
    "run_tests":         run_tests,
    "run_shell_command": run_shell_command,
    "index_repo":        index_repo,
    # Research
    "web_search":        web_search,
    "scrape_page":       scrape_page,
    "search_algorithms": search_algorithms,
    # DB
    "save_code_snippet": save_code_snippet,
    "get_snippets":      get_snippets,
    "log_improvement":   log_improvement,
}


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


async def worker_node(
    state: AgentState,
    config: RunnableConfig,
    agent_name: str,
    system_prompt: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    use_memory: bool = False,
) -> dict:
    """
    ReAct (Reason + Act) loop with optional persistent Memory Tokens.

    When use_memory=True (Codex), a **planning pass** is enforced on the
    first iteration: the LLM must output a plan with NO tool calls before
    it starts acting.  This prevents wasted steps on complex tasks.
    """
    conversation = list(state["messages"])
    working:  list[BaseMessage] = []
    steps = 0
    plan_done = False

    # Dynamic LLM resolution
    tools = []
    if agent_name == "codex":
        tools = codex_tools
    elif agent_name == "research":
        tools = research_tools
    elif agent_name == "db":
        tools = db_tools

    agent_llm = await get_agent_llm(agent_name, config, tools=tools)
    routing_llm = await get_agent_llm("routing", config)

    # Load memory if this agent uses it (e.g. Codex)
    memory_state = load_memory() if use_memory else None

    # ── PLANNING PASS (Codex only) ──────────────────────────────────────────
    if use_memory:
        memory_sys_msg = [memory_to_system_message(memory_state)] if memory_state else []
        plan_prompt = system_prompt + "\n\n" + PLANNING_INJECTION

        plan_response: AIMessage = await agent_llm.ainvoke(
            [SystemMessage(content=plan_prompt)] + memory_sys_msg + conversation
        )
        working.append(plan_response)
        plan_done = True
        log.info("Codex planning pass completed — plan: %s", plan_response.content[:200])

        # If the LLM ignored us and made tool calls in the plan, that's fine —
        # treat it as step 1 of the ReAct loop (don't double-count).
        if plan_response.tool_calls:
            steps += 1
            tool_results: list[ToolMessage] = []
            for tc in plan_response.tool_calls:
                tool_fn = _ALL_TOOLS.get(tc["name"])
                if tool_fn is None:
                    result_content = f"Unknown tool: '{tc['name']}'"
                else:
                    try:
                        result_content = str(await tool_fn.ainvoke(tc["args"]))
                    except Exception as e:
                        result_content = f"Tool error in '{tc['name']}': {repr(e)}"
                if memory_state is not None:
                    memory_state = await update_memory(
                        current=memory_state,
                        tool_name=tc["name"],
                        tool_input=str(tc["args"]),
                        tool_result=result_content,
                        compressor_llm=routing_llm,
                    )
                tool_results.append(ToolMessage(
                    content=result_content, name=tc["name"], tool_call_id=tc["id"],
                ))
            working.extend(tool_results)

    # ── MAIN ReAct LOOP ─────────────────────────────────────────────────────
    while steps < max_steps:
        steps += 1

        # Inject memory tokens directly before inference
        memory_sys_msg = []
        if use_memory and memory_state is not None:
            memory_sys_msg = [memory_to_system_message(memory_state)]

        # Invoke the specialist
        response: AIMessage = await agent_llm.ainvoke(
            [SystemMessage(content=system_prompt)] + memory_sys_msg + conversation + working
        )
        working.append(response)

        # No tool calls → LLM is done reasoning
        if not response.tool_calls:
            break

        # Execute tools
        tool_results: list[ToolMessage] = []
        for tc in response.tool_calls:
            tool_fn = _ALL_TOOLS.get(tc["name"])
            if tool_fn is None:
                result_content = f"Unknown tool: '{tc['name']}'"
            else:
                try:
                    result_content = str(await tool_fn.ainvoke(tc["args"]))
                except Exception as e:
                    result_content = f"Tool error in '{tc['name']}': {repr(e)}"

            # COMPRESS AND UPDATE MEMORY TOKENS
            if use_memory and memory_state is not None:
                memory_state = await update_memory(
                    current=memory_state,
                    tool_name=tc["name"],
                    tool_input=str(tc["args"]),
                    tool_result=result_content,
                    compressor_llm=routing_llm,
                )

            tool_results.append(
                ToolMessage(
                    content=result_content,
                    name=tc["name"],
                    tool_call_id=tc["id"],
                )
            )

        working.extend(tool_results)

    if steps >= max_steps and working and getattr(working[-1], 'tool_calls', None):
        working.append(AIMessage(
            content=f"[{max_steps}-step limit reached. Reporting current state.]"
        ))

    return {"messages": working}


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor node
# ══════════════════════════════════════════════════════════════════════════════

async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Commenter Supervisor: routes to specialists then writes the final synthesis."""
    messages   = list(state["messages"])
    iterations = state.get("iterations", 0)

    # Dynamic LLM resolution
    llm = await get_agent_llm("supervisor", config)
    routing_structured = await get_agent_llm("routing", config, structured_schema=Route)

    # Safety: force synthesis if routing budget exhausted
    if iterations >= MAX_SUPERVISOR_ITERATIONS:
        final = await llm.ainvoke(
            [SystemMessage(content=COMMENTER_SYNTHESIS_PROMPT)] + messages
        )
        return {"messages": [final], "next": "FINISH", "iterations": iterations}

    # Route using the faster flash model
    route = await routing_structured.ainvoke(
        [SystemMessage(content=COMMENTER_ROUTING_PROMPT)] + messages
    )
    next_agent = route.next if isinstance(route, Route) else "FINISH"

    if next_agent == "FINISH":
        final = await llm.ainvoke(
            [SystemMessage(content=COMMENTER_SYNTHESIS_PROMPT)] + messages
        )
        return {"messages": [final], "next": "FINISH", "iterations": iterations + 1}

    return {"next": next_agent, "iterations": iterations + 1}


# ── Bind tools to LLMs ──────────────────────────────────────────────────────────

codex_tools    = [execute_code, analyze_code, read_file, list_directory,
                  write_file, patch_file, search_codebase, run_tests,
                  run_shell_command, index_repo]
research_tools = [web_search, scrape_page, search_algorithms]
db_tools       = [save_code_snippet, get_snippets, log_improvement]


# ── Build graph ──────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("Supervisor",    supervisor_node)
workflow.add_node("CodexEngine",   partial(worker_node, agent_name="codex",
                                           system_prompt=CODEX_SYSTEM_PROMPT,
                                           max_steps=CODEX_MAX_STEPS,
                                           use_memory=True))
workflow.add_node("ResearchAgent", partial(worker_node, agent_name="research",
                                           system_prompt=RESEARCH_SYSTEM_PROMPT,
                                           max_steps=DEFAULT_MAX_STEPS,
                                           use_memory=False))
workflow.add_node("DBAgent",       partial(worker_node, agent_name="db",
                                           system_prompt=DB_SYSTEM_PROMPT,
                                           max_steps=DEFAULT_MAX_STEPS,
                                           use_memory=False))

# All workers report back to the Commenter Supervisor
workflow.add_edge("CodexEngine",   "Supervisor")
workflow.add_edge("ResearchAgent", "Supervisor")
workflow.add_edge("DBAgent",       "Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "CodexEngine":   "CodexEngine",
        "ResearchAgent": "ResearchAgent",
        "DBAgent":       "DBAgent",
        "FINISH":        END,
    },
)
workflow.set_entry_point("Supervisor")

# Export uncompiled — main.py attaches the MemorySaver checkpointer
graph = workflow

__all__ = ["graph", "AgentState"]
