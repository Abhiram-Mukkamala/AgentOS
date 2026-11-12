"""
edge_router.py — Model-Agnostic Tier-0 Edge Reflex Router for AgentOS.

Sits in front of the heavy LangGraph multi-agent kernel (`agent.py`) to:
  1. Intercept deterministic, single-hop tool requests at sub-millisecond to
     few-millisecond latency without waking the full Supervisor -> Worker ->
     Supervisor LLM loop.
  2. Enforce strict uncertainty gating via joint confidence margin
     (`confidence_margin = top1 - top2`) and token-level slot uncertainty
     (`span_entropy`).
  3. Validate and coerce extracted slot arguments against the target tool's
     Pydantic schema before execution, escalating cleanly on any mismatch.
  4. Support pluggable backends (`rules`, `embedding`, `onnx`, `microllm`, `none`)
     via a clean `EdgeRouterBackend` protocol.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import httpx
import numpy as np
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

import tools as agentos_tools

log = logging.getLogger("agentos.edge_router")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Standardized Routing Contract
# ═══════════════════════════════════════════════════════════════════════════════

ESCALATION_LOW_MARGIN = "LOW_MARGIN"
ESCALATION_HIGH_SPAN_ENTROPY = "HIGH_SPAN_ENTROPY"
ESCALATION_MULTI_HOP_REQUIRED = "MULTI_HOP_REQUIRED"
ESCALATION_SCHEMA_COERCION_FAILED = "SCHEMA_COERCION_FAILED"
ESCALATION_ROUTER_DISABLED = "ROUTER_DISABLED"
ESCALATION_EXECUTION_ERROR = "EDGE_EXECUTION_ERROR"


@dataclass
class EdgeRouteDecision:
    handled_at_edge: bool
    target_agent_or_tool: Optional[str]
    extracted_args: Dict[str, Any]
    confidence_margin: float          # Joint score margin (top1 - top2)
    span_entropy: float               # Token-level uncertainty over extracted slots
    escalate_to_kernel: bool
    escalation_reason: Optional[str]  # e.g., "LOW_MARGIN", "HIGH_SPAN_ENTROPY", "MULTI_HOP_REQUIRED", "SCHEMA_COERCION_FAILED"
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BackendPrediction:
    """Raw prediction emitted by any pluggable EdgeRouterBackend."""
    target_agent_or_tool: Optional[str] = None
    extracted_args: Dict[str, Any] = field(default_factory=dict)
    confidence_margin: float = 0.0
    span_entropy: float = 0.0
    multi_hop_detected: bool = False
    raw_scores: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class EdgeRouterBackend(Protocol):
    """Protocol implemented by all Tier-0 Edge Router backends."""

    name: str

    def predict(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BackendPrediction:
        """Analyze prompt and return raw routing prediction + uncertainty metrics."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Math & Schema Validation Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def compute_shannon_entropy(probs: List[float]) -> float:
    """
    Compute normalized Shannon entropy in [0.0, 1.0] for a discrete probability
    distribution. Higher values indicate greater uncertainty.
    """
    clean = [p for p in probs if p > 1e-12]
    if len(clean) <= 1:
        return 0.0
    total = sum(clean)
    normed = [p / total for p in clean]
    raw_h = -sum(p * math.log(p) for p in normed)
    max_h = math.log(len(normed))
    return round(min(1.0, max(0.0, raw_h / max_h if max_h > 0 else 0.0)), 4)


def compute_margin_from_scores(scores: Dict[str, float]) -> Tuple[Optional[str], float]:
    """Return `(top1_key, top1_score - top2_score)` from a score dictionary."""
    if not scores:
        return None, 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top1_key, top1_val = ranked[0]
    if top1_val <= 0.0:
        return None, 0.0
    top2_val = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = round(max(0.0, float(top1_val - top2_val)), 4)
    return top1_key, margin


# Map of safe single-hop edge tools -> LangChain StructuredTool instances
EDGE_SAFE_TOOLS: Dict[str, Any] = {
    "read_file": agentos_tools.read_file,
    "list_directory": agentos_tools.list_directory,
    "search_codebase": agentos_tools.search_codebase,
    "run_tests": agentos_tools.run_tests,
    "execute_code": agentos_tools.execute_code,
    "analyze_code": agentos_tools.analyze_code,
    "get_snippets": agentos_tools.get_snippets,
    "web_search": agentos_tools.web_search,
    "search_algorithms": agentos_tools.search_algorithms,
}


class _FallbackToolSchemas:
    """Explicit Pydantic schemas used when a tool does not expose `args_schema`."""

    class ReadFileArgs(BaseModel):
        filepath: str

    class ListDirectoryArgs(BaseModel):
        directory_path: str = "."

    class SearchCodebaseArgs(BaseModel):
        query: str
        directory_path: str = "."

    class RunTestsArgs(BaseModel):
        test_path: str = "."

    class ExecuteCodeArgs(BaseModel):
        code: str

    class AnalyzeCodeArgs(BaseModel):
        code: str

    class GetSnippetsArgs(BaseModel):
        language: str = ""
        tag: str = ""

    class WebSearchArgs(BaseModel):
        query: str

    class SearchAlgorithmsArgs(BaseModel):
        topic: str

    MAP: Dict[str, type[BaseModel]] = {
        "read_file": ReadFileArgs,
        "list_directory": ListDirectoryArgs,
        "search_codebase": SearchCodebaseArgs,
        "run_tests": RunTestsArgs,
        "execute_code": ExecuteCodeArgs,
        "analyze_code": AnalyzeCodeArgs,
        "get_snippets": GetSnippetsArgs,
        "web_search": WebSearchArgs,
        "search_algorithms": SearchAlgorithmsArgs,
    }


def validate_and_coerce_tool_args(
    tool_name: str,
    extracted_args: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validate and coerce `extracted_args` against the target tool's Pydantic schema.

    Returns:
        `(is_valid, coerced_args_dict, error_message)`
    """
    if not isinstance(extracted_args, dict):
        return False, {}, "Extracted arguments must be a dictionary."

    schema_cls = _FallbackToolSchemas.MAP.get(tool_name)
    tool_obj = EDGE_SAFE_TOOLS.get(tool_name)
    if schema_cls is None and tool_obj is not None:
        schema_cls = getattr(tool_obj, "args_schema", None)

    if schema_cls is None:
        return False, {}, f"Unknown or non-edge tool: {tool_name}"

    try:
        validated = schema_cls(**extracted_args)
        coerced = validated.model_dump()
        # Reject empty required string fields (e.g., whitespace-only filepath/query)
        for field_name, field_info in schema_cls.model_fields.items():
            if field_info.is_required():
                val = coerced.get(field_name)
                if isinstance(val, str) and not val.strip():
                    return False, {}, f"Required field '{field_name}' cannot be empty."
        return True, coerced, None
    except (ValidationError, TypeError, ValueError) as exc:
        return False, {}, str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-Hop & Slot Extraction Heuristics
# ═══════════════════════════════════════════════════════════════════════════════

_MULTI_HOP_CONJUNCTIONS = re.compile(
    r"\b(?:"
    r"and\s+then|then\s+(?:write|edit|patch|fix|save|run|create|delete|update|refactor)|"
    r"after\s+that|followed\s+by|first\s+.*\s+then|"
    r"read\s+.*\s+and\s+(?:fix|refactor|update|modify|edit|patch|save|explain\s+how\s+to\s+change)|"
    r"search\s+.*\s+and\s+(?:save|implement|write|create|log)|"
    r"run\s+tests\s+and\s+fix"
    r")\b",
    re.IGNORECASE,
)

_COMPLEX_REASONING_PATTERNS = re.compile(
    r"\b(?:"
    r"architect|design\s+a\s+system|refactor\s+the\s+entire|build\s+a\s+full|"
    r"implement\s+a\s+new\s+feature|debug\s+and\s+fix|why\s+is\s+.*\s+failing\s+and\s+fix"
    r")\b",
    re.IGNORECASE,
)


def detect_multi_hop(prompt: str) -> bool:
    """Return True if the prompt requires multi-hop orchestration or synthesis."""
    text = prompt.strip()
    if not text:
        return False
    if _MULTI_HOP_CONJUNCTIONS.search(text):
        return True
    if _COMPLEX_REASONING_PATTERNS.search(text):
        return True
    # Multiple distinct imperative sentences (do not split on file extensions like .py)
    sentences = [s.strip() for s in re.split(r"(?:[;!?]|\.(?=\s|$)|\n)+", text) if s.strip()]
    if len(sentences) >= 3:
        return True
    return False


_FILE_TOKEN_RE = re.compile(
    r"(?:['\"`])([^'\"`\s]+\.[a-zA-Z0-9]{1,8})(?:['\"`])"
    r"|\b([a-zA-Z0-9_./\\-]+\.(?:py|js|ts|tsx|jsx|json|md|txt|ya?ml|toml|html|css|sql|sh|cfg|ini))\b"
)

_DIR_TOKEN_RE = re.compile(
    r"(?:in|of|inside|dir|directory|folder)\s+(?:['\"`]?([a-zA-Z0-9_./\\-]+)['\"`]?)",
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n([\s\S]+?)```", re.IGNORECASE)


def extract_slots_and_entropy(tool_name: str, prompt: str) -> Tuple[Dict[str, Any], float]:
    """
    Extract arguments for `tool_name` from `prompt` along with a token-level
    slot uncertainty (`span_entropy` in [0.0, 1.0]).
    """
    text = prompt.strip()

    if tool_name == "read_file":
        matches = _FILE_TOKEN_RE.findall(text)
        candidates = [m[0] or m[1] for m in matches if (m[0] or m[1])]
        if not candidates:
            # Check for trailing bare token after "read file" / "cat" / "show file"
            m = re.search(r"\b(?:read|cat|open|view|show)\s+(?:file\s+)?(\S+)$", text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip("'\"`")
                # Higher entropy if no file extension
                entropy = 0.25 if "." in candidate else 0.78
                return {"filepath": candidate}, entropy
            return {}, 1.0
        if len(candidates) == 1:
            return {"filepath": candidates[0]}, 0.05
        # Multiple file candidates -> ambiguous slot span!
        probs = [1.0 / len(candidates)] * len(candidates)
        return {"filepath": candidates[0]}, max(0.72, compute_shannon_entropy(probs))

    if tool_name == "list_directory":
        m = _DIR_TOKEN_RE.search(text)
        if m:
            candidate = m.group(1).strip("'\"`")
            if candidate.lower() in {"current", "here", "workspace", "root", "project"}:
                return {"directory_path": "."}, 0.05
            return {"directory_path": candidate}, 0.12
        return {"directory_path": "."}, 0.08

    if tool_name == "run_tests":
        matches = _FILE_TOKEN_RE.findall(text)
        candidates = [m[0] or m[1] for m in matches if (m[0] or m[1])]
        if len(candidates) == 1:
            return {"test_path": candidates[0]}, 0.08
        if len(candidates) > 1:
            return {"test_path": candidates[0]}, 0.75
        return {"test_path": "."}, 0.10

    if tool_name == "search_codebase":
        quoted = re.findall(r"['\"`]([^'\"`]{2,80})['\"`]", text)
        dir_match = _DIR_TOKEN_RE.search(text)
        dir_path = dir_match.group(1).strip("'\"`") if dir_match else "."
        if len(quoted) == 1:
            return {"query": quoted[0], "directory_path": dir_path}, 0.08
        if len(quoted) > 1:
            return {"query": quoted[0], "directory_path": dir_path}, 0.72
        m = re.search(
            r"\b(?:grep|search\s+code(?:base)?|find\s+in\s+code(?:base)?)\s+(?:for\s+)?([a-zA-Z0-9_.-]+)",
            text,
            re.IGNORECASE,
        )
        if m:
            return {"query": m.group(1), "directory_path": dir_path}, 0.22
        return {}, 0.90

    if tool_name in {"execute_code", "analyze_code"}:
        fence = _CODE_FENCE_RE.search(text)
        if fence:
            return {"code": fence.group(1).strip()}, 0.05
        inline = re.search(
            r"\b(?:execute|run|eval|analyze)\s+(?:python\s+)?(?:code\s*)?`([^`]+)`",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if inline:
            return {"code": inline.group(1).strip()}, 0.12
        return {}, 0.92

    if tool_name == "get_snippets":
        lang_match = re.search(
            r"\b(python|javascript|typescript|rust|go|sql|java|cpp|bash|html|css)\b",
            text,
            re.IGNORECASE,
        )
        tag_match = re.search(r"\btag(?:ged)?\s+['\"`]?([a-zA-Z0-9_-]+)['\"`]?", text, re.IGNORECASE)
        return {
            "language": lang_match.group(1).lower() if lang_match else "",
            "tag": tag_match.group(1).lower() if tag_match else "",
        }, 0.10

    if tool_name == "web_search":
        m = re.sub(
            r"^(?:please\s+)?(?:web\s*search|search\s+the\s+web|look\s+up\s+online|google)\s*(?:for|about)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" ?.'\"`")
        if len(m) >= 3:
            return {"query": m}, 0.15
        return {}, 0.85

    if tool_name == "search_algorithms":
        m = re.sub(
            r"^(?:please\s+)?(?:search\s+algorithm(?:s)?|find\s+algorithm(?:s)?|algorithm\s+search)\s*(?:for|on|about)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" ?.'\"`")
        if len(m) >= 2:
            return {"topic": m}, 0.15
        return {}, 0.85

    return {}, 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Concrete Pluggable Backends
# ═══════════════════════════════════════════════════════════════════════════════


class RulesEdgeBackend:
    """
    Default zero-dependency deterministic reflex backend.
    Uses weighted pattern scoring + slot ambiguity entropy calculation.
    """

    name = "rules"

    _INTENT_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
        "read_file": [
            (re.compile(r"^(?:please\s+)?(?:read|cat|view|open|show)\s+(?!all\s+files|files\s+in|directory|folder|workspace|my\s+|saved\s+|snippets?\b)(?:the\s+)?(?:contents?\s+of\s+)?(?:file\s+)?\S+", re.I), 0.92),
            (re.compile(r"\bread_file\b", re.I), 0.95),
        ],
        "list_directory": [
            (re.compile(r"^(?:please\s+)?(?:list|ls|dir|show)\s+(?:all\s+)?(?:the\s+)?(?:files|directory|folder|workspace)\b", re.I), 0.90),
            (re.compile(r"\blist_directory\b", re.I), 0.95),
        ],
        "search_codebase": [
            (re.compile(r"^(?:please\s+)?(?:grep|search\s+code(?:base)?|find\s+in\s+code(?:base)?)\b", re.I), 0.90),
            (re.compile(r"\bsearch_codebase\b", re.I), 0.95),
        ],
        "run_tests": [
            (re.compile(r"^(?:please\s+)?(?:run|execute)\s+(?:the\s+)?(?:unit\s+)?(?:pytest|tests?)\b", re.I), 0.92),
            (re.compile(r"\brun_tests\b", re.I), 0.95),
        ],
        "execute_code": [
            (re.compile(r"^(?:please\s+)?(?:run|execute|eval)\s+(?:this\s+)?(?:python\s+)?code\b", re.I), 0.90),
            (re.compile(r"\bexecute_code\b", re.I), 0.95),
        ],
        "analyze_code": [
            (re.compile(r"^(?:please\s+)?(?:analyze|lint|ast\s+check)\s+(?:this\s+)?(?:python\s+)?code\b", re.I), 0.88),
            (re.compile(r"\banalyze_code\b", re.I), 0.95),
        ],
        "get_snippets": [
            (re.compile(r"^(?:please\s+)?(?:list|show|get|fetch)\s+(?:(?:my|saved|all|python|javascript|typescript|rust|go|sql|java|cpp|bash)\s+)*snippets?\b", re.I), 0.90),
            (re.compile(r"\bget_snippets\b", re.I), 0.95),
        ],
        "web_search": [
            (re.compile(r"^(?:please\s+)?(?:web\s*search|search\s+the\s+web|look\s+up\s+online)\b", re.I), 0.88),
            (re.compile(r"\bweb_search\b", re.I), 0.95),
        ],
        "search_algorithms": [
            (re.compile(r"^(?:please\s+)?(?:search\s+algorithms?|find\s+algorithms?|algorithm\s+search)\b", re.I), 0.88),
            (re.compile(r"\bsearch_algorithms\b", re.I), 0.95),
        ],
    }

    def predict(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BackendPrediction:
        text = (prompt or "").strip()
        if not text:
            return BackendPrediction()

        if detect_multi_hop(text):
            return BackendPrediction(multi_hop_detected=True)

        scores: Dict[str, float] = {}
        for intent, patterns in self._INTENT_PATTERNS.items():
            best = 0.0
            for regex, weight in patterns:
                if regex.search(text):
                    best = max(best, weight)
            if best > 0.0:
                scores[intent] = best

        target, margin = compute_margin_from_scores(scores)
        if not target:
            return BackendPrediction(raw_scores=scores)

        extracted_args, span_entropy = extract_slots_and_entropy(target, text)
        return BackendPrediction(
            target_agent_or_tool=target,
            extracted_args=extracted_args,
            confidence_margin=margin,
            span_entropy=span_entropy,
            multi_hop_detected=False,
            raw_scores=scores,
        )


class EmbeddingEdgeBackend:
    """
    Lightweight embedding similarity router.
    Uses `fastembed` when installed, or falls back to fast L2-normalized character
    3-gram + word hashing vectors in `numpy` so it works offline without extra downloads.
    """

    name = "embedding"

    _EXEMPLARS: Dict[str, List[str]] = {
        "read_file": [
            "read file main.py",
            "show the contents of database.py",
            "open and view file config.json",
        ],
        "list_directory": [
            "list files in directory",
            "show all files in workspace folder",
            "ls current directory",
        ],
        "search_codebase": [
            "search codebase for wrap_robust_structured_output",
            "grep for def supervisor_node in code",
            "find in codebase EdgeRouteDecision",
        ],
        "run_tests": [
            "run unit tests",
            "execute pytest test suite",
            "run_tests on test_auth_settings.py",
        ],
        "execute_code": [
            "execute this python code",
            "run python code snippet",
        ],
        "analyze_code": [
            "analyze this python code for syntax and ast issues",
            "lint and analyze code",
        ],
        "get_snippets": [
            "list saved code snippets",
            "get my python snippets",
            "show all saved snippets",
        ],
        "web_search": [
            "search the web for latest fastapi release notes",
            "web search python asyncio tutorial",
        ],
        "search_algorithms": [
            "search algorithms for dijkstra shortest path",
            "find algorithm complexity and implementation",
        ],
    }

    def __init__(self, dim: int = 256):
        self._dim = dim
        self._fastembed_model = None
        if os.getenv("EDGE_ROUTER_USE_FASTEMBED", "false").lower() == "true":
            try:
                from fastembed import TextEmbedding  # type: ignore
                self._fastembed_model = TextEmbedding()
            except Exception:
                self._fastembed_model = None
        self._centroids: Dict[str, np.ndarray] = self._build_centroids()

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self._fastembed_model is not None:
            vecs = list(self._fastembed_model.embed(texts))
            arr = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            return arr / norms

        # Deterministic hashing trick over word unigrams + char 3-grams
        mat = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row_idx, t in enumerate(texts):
            clean = re.sub(r"[^a-z0-9_ ]+", " ", t.lower())
            words = clean.split()
            for w in words:
                idx = hash(("w", w)) % self._dim
                mat[row_idx, idx] += 2.0
                for i in range(max(0, len(w) - 2)):
                    trigram = w[i : i + 3]
                    t_idx = hash(("c3", trigram)) % self._dim
                    mat[row_idx, t_idx] += 0.5
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms

    def _build_centroids(self) -> Dict[str, np.ndarray]:
        centroids: Dict[str, np.ndarray] = {}
        for intent, samples in self._EXEMPLARS.items():
            emb = self._embed_texts(samples)
            mean_vec = emb.mean(axis=0)
            norm = np.linalg.norm(mean_vec) + 1e-12
            centroids[intent] = mean_vec / norm
        return centroids

    def predict(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BackendPrediction:
        text = (prompt or "").strip()
        if not text:
            return BackendPrediction()
        if detect_multi_hop(text):
            return BackendPrediction(multi_hop_detected=True)

        q_vec = self._embed_texts([text])[0]
        scores: Dict[str, float] = {}
        for intent, c_vec in self._centroids.items():
            sim = float(np.dot(q_vec, c_vec))
            scores[intent] = max(0.0, sim)

        target, margin = compute_margin_from_scores(scores)
        if not target:
            return BackendPrediction(raw_scores=scores)

        extracted_args, span_entropy = extract_slots_and_entropy(target, text)
        return BackendPrediction(
            target_agent_or_tool=target,
            extracted_args=extracted_args,
            confidence_margin=margin,
            span_entropy=span_entropy,
            multi_hop_detected=False,
            raw_scores=scores,
        )


class OnnxTokenTaggerBackend:
    """
    ONNX Runtime joint intent classifier & token-slot tagger backend (e.g., ARN-v2).

    Expects an ONNX session producing `(intent_logits, slot_logits)` or accepts a
    custom callable/session for unit testing. When no ONNX model file is configured,
    delegates to `RulesEdgeBackend` while preserving the ONNX tensor scoring math.
    """

    name = "onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        intent_labels: Optional[List[str]] = None,
        session_runner: Optional[Callable[[str], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]] = None,
    ):
        self.model_path = model_path or os.getenv("EDGE_ROUTER_ONNX_MODEL_PATH", "").strip()
        self.intent_labels = intent_labels or list(EDGE_SAFE_TOOLS.keys())
        self._session = None
        self._session_runner = session_runner
        self._fallback = RulesEdgeBackend()

        if self.model_path and os.path.isfile(self.model_path):
            try:
                import onnxruntime as ort  # type: ignore
                self._session = ort.InferenceSession(
                    self.model_path,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                log.warning("Failed to initialize ONNX session (%s): %s", self.model_path, exc)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-12)

    def score_logits(
        self,
        intent_logits: np.ndarray,
        slot_logits: np.ndarray,
        extracted_args: Dict[str, Any],
    ) -> BackendPrediction:
        """Compute joint confidence margin and mean token-level Shannon span entropy from logits."""
        intent_probs = self._softmax(np.asarray(intent_logits, dtype=np.float64).reshape(-1))
        scores = {
            self.intent_labels[i]: float(intent_probs[i])
            for i in range(min(len(self.intent_labels), len(intent_probs)))
        }
        target, margin = compute_margin_from_scores(scores)

        slot_arr = np.asarray(slot_logits, dtype=np.float64)
        if slot_arr.ndim == 2 and slot_arr.shape[0] > 0:
            token_probs = self._softmax(slot_arr)
            token_entropies = [
                compute_shannon_entropy(row.tolist()) for row in token_probs
            ]
            span_entropy = round(float(np.mean(token_entropies)), 4)
        else:
            span_entropy = 0.0

        return BackendPrediction(
            target_agent_or_tool=target,
            extracted_args=extracted_args,
            confidence_margin=margin,
            span_entropy=span_entropy,
            multi_hop_detected=False,
            raw_scores=scores,
        )

    def predict(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BackendPrediction:
        text = (prompt or "").strip()
        if not text:
            return BackendPrediction()
        if detect_multi_hop(text):
            return BackendPrediction(multi_hop_detected=True)

        if self._session_runner is not None:
            intent_logits, slot_logits, extracted_args = self._session_runner(text)
            return self.score_logits(intent_logits, slot_logits, extracted_args)

        return self._fallback.predict(text, context=context)


class MicroLLMEdgeBackend:
    """
    Ultra-fast local micro-LLM classifier backend using an OpenAI-compatible
    endpoint (or injectable callable for testing).
    """

    name = "microllm"

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout_s: float = 0.8,
        responder: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self.endpoint_url = endpoint_url or os.getenv(
            "EDGE_ROUTER_MICROLLM_URL",
            "http://127.0.0.1:8090/v1/chat/completions",
        )
        self.timeout_s = timeout_s
        self._responder = responder

    def predict(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BackendPrediction:
        text = (prompt or "").strip()
        if not text:
            return BackendPrediction()
        if detect_multi_hop(text):
            return BackendPrediction(multi_hop_detected=True)

        try:
            if self._responder is not None:
                data = self._responder(text)
            else:
                resp = httpx.post(
                    self.endpoint_url,
                    json={
                        "model": "local",
                        "temperature": 0.0,
                        "max_tokens": 128,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Respond ONLY with JSON: "
                                    '{"target": "<tool_or_null>", "args": {}, '
                                    '"margin": 0.0, "entropy": 0.0, "multi_hop": false}'
                                ),
                            },
                            {"role": "user", "content": text},
                        ],
                    },
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                import json
                content = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(content)

            return BackendPrediction(
                target_agent_or_tool=data.get("target"),
                extracted_args=data.get("args") or {},
                confidence_margin=float(data.get("margin", 0.0)),
                span_entropy=float(data.get("entropy", 1.0)),
                multi_hop_detected=bool(data.get("multi_hop", False)),
            )
        except Exception as exc:
            log.debug("MicroLLMEdgeBackend fallback escalation: %s", exc)
            return BackendPrediction(confidence_margin=0.0, span_entropy=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Tier-0 Edge Router Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class EdgeRouter:
    """
    Model-Agnostic Tier-0 Edge Reflex Router.

    Evaluates incoming user prompts via a pluggable `EdgeRouterBackend`, applies
    confidence margin and token-level span-entropy gates, validates/coerces tool
    arguments against Pydantic schemas, and either executes safe single-hop tools
    directly at the edge or escalates to the LangGraph kernel with structured
    diagnostics.
    """

    def __init__(
        self,
        backend: Optional[EdgeRouterBackend | str] = None,
        min_margin: Optional[float] = None,
        max_span_entropy: Optional[float] = None,
    ):
        self.min_margin = (
            float(min_margin)
            if min_margin is not None
            else float(os.getenv("EDGE_ROUTER_MIN_MARGIN", "0.25"))
        )
        self.max_span_entropy = (
            float(max_span_entropy)
            if max_span_entropy is not None
            else float(os.getenv("EDGE_ROUTER_MAX_SPAN_ENTROPY", "0.65"))
        )
        self._disabled = False
        self.backend = self._resolve_backend(backend)

        # Telemetry counters
        self.total_decisions: int = 0
        self.edge_handled_count: int = 0
        self.kernel_escalated_count: int = 0
        self.escalation_counts: Dict[str, int] = {}
        self._cumulative_latency_ms: float = 0.0

    def _resolve_backend(
        self,
        spec: Optional[EdgeRouterBackend | str],
    ) -> Optional[EdgeRouterBackend]:
        if spec is not None and not isinstance(spec, str):
            self._disabled = False
            return spec

        name = (spec or os.getenv("EDGE_ROUTER_BACKEND", "rules")).strip().lower()
        if name in {"none", "disabled", "off", "false", "0"}:
            self._disabled = True
            return None
        self._disabled = False
        if name == "embedding":
            return EmbeddingEdgeBackend()
        if name == "onnx":
            return OnnxTokenTaggerBackend()
        if name == "microllm":
            return MicroLLMEdgeBackend()
        return RulesEdgeBackend()

    def _record_decision(self, decision: EdgeRouteDecision) -> EdgeRouteDecision:
        self.total_decisions += 1
        self._cumulative_latency_ms += decision.latency_ms
        if decision.handled_at_edge:
            self.edge_handled_count += 1
        else:
            self.kernel_escalated_count += 1
            reason = decision.escalation_reason or "UNKNOWN"
            self.escalation_counts[reason] = self.escalation_counts.get(reason, 0) + 1
        return decision

    def get_telemetry(self) -> Dict[str, Any]:
        avg_lat = (
            round(self._cumulative_latency_ms / self.total_decisions, 3)
            if self.total_decisions > 0
            else 0.0
        )
        return {
            "backend": "none" if self._disabled or self.backend is None else getattr(self.backend, "name", "custom"),
            "total_decisions": self.total_decisions,
            "edge_handled_count": self.edge_handled_count,
            "kernel_escalated_count": self.kernel_escalated_count,
            "escalation_counts": dict(self.escalation_counts),
            "avg_latency_ms": avg_lat,
        }

    def route(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EdgeRouteDecision:
        """
        Evaluate `prompt` and return a standardized `EdgeRouteDecision`.
        Never raises; any internal failure escalates cleanly to the kernel.
        """
        t0 = time.perf_counter()

        if self._disabled or self.backend is None:
            latency = round((time.perf_counter() - t0) * 1000.0, 3)
            return self._record_decision(
                EdgeRouteDecision(
                    handled_at_edge=False,
                    target_agent_or_tool=None,
                    extracted_args={},
                    confidence_margin=0.0,
                    span_entropy=0.0,
                    escalate_to_kernel=True,
                    escalation_reason=ESCALATION_ROUTER_DISABLED,
                    latency_ms=latency,
                )
            )

        pred = self.backend.predict(prompt, context=context)

        # Gate 1: Multi-hop / complex synthesis check
        if pred.multi_hop_detected:
            latency = round((time.perf_counter() - t0) * 1000.0, 3)
            return self._record_decision(
                EdgeRouteDecision(
                    handled_at_edge=False,
                    target_agent_or_tool=pred.target_agent_or_tool,
                    extracted_args=pred.extracted_args,
                    confidence_margin=round(float(pred.confidence_margin), 4),
                    span_entropy=round(float(pred.span_entropy), 4),
                    escalate_to_kernel=True,
                    escalation_reason=ESCALATION_MULTI_HOP_REQUIRED,
                    latency_ms=latency,
                )
            )

        # Gate 2: Confidence margin threshold (top1 - top2)
        if not pred.target_agent_or_tool or pred.confidence_margin < self.min_margin:
            latency = round((time.perf_counter() - t0) * 1000.0, 3)
            return self._record_decision(
                EdgeRouteDecision(
                    handled_at_edge=False,
                    target_agent_or_tool=pred.target_agent_or_tool,
                    extracted_args=pred.extracted_args,
                    confidence_margin=round(float(pred.confidence_margin), 4),
                    span_entropy=round(float(pred.span_entropy), 4),
                    escalate_to_kernel=True,
                    escalation_reason=ESCALATION_LOW_MARGIN,
                    latency_ms=latency,
                )
            )

        # Gate 3: Token-level span entropy threshold over extracted slots
        if pred.span_entropy > self.max_span_entropy:
            latency = round((time.perf_counter() - t0) * 1000.0, 3)
            return self._record_decision(
                EdgeRouteDecision(
                    handled_at_edge=False,
                    target_agent_or_tool=pred.target_agent_or_tool,
                    extracted_args=pred.extracted_args,
                    confidence_margin=round(float(pred.confidence_margin), 4),
                    span_entropy=round(float(pred.span_entropy), 4),
                    escalate_to_kernel=True,
                    escalation_reason=ESCALATION_HIGH_SPAN_ENTROPY,
                    latency_ms=latency,
                )
            )

        # Gate 4: Pydantic schema coercion & validation for the target tool
        valid, coerced_args, _err = validate_and_coerce_tool_args(
            pred.target_agent_or_tool,
            pred.extracted_args,
        )
        if not valid:
            latency = round((time.perf_counter() - t0) * 1000.0, 3)
            return self._record_decision(
                EdgeRouteDecision(
                    handled_at_edge=False,
                    target_agent_or_tool=pred.target_agent_or_tool,
                    extracted_args=pred.extracted_args,
                    confidence_margin=round(float(pred.confidence_margin), 4),
                    span_entropy=round(float(pred.span_entropy), 4),
                    escalate_to_kernel=True,
                    escalation_reason=ESCALATION_SCHEMA_COERCION_FAILED,
                    latency_ms=latency,
                )
            )

        latency = round((time.perf_counter() - t0) * 1000.0, 3)
        return self._record_decision(
            EdgeRouteDecision(
                handled_at_edge=True,
                target_agent_or_tool=pred.target_agent_or_tool,
                extracted_args=coerced_args,
                confidence_margin=round(float(pred.confidence_margin), 4),
                span_entropy=round(float(pred.span_entropy), 4),
                escalate_to_kernel=False,
                escalation_reason=None,
                latency_ms=latency,
            )
        )

    async def route_and_execute(
        self,
        prompt: str,
        user_id: int = 1,
        thread_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[EdgeRouteDecision, Optional[str]]:
        """
        Route `prompt` at Tier-0 and, if `handled_at_edge` is True, directly
        invoke the target tool with user/thread `RunnableConfig`.

        Returns:
            `(decision, response_text_or_none)`
        """
        decision = self.route(prompt, context=context)
        if not decision.handled_at_edge or not decision.target_agent_or_tool:
            return decision, None

        tool_obj = EDGE_SAFE_TOOLS.get(decision.target_agent_or_tool)
        if tool_obj is None:
            decision.handled_at_edge = False
            decision.escalate_to_kernel = True
            decision.escalation_reason = ESCALATION_SCHEMA_COERCION_FAILED
            return decision, None

        config = RunnableConfig(
            configurable={"user_id": user_id, "thread_id": thread_id}
        )
        invoke_args = dict(decision.extracted_args)
        if decision.target_agent_or_tool == "read_file" and "filepath" in invoke_args and "path" not in invoke_args:
            invoke_args["path"] = invoke_args.pop("filepath")
        elif decision.target_agent_or_tool == "list_directory" and "directory_path" in invoke_args and "path" not in invoke_args:
            invoke_args["path"] = invoke_args.pop("directory_path")
        elif decision.target_agent_or_tool == "search_codebase":
            if "query" in invoke_args and "pattern" not in invoke_args:
                invoke_args["pattern"] = invoke_args.pop("query")
            if "directory_path" in invoke_args and "path" not in invoke_args:
                invoke_args["path"] = invoke_args.pop("directory_path")

        try:
            if hasattr(tool_obj, "ainvoke"):
                raw_result = await tool_obj.ainvoke(invoke_args, config=config)
            elif hasattr(tool_obj, "invoke"):
                raw_result = await asyncio.to_thread(
                    tool_obj.invoke, invoke_args, config
                )
            else:
                raw_result = await asyncio.to_thread(tool_obj, **invoke_args)
            return decision, str(raw_result)
        except Exception as exc:
            log.warning(
                "Edge execution failed for %s (%s); escalating to kernel.",
                decision.target_agent_or_tool,
                exc,
            )
            decision.handled_at_edge = False
            decision.escalate_to_kernel = True
            decision.escalation_reason = ESCALATION_EXECUTION_ERROR
            self.edge_handled_count = max(0, self.edge_handled_count - 1)
            self.kernel_escalated_count += 1
            self.escalation_counts[ESCALATION_EXECUTION_ERROR] = (
                self.escalation_counts.get(ESCALATION_EXECUTION_ERROR, 0) + 1
            )
            return decision, None


# Singleton router instance used by the FastAPI gateway
edge_router = EdgeRouter()
