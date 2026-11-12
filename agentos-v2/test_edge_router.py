"""
test_edge_router.py — Unit and integration tests for Phase 3 (`edge_router.py`).
"""

import asyncio
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from edge_router import (
    ESCALATION_HIGH_SPAN_ENTROPY,
    ESCALATION_LOW_MARGIN,
    ESCALATION_MULTI_HOP_REQUIRED,
    ESCALATION_ROUTER_DISABLED,
    ESCALATION_SCHEMA_COERCION_FAILED,
    BackendPrediction,
    EdgeRouteDecision,
    EdgeRouter,
    EmbeddingEdgeBackend,
    MicroLLMEdgeBackend,
    OnnxTokenTaggerBackend,
)


def test_edge_route_decision_contract():
    """Verify exact dataclass field names and types on EdgeRouteDecision."""
    field_names = [f.name for f in fields(EdgeRouteDecision)]
    assert field_names == [
        "handled_at_edge",
        "target_agent_or_tool",
        "extracted_args",
        "confidence_margin",
        "span_entropy",
        "escalate_to_kernel",
        "escalation_reason",
        "latency_ms",
    ]


def test_rules_backend_single_hop_and_escalations():
    router = EdgeRouter(backend="rules", min_margin=0.25, max_span_entropy=0.65)

    # 1. Clear single-hop read_file request -> handled at edge
    d1 = router.route("read file database.py")
    assert d1.handled_at_edge is True
    assert d1.escalate_to_kernel is False
    assert d1.target_agent_or_tool == "read_file"
    assert d1.extracted_args == {"filepath": "database.py"}
    assert d1.confidence_margin >= 0.25
    assert d1.span_entropy <= 0.65
    assert d1.escalation_reason is None
    assert d1.latency_ms >= 0.0

    # 2. Multi-hop compound request -> MULTI_HOP_REQUIRED
    d2 = router.route("read file agent.py and then refactor the supervisor node")
    assert d2.handled_at_edge is False
    assert d2.escalate_to_kernel is True
    assert d2.escalation_reason == ESCALATION_MULTI_HOP_REQUIRED

    # 3. Ambiguous / open-ended conversational prompt -> LOW_MARGIN
    d3 = router.route("What is the best way to structure a microservices event bus?")
    assert d3.handled_at_edge is False
    assert d3.escalate_to_kernel is True
    assert d3.escalation_reason == ESCALATION_LOW_MARGIN

    # 4. High slot ambiguity (multiple conflicting filenames) -> HIGH_SPAN_ENTROPY
    d4 = router.route("read file alpha.py beta.py gamma.py")
    assert d4.handled_at_edge is False
    assert d4.escalate_to_kernel is True
    assert d4.escalation_reason == ESCALATION_HIGH_SPAN_ENTROPY
    assert d4.span_entropy > 0.65


def test_schema_coercion_failure_escalation():
    """Verify invalid extracted tool args trigger SCHEMA_COERCION_FAILED."""

    class BadArgsBackend:
        name = "bad_args"

        def predict(self, prompt: str, context=None) -> BackendPrediction:
            return BackendPrediction(
                target_agent_or_tool="read_file",
                extracted_args={"filepath": "   "},  # empty required string
                confidence_margin=0.90,
                span_entropy=0.05,
                multi_hop_detected=False,
            )

    router = EdgeRouter(backend=BadArgsBackend())
    decision = router.route("read file")
    assert decision.handled_at_edge is False
    assert decision.escalate_to_kernel is True
    assert decision.escalation_reason == ESCALATION_SCHEMA_COERCION_FAILED


def test_pluggable_backends_onnx_embedding_microllm_disabled():
    # 1. ONNX Token Tagger backend with synthetic logits
    def fake_onnx_runner(text: str):
        # 9 intent classes; index 1 is list_directory with high logit
        intent_logits = np.array([-2.0, 5.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        # 3 tokens x 4 BIO tags with low entropy (peaked distributions)
        slot_logits = np.array([
            [6.0, -2.0, -2.0, -2.0],
            [-2.0, 6.0, -2.0, -2.0],
            [6.0, -2.0, -2.0, -2.0],
        ])
        return intent_logits, slot_logits, {"directory_path": "."}

    onnx_backend = OnnxTokenTaggerBackend(session_runner=fake_onnx_runner)
    onnx_router = EdgeRouter(backend=onnx_backend)
    onnx_dec = onnx_router.route("list files in workspace")
    assert onnx_dec.handled_at_edge is True
    assert onnx_dec.target_agent_or_tool == "list_directory"
    assert onnx_dec.confidence_margin > 0.80
    assert onnx_dec.span_entropy < 0.20

    # 2. Embedding backend
    emb_router = EdgeRouter(backend=EmbeddingEdgeBackend(), min_margin=0.05)
    emb_dec = emb_router.route("read file main.py")
    assert emb_dec.handled_at_edge is True
    assert emb_dec.target_agent_or_tool == "read_file"
    assert emb_dec.extracted_args == {"filepath": "main.py"}

    # 3. MicroLLM backend with injectable responder
    micro_backend = MicroLLMEdgeBackend(
        responder=lambda p: {
            "target": "get_snippets",
            "args": {"language": "python", "tag": ""},
            "margin": 0.82,
            "entropy": 0.11,
            "multi_hop": False,
        }
    )
    micro_router = EdgeRouter(backend=micro_backend)
    micro_dec = micro_router.route("show my python snippets")
    assert micro_dec.handled_at_edge is True
    assert micro_dec.target_agent_or_tool == "get_snippets"
    assert micro_dec.extracted_args == {"language": "python", "tag": ""}

    # 4. Disabled router ("none")
    off_router = EdgeRouter(backend="none")
    off_dec = off_router.route("read file main.py")
    assert off_dec.handled_at_edge is False
    assert off_dec.escalate_to_kernel is True
    assert off_dec.escalation_reason == ESCALATION_ROUTER_DISABLED


async def _run_async_edge_execution_test():
    router = EdgeRouter(backend="rules")
    decision, output = await router.route_and_execute(
        "execute python code `print(6 * 7)`",
        user_id=1,
        thread_id="edge-test-thread",
    )
    assert decision.handled_at_edge is True
    assert decision.target_agent_or_tool == "execute_code"
    assert output is not None and "42" in output

    telemetry = router.get_telemetry()
    assert telemetry["total_decisions"] == 1
    assert telemetry["edge_handled_count"] == 1


def test_edge_direct_tool_execution():
    asyncio.run(_run_async_edge_execution_test())


if __name__ == "__main__":
    test_edge_route_decision_contract()
    test_rules_backend_single_hop_and_escalations()
    test_schema_coercion_failure_escalation()
    test_pluggable_backends_onnx_embedding_microllm_disabled()
    test_edge_direct_tool_execution()
    print("ALL_EDGE_ROUTER_TESTS_PASSED")
