"""
codex_engine — AgentOS self-contained local inference engine.

Auto-detects hardware (RAM, VRAM, CPU, SSD), picks the optimal Qwen2.5-Coder
model, loads it via llama-cpp-python with mmap SSD overflow, and exposes an
OpenAI-compatible API with tool-calling support.

Usage:
    # Embedded (inside AgentOS)
    from codex_engine import start_codex_server
    start_codex_server(background=True)

    # Standalone (for testing)
    python -m codex_engine.server
"""

from codex_engine.engine import CodexEngine
from codex_engine.server import start_codex_server

__all__ = ["CodexEngine", "start_codex_server"]
