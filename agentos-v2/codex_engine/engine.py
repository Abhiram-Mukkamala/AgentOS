"""
engine.py — AgentOS Codex inference engine.

Wraps llama-cpp-python with auto-configuration from hardware detection.
Handles model downloading, loading, idle-unload, and OpenAI-compatible chat
completions with tool-calling support.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

from codex_engine.hardware import HardwareProfile, detect_hardware

log = logging.getLogger("codex_engine.engine")


# ═══════════════════════════════════════════════════════════════════════════════
# Model Downloader
# ═══════════════════════════════════════════════════════════════════════════════


def _download_model(repo_id: str, filename: str) -> str:
    """
    Download a GGUF model from HuggingFace Hub.

    Returns the local path to the downloaded file.
    Uses HF cache so re-downloads are skipped automatically.
    """
    try:
        from huggingface_hub import hf_hub_download

        log.info(
            "Downloading model: %s/%s  (this is a one-time download, "
            "may take several minutes depending on your connection...)",
            repo_id, filename,
        )

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
        )

        size_gb = Path(path).stat().st_size / (1024 ** 3)
        log.info("Model downloaded: %s (%.1f GB)", path, size_gb)
        return path

    except ImportError:
        raise RuntimeError(
            "huggingface-hub is not installed. "
            "Run: pip install huggingface-hub"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {repo_id}/{filename}: {e}\n"
            f"You can manually download from https://huggingface.co/{repo_id} "
            f"and set LOCAL_CODEX_MODEL_PATH in .env"
        )


def _find_local_model(profile: HardwareProfile, model_path: str = None) -> str:
    """
    Resolve the model path.

    Priority:
      1. Explicit model_path argument
      2. LOCAL_CODEX_MODEL_PATH env var
      3. .gguf files in the project directory
      4. Auto-download from HuggingFace
    """
    # 1. Explicit path
    if model_path and Path(model_path).is_file():
        log.info("Using explicit model path: %s", model_path)
        return model_path

    # 2. Environment variable
    env_path = os.getenv("LOCAL_CODEX_MODEL_PATH", "").strip()
    if env_path and Path(env_path).is_file():
        log.info("Using model from LOCAL_CODEX_MODEL_PATH: %s", env_path)
        return env_path

    # 3. Search project directory for .gguf files
    project_dir = Path(os.getcwd())
    gguf_files = sorted(project_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    if gguf_files:
        chosen = str(gguf_files[0])
        log.info("Found local model: %s", chosen)
        return chosen

    # 4. Auto-download from HuggingFace
    if not profile.recommended_repo or not profile.recommended_file:
        raise RuntimeError(
            "No local model available and hardware is too limited for auto-download. "
            "Configure cloud API keys in Settings or set LOCAL_CODEX=false in .env."
        )

    return _download_model(profile.recommended_repo, profile.recommended_file)


# ═══════════════════════════════════════════════════════════════════════════════
# Codex Engine
# ═══════════════════════════════════════════════════════════════════════════════


class CodexEngine:
    """
    Self-contained local LLM inference engine.

    Auto-detects hardware, downloads the optimal model, configures
    context/threads/layers, and provides OpenAI-compatible chat completions
    with tool-calling support.
    """

    def __init__(
        self,
        model_path: str = None,
        force_size: str = None,
        n_ctx: int = None,
        n_threads: int = None,
    ):
        self._start_time = time.time()
        self._last_used_time = time.time()
        self._idle_unload_seconds = int(os.getenv("LOCAL_CODEX_IDLE_UNLOAD_SECONDS", "1800"))
        self._total_tokens = 0
        self._total_requests = 0
        self._model = None

        # ── Detect hardware ────────────────────────────────────────────────
        self.profile = detect_hardware(force_size=force_size)

        if not self.profile.recommended_model and not model_path:
            raise RuntimeError(
                f"Not enough RAM for local inference "
                f"({self.profile.available_ram_gb:.1f} GB available). "
                f"Need at least 4 GB (or 3 GB + SSD)."
            )

        # ── Override auto-detected settings if specified ───────────────────
        if n_ctx:
            self.profile.n_ctx = n_ctx
        if n_threads:
            self.profile.n_threads = n_threads

        # ── Resolve model path ─────────────────────────────────────────────
        self._model_path = _find_local_model(self.profile, model_path)

        # ── Load model ─────────────────────────────────────────────────────
        self._load_model()

    def _load_model(self):
        """Load the GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "Install with:  pip install llama-cpp-python\n"
                "For NVIDIA GPU: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )

        log.info(
            "Loading model: %s\n"
            "  Context:  %d tokens\n"
            "  Threads:  %d\n"
            "  GPU:      %s\n"
            "  mmap:     %s (SSD overflow: %s)",
            Path(self._model_path).name,
            self.profile.n_ctx,
            self.profile.n_threads,
            f"{self.profile.n_gpu_layers} layers" if self.profile.n_gpu_layers != 0 else "disabled",
            "enabled" if self.profile.use_mmap else "disabled",
            "ACTIVE" if self.profile.ssd_overflow_active else "standby",
        )

        load_start = time.time()

        self._model = Llama(
            model_path=self._model_path,
            n_ctx=self.profile.n_ctx,
            n_gpu_layers=self.profile.n_gpu_layers,
            n_threads=self.profile.n_threads,
            use_mmap=self.profile.use_mmap,
            use_mlock=self.profile.use_mlock,
            chat_format="chatml-function-calling",
            verbose=False,
        )

        self._last_used_time = time.time()
        load_time = time.time() - load_start
        log.info("Model loaded in %.1f seconds.", load_time)

    # ── Chat Completion ─────────────────────────────────────────────────────

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> dict:
        """
        OpenAI-compatible chat completion.

        Auto-reloads the model if it was previously unloaded via idle timeout or /unload.
        """
        if self._model is None:
            log.info("Model was unloaded; reloading on demand...")
            self._load_model()

        self._last_used_time = time.time()
        self._total_requests += 1
        start = time.time()

        try:
            completion_kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if tools:
                completion_kwargs["tools"] = tools
                completion_kwargs["tool_choice"] = tool_choice

            response = self._model.create_chat_completion(**completion_kwargs)

            usage = response.get("usage", {})
            self._total_tokens += usage.get("total_tokens", 0)

            elapsed = time.time() - start
            gen_tokens = usage.get("completion_tokens", 0)
            tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0

            log.debug(
                "Completion: %d tokens in %.1fs (%.1f tok/s)",
                gen_tokens, elapsed, tok_per_sec,
            )

            self._last_used_time = time.time()
            return response

        except Exception as e:
            log.error("Chat completion failed: %s", repr(e))
            raise

    # ── Health ──────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Engine health and status report."""
        uptime = time.time() - self._start_time

        ram_info = {}
        try:
            import psutil
            mem = psutil.virtual_memory()
            process = psutil.Process()
            ram_info = {
                "system_ram_used_gb": round(mem.used / (1024 ** 3), 1),
                "system_ram_available_gb": round(mem.available / (1024 ** 3), 1),
                "process_ram_gb": round(process.memory_info().rss / (1024 ** 3), 2),
            }
        except ImportError:
            pass

        return {
            "status": "ok" if self._model is not None else "unloaded",
            "model": self.profile.recommended_model,
            "model_path": self._model_path if self._model else None,
            "context_window": self.profile.n_ctx,
            "gpu": self.profile.gpu_name,
            "ssd_overflow": self.profile.ssd_overflow_active,
            "uptime_seconds": round(uptime, 1),
            "idle_seconds": round(time.time() - self._last_used_time, 1),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "hardware": {
                "total_ram_gb": self.profile.total_ram_gb,
                "cpu_threads": self.profile.n_threads,
                "gpu_layers": self.profile.n_gpu_layers,
            },
            **ram_info,
        }

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def check_idle_unload(self) -> bool:
        """Unload model if idle longer than LOCAL_CODEX_IDLE_UNLOAD_SECONDS."""
        if (
            self._model is not None
            and self._idle_unload_seconds > 0
            and (time.time() - self._last_used_time) > self._idle_unload_seconds
        ):
            log.info(
                "Model idle for >%ds; unloading to reclaim VRAM/RAM.",
                self._idle_unload_seconds,
            )
            self.unload()
            return True
        return False

    def unload(self):
        """Unload the model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            log.info("Model unloaded.")
