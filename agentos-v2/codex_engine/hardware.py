"""
hardware.py — System hardware detection for AgentOS Codex Engine.

Detects RAM, GPU VRAM, CPU cores, and SSD presence to auto-configure
the optimal model size, context window, and thread count.
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("codex_engine.hardware")


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Profile
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HardwareProfile:
    """Snapshot of the system's capabilities."""

    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    cpu_cores: int = 1
    cpu_threads: int = 1
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    has_ssd: bool = False
    os_name: str = ""

    # ── Derived configuration (filled by select_model_config) ──────────────
    recommended_model: str = ""
    recommended_repo: str = ""
    recommended_file: str = ""
    model_size_gb: float = 0.0
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    n_threads: int = 4
    use_mmap: bool = True
    use_mlock: bool = False
    ssd_overflow_active: bool = False

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║       AgentOS Codex — Hardware Profile           ║",
            "╚══════════════════════════════════════════════════╝",
            f"  OS:             {self.os_name}",
            f"  CPU:            {self.cpu_cores} cores / {self.cpu_threads} threads",
            f"  RAM:            {self.total_ram_gb:.1f} GB total / {self.available_ram_gb:.1f} GB available",
        ]
        if self.gpu_name:
            lines.append(f"  GPU:            {self.gpu_name} ({self.gpu_vram_gb:.1f} GB VRAM)")
        else:
            lines.append("  GPU:            None detected (CPU-only mode)")
        lines.append(f"  SSD detected:   {'Yes' if self.has_ssd else 'No'}")
        lines.append("  ──────────────────────────────────────────────")
        lines.append(f"  Model:          {self.recommended_model}")
        lines.append(f"  Context:        {self.n_ctx} tokens")
        lines.append(f"  Threads:        {self.n_threads}")
        lines.append(f"  GPU layers:     {self.n_gpu_layers}")
        lines.append(f"  mmap (SSD):     {'ACTIVE — SSD overflow enabled' if self.ssd_overflow_active else 'standby'}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Detection Functions
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_ram() -> tuple[float, float]:
    """Returns (total_gb, available_gb)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3), mem.available / (1024 ** 3)
    except ImportError:
        log.warning("psutil not installed — cannot detect RAM, assuming 8 GB")
        return 8.0, 6.0


def _detect_cpu() -> tuple[int, int]:
    """Returns (physical_cores, logical_threads)."""
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or 1
        threads = psutil.cpu_count(logical=True) or 1
        return cores, threads
    except ImportError:
        cores = os.cpu_count() or 1
        return cores, cores


def _detect_gpu() -> tuple[Optional[str], Optional[float]]:
    """Returns (gpu_name, vram_gb) or (None, None) if no NVIDIA GPU."""
    try:
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
        )

        nvmlInit()
        count = nvmlDeviceGetCount()
        if count == 0:
            nvmlShutdown()
            return None, None

        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        name = nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        vram_gb = info.total / (1024 ** 3)
        nvmlShutdown()
        return name, vram_gb

    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            name = parts[0].strip()
            vram_mb = float(parts[1].strip())
            return name, vram_mb / 1024
    except Exception:
        pass

    return None, None


def _detect_ssd() -> bool:
    """Best-effort SSD detection."""
    system = platform.system()

    if system == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-PhysicalDisk | Select-Object MediaType | ConvertTo-Json"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                return "ssd" in output or "nvme" in output or "solid" in output
        except Exception:
            pass

    elif system == "Linux":
        try:
            result = subprocess.run(
                ["lsblk", "-d", "-o", "name,rota", "--noheadings"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == "0":
                        return True
        except Exception:
            pass

    elif system == "Darwin":
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Model Selection
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = [
    {
        "name": "Qwen2.5-Coder-32B-Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "file": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "size_gb": 20.0,
        "min_ram_gb": 24.0,
        "min_ram_ssd_gb": 12.0,
        "tier": "32b",
    },
    {
        "name": "Qwen2.5-Coder-14B-Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "file": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "size_gb": 9.0,
        "min_ram_gb": 12.0,
        "min_ram_ssd_gb": 6.0,
        "tier": "14b",
    },
    {
        "name": "Qwen2.5-Coder-7B-Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.4,
        "min_ram_gb": 6.0,
        "min_ram_ssd_gb": 4.0,
        "tier": "7b",
    },
    {
        "name": "Qwen2.5-Coder-7B-Instruct (Q3_K_S — ultra-light)",
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "qwen2.5-coder-7b-instruct-q3_k_s.gguf",
        "size_gb": 3.2,
        "min_ram_gb": 4.0,
        "min_ram_ssd_gb": 3.0,
        "tier": "7b-light",
    },
]


def select_model_config(profile: HardwareProfile, force_size: str = None) -> HardwareProfile:
    """
    Pick the optimal model and configure inference parameters.

    Strategy:
      1. If RAM alone can hold the model → load normally
      2. If RAM is tight, SSD exists, and LOCAL_CODEX_SSD_OVERFLOW != "false" → enable mmap overflow
      3. If nothing fits → return empty (caller should fall back to cloud)
    """
    available = profile.available_ram_gb
    ssd_overflow_enabled = os.getenv("LOCAL_CODEX_SSD_OVERFLOW", "true").lower() == "true"
    allow_ssd_overflow = profile.has_ssd and ssd_overflow_enabled

    chosen = None

    for model in MODEL_REGISTRY:
        if force_size and model["tier"] != force_size:
            continue

        if available >= model["min_ram_gb"]:
            chosen = model
            profile.ssd_overflow_active = False
            break

        if allow_ssd_overflow and available >= model["min_ram_ssd_gb"]:
            chosen = model
            profile.ssd_overflow_active = True
            log.warning(
                "Model %s (%.1f GB) exceeds available RAM (%.1f GB) — "
                "SSD mmap overflow will be used. Expect ~2-5 tok/s instead of ~15 tok/s.",
                model["name"], model["size_gb"], available,
            )
            break

    if chosen is None:
        if os.getenv("BYPASS_RAM_CHECK", "false").lower() == "true":
            if force_size:
                for model in MODEL_REGISTRY:
                    if model["tier"] == force_size:
                        chosen = model
                        break
            if chosen is None:
                chosen = MODEL_REGISTRY[-1]
            profile.ssd_overflow_active = ssd_overflow_enabled
            log.warning(
                "BYPASS_RAM_CHECK is active. Forcing model selection: %s (%.1f GB) despite low RAM (%.1f GB available).",
                chosen["name"], chosen["size_gb"], available,
            )
        else:
            log.error(
                "Not enough memory for any local model (%.1f GB available). "
                "Need at least 3 GB RAM + SSD, or 4 GB RAM. Falling back to cloud.",
                available,
            )
            profile.recommended_model = ""
            return profile

    profile.recommended_model = chosen["name"]
    profile.recommended_repo = chosen["repo"]
    profile.recommended_file = chosen["file"]
    profile.model_size_gb = chosen["size_gb"]

    remaining = max(0.5, available - chosen["size_gb"] - 1.0)
    estimated_ctx = int(remaining * 2048)
    profile.n_ctx = max(4096, min(32768, estimated_ctx))

    profile.n_gpu_layers = -1 if profile.gpu_vram_gb else 0
    profile.n_threads = max(1, min(profile.cpu_cores, 12))
    profile.use_mmap = ssd_overflow_enabled or not profile.ssd_overflow_active
    profile.use_mlock = False

    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def detect_hardware(force_size: str = None) -> HardwareProfile:
    """
    Full hardware detection + model selection in one call.
    """
    log.info("Detecting hardware...")

    total_ram, available_ram = _detect_ram()
    cores, threads = _detect_cpu()
    gpu_name, gpu_vram = _detect_gpu()
    has_ssd = _detect_ssd()

    profile = HardwareProfile(
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        cpu_cores=cores,
        cpu_threads=threads,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        has_ssd=has_ssd,
        os_name=f"{platform.system()} {platform.release()}",
    )

    profile = select_model_config(profile, force_size=force_size)

    log.info("\n%s", profile.summary())
    return profile


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    profile = detect_hardware()
    print()
    print(profile.summary())
