"""
demo.py — AgentOS Capability and Comparison Demo.

Showcases:
1. Architectural Comparison (Original vs Current v3)
2. Hardware Auto-Detection & SSD Overflow capacity
3. IDE Code Recommendation & Patch Formatting
"""

import sys
import os
from pathlib import Path

# Force UTF-8 stdout for Windows compatibility with box-drawing characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project path to python path
sys.path.append(str(Path(__file__).parent))

# Import hardware detection
try:
    from codex_engine.hardware import detect_hardware
except ImportError:
    detect_hardware = None


def print_banner():
    banner = """\033[94m
 ╦═╗╔═╗╔═╗╔═╗╔╦╗╔═╗╔═╗  ╦  ╦╔═  ╔╦╗╔═╗╔╦╗╔═╗
 ╠╦╝║╣ ║  ║ ║║║║║╣ ║ ║  ╚╗╔╝║    ║║║╣ ║║║║ ║
 ╩╚═╚═╝╚═╝╚═╝╩ ╩╚═╝╚═╝   ╚╝ ╚═╝  ╩ ╩╚═╝╩ ╩╚═╝
    \033[92m* SYSTEM AUDIT, CAPABILITY DEMO & IDE INTEGRATION *\033[0m
"""
    print(banner)


def show_comparison():
    print("\033[1m\033[93m" + "=" * 80 + "\033[0m")
    print("\033[1m  1. ARCHITECTURAL COMPARISON: ORIGINAL VS CURRENT AgentOS v3\033[0m")
    print("\033[1m\033[93m" + "=" * 80 + "\033[0m")

    # Beautiful formatted table
    headers = f"{'Feature Area':<25} | {'Original AgentOS (v1)':<24} | {'Current AgentOS (v3)':<25}"
    separator = "-" * len(headers)
    print(headers)
    print(separator)

    rows = [
        ("Multi-Agent Flow", "Single Linear Chain", "Supervisor Route Graph"),
        ("Model Tiering", "Single LLM (Gemini Pro)", "3-Tier Hybrid (Claude/Gem/Local)"),
        ("Inference Engine", "Cloud Only", "Self-Contained Local + Cloud"),
        ("Local Auto-Scale", "No (static manual)", "RAM/VRAM Auto-Detect (7B/14B/32B)"),
        ("SSD Overflow Memory", "Unsupported (Crashes)", "mmap swapping (Guarantees load)"),
        ("ReAct Loop", "Single execution turn", "Autonomous ReAct Loop (Multi-turn)"),
        ("Memory Token Budget", "None (system message only)", "Compressed Memory Token Manager"),
        ("Planning Pass", "No planning step", "Forced Plan-Before-Action Pass"),
        ("File Operation", "Write/Replace entire file", "Targeted Line Patching (patch_file)"),
        ("IDE Dashboard", "Basic Chat Box Only", "Integrated IDE & Admin Dashboard"),
    ]

    for feat, orig, current in rows:
        print(f"\033[91m{feat:<25}\033[0m | {orig:<24} | \033[92m{current:<25}\033[0m")

    print(separator)
    print()


def show_hardware_profile():
    print("\033[1m\033[93m" + "=" * 80 + "\033[0m")
    print("\033[1m  2. HARDWARE DIAGNOSTICS & SCALABILITY TIER\033[0m")
    print("\033[1m\033[93m" + "=" * 80 + "\033[0m")

    if not detect_hardware:
        print("\033[91mError: Could not import codex_engine.hardware\033[0m")
        return

    profile = detect_hardware()
    print(profile.summary())

    print("\n\033[1m\033[96mSSD Overflow Capacity Scenario Analyser:\033[0m")
    print("  • If you run Qwen 7B (requires 6.0 GB RAM):")
    if profile.available_ram_gb >= 6.0:
        print("    --> \033[92mLoads fully in RAM (Instant response, high speed).\033[0m")
    elif profile.has_ssd:
        print("    --> \033[93mRAM tight. Loads with SSD mmap overflow. (Runs successfully, medium speed).\033[0m")
    else:
        print("    --> \033[91mRAM tight and SSD not detected. Will fall back to Cloud Tier to avoid system crash.\033[0m")

    print("  • If you run Qwen 32B (requires 24.0 GB RAM):")
    if profile.available_ram_gb >= 24.0:
        print("    --> \033[92mLoads fully in RAM (Instant response, high speed).\033[0m")
    elif profile.has_ssd:
        print("    --> \033[93mLoads with SSD mmap overflow. (Runs successfully, slower speed ~2-5 tokens/sec).\033[0m")
    else:
        print("    --> \033[91mExceeds memory limits. Falls back to Senior Cloud Tier (Claude Opus/Gemini Pro).\033[0m")
    print()


def show_ide_patch_format():
    print("\033[1m\033[93m" + "=" * 80 + "\033[0m")
    print("\033[1m  3. IDE COMPATIBILITY & PATCH RECOMMENDATIONS\033[0m")
    print("\033[1m\033[93m" + "=" * 80 + "\033[0m")
    print("AgentOS outputs code recommendations in standard Unified Diff or JSON Patch formats,")
    print("allowing seamless integration with VS Code, Cursor, or any other IDE.")
    print("\n\033[1m\033[96mExample VS Code Unified Diff Patch recommendation:\033[0m\n")

    diff_example = """\033[36m@@ -10,8 +10,8 @@\033[0m
-def calculate_fibonacci(n):
-    if n <= 1:
-        return n
-    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
+def calculate_fibonacci(n):
+    # Optimized using Dynamic Programming (O(n) time, O(1) space)
+    if n <= 1:
+        return n
+    a, b = 0, 1
+    for _ in range(2, n + 1):
+        a, b = b, a + b
+    return b"""

    print(diff_example)
    print("\n\033[92m✔ Clean format: Easy to parse, can be applied directly using 'git apply' or VS Code patch extension.\033[0m")
    print("\033[92m✔ Safety: Only touches specified lines (no complete file overrides which can wipe changes).\033[0m")
    print("\033[93m" + "=" * 80 + "\033[0m\n")


def main():
    print_banner()
    show_comparison()
    show_hardware_profile()
    show_ide_patch_format()


if __name__ == "__main__":
    main()
