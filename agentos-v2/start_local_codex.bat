@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  AgentOS Local Codex — Standalone Testing Server
REM
REM  Starts the codex_engine as a standalone server for testing.
REM  The engine auto-detects your hardware and picks the best model.
REM
REM  USAGE:
REM    start_local_codex.bat              (auto-detect everything)
REM
REM  PREREQUISITES:
REM    pip install psutil huggingface-hub
REM    pip install llama-cpp-python       (CPU)
REM    -- or --
REM    set CMAKE_ARGS=-DGGML_CUDA=on
REM    pip install llama-cpp-python       (NVIDIA GPU)
REM ═══════════════════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   AgentOS Codex Engine — Standalone Mode         ║
echo  ╚══════════════════════════════════════════════════╝
echo.

REM Run the Python engine server
python -m codex_engine.server

if errorlevel 1 (
    echo.
    echo  [ERROR] Server failed to start.
    echo  Make sure you have installed the dependencies:
    echo    pip install psutil huggingface-hub llama-cpp-python
    echo.
    pause
)
