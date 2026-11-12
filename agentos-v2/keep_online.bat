@echo off
title AgentOS Watchdog & Keep-Alive Service
echo ===================================================
echo   AgentOS Keep-Alive Daemon
echo   Monitoring FastAPI server at http://127.0.0.1:8080/health
echo ===================================================
echo.

:loop
netstat -ano | findstr LISTENING | findstr :8080 >nul
if %errorlevel% equ 0 (
    echo [%date% %time%] Server is online and listening on port 8080.
) else (
    echo [%date% %time%] WARNING: FastAPI server is OFFLINE! Restarting it now...
    start /b python main.py
    timeout /t 5 >nul
)

timeout /t 30 >nul
goto loop
