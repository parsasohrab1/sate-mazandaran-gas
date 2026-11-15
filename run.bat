@echo off
REM run.bat — launcher for run.ps1 (bypasses PowerShell execution policy)
REM Double-click this file to set up and start the backend.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
echo.
echo Server has stopped. Press any key to close this window.
pause >nul
