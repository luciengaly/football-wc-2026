@echo off
REM Daily refresh script for WC 2026 prediction system.
REM Logs to data/snapshots/refresh.log so failures stay visible.
REM
REM Manual run:
REM   scripts\daily_refresh.bat
REM
REM Schedule via Windows Task Scheduler (run as current user, no elevation):
REM   1. Open Task Scheduler
REM   2. Create Basic Task -> trigger: Daily at 09:00
REM   3. Action: Start a program
REM        Program/script: %~dp0daily_refresh.bat
REM        Start in:       <repo root>
REM
REM See docs/OPERATIONS.md for full instructions.

setlocal
cd /d "%~dp0\.."

REM Timestamp every line
set TS=%date% %time%

echo. >> data\snapshots\refresh.log
echo === %TS% === Starting daily refresh >> data\snapshots\refresh.log

call .venv\Scripts\activate.bat

python -m wc2026.pipeline refresh >> data\snapshots\refresh.log 2>&1
set RC=%errorlevel%

if %RC% NEQ 0 (
    echo === %TS% === FAILED with exit code %RC% >> data\snapshots\refresh.log
    exit /b %RC%
)

echo === %TS% === DONE >> data\snapshots\refresh.log
endlocal
