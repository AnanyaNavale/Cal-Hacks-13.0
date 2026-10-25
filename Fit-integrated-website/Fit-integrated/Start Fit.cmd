@echo off
setlocal
pushd "%~dp0"
echo Preparing Fit. Keep this window open.
where python >nul 2>nul
if errorlevel 1 (
  echo Python is required. Install Python 3.11 or 3.12 and enable Add Python to PATH.
  pause
  exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo Node.js LTS is required. Install it, then run this file again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
if errorlevel 1 goto failed
pushd figma-web
call npm.cmd install --no-audit --no-fund
if errorlevel 1 goto failed
call npm.cmd run build
if errorlevel 1 goto failed
popd
echo Open http://localhost:8765 in Chrome or Edge.
".venv\Scripts\python.exe" run_fit.py
if errorlevel 1 goto failed
popd
exit /b 0
:failed
echo Setup did not finish. Copy the error above for help.
pause
exit /b 1
