@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo ERROR: Local Python environment was not found:
  echo   %VENV_PY%
  echo.
  echo Create it with:
  echo   python -m venv .venv
  echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

"%VENV_PY%" -B main.py from-settings %*
exit /b %ERRORLEVEL%
