@echo off
rem SessionStart hook: emit the scout summary as additionalContext.
rem Windows uses this wrapper because .sh hooks can be opened by the default
rem file association instead of being executed.
setlocal EnableExtensions DisableDelayedExpansion

set "ROOT=%CODEX_PROJECT_DIR%"
if not defined ROOT set "ROOT=%CD%"

set "TINYCTX_HOME_VALUE=%TINYCTX_HOME%"
if not defined TINYCTX_HOME_VALUE set "TINYCTX_HOME_VALUE=%USERPROFILE%\.tinyctx"

set "VENV_PY="
for %%P in ("%TINYCTX_HOME_VALUE%\.venv\Scripts\python.exe" "%~dp0..\.venv\Scripts\python.exe" "%USERPROFILE%\dev\tinyctx\.venv\Scripts\python.exe") do (
  if not defined VENV_PY if exist "%%~P" set "VENV_PY=%%~P"
)

if not defined VENV_PY (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined VENV_PY set "VENV_PY=%%P"
  )
)

if not defined VENV_PY exit /b 0

set "STATE="
for /f "usebackq delims=" %%S in (`"%VENV_PY%" -m tinyctx.scout status --root "%ROOT%" --json 2^>nul ^| "%VENV_PY%" -c "import json,sys; d=json.load(sys.stdin); print(d.get('state',''))" 2^>nul`) do (
  set "STATE=%%S"
)

if "%STATE%"=="fresh" goto fresh
if "%STATE%"=="stale" goto stale
exit /b 0

:fresh
call :emit_scout ""
exit /b 0

:stale
if not exist "%TINYCTX_HOME_VALUE%\logs" mkdir "%TINYCTX_HOME_VALUE%\logs" >nul 2>nul
start "" /b cmd /c ""%VENV_PY%" -m tinyctx.scout refresh --root "%ROOT%" >> "%TINYCTX_HOME_VALUE%\logs\scout-refresh.log" 2>&1"
call :emit_scout "[tinyctx: refreshing scout in background]"
exit /b 0

:emit_scout
set "SCOUT_MD="
for /f "usebackq delims=" %%P in (`"%VENV_PY%" -m tinyctx.scout path --root "%ROOT%" 2^>nul`) do (
  set "SCOUT_MD=%%P"
)
if not defined SCOUT_MD exit /b 0
if not exist "%SCOUT_MD%" exit /b 0
"%VENV_PY%" -c "import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); suffix=sys.argv[2]; text=p.read_text(encoding='utf-8'); text=text + ('\n\n' + suffix if suffix else ''); print(json.dumps({'additionalContext': text}))" "%SCOUT_MD%" "%~1"
exit /b 0
