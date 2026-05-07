# tinyctx-proxy start script for Windows
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $root) { $root = Split-Path -Parent $PSScriptRoot }

$venvPython = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
}

if (-not (Test-Path $venvPython)) {
    Write-Error "venv not found. Run: python -m venv C:\Dev\tinyctx\.venv && .venv\Scripts\pip install -e ."
    exit 1
}

Write-Host "[tinyctx] starting proxy on 127.0.0.1:4141" -ForegroundColor Cyan
& $venvPython -m tinyctx.proxy
