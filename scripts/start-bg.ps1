# tinyctx-proxy — background launcher
# Starts the proxy detached, logs to ~/.tinyctx/logs/proxy.log
$logDir = "$env:USERPROFILE\.tinyctx\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$logFile = "$logDir\proxy.log"
$python = "C:\Dev\tinyctx\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "[tinyctx] ERROR: venv not found at $python" -ForegroundColor Red
    pause
    exit 1
}

# Kill any existing proxy by port
$existing = Get-NetTCPConnection -LocalPort 4141 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($existing) {
    foreach ($procId in $existing) {
        Write-Host "[tinyctx] killing old proxy (PID $procId)" -ForegroundColor Yellow
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# Load persistent user env vars (so DEEPSEEK_API_KEY etc. are available)
foreach ($name in @("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "TINYCTX_LOCAL_API_KEY", "TINYCTX_FRONTIER_API_KEY")) {
    $val = [Environment]::GetEnvironmentVariable($name, "User")
    if ($val -and -not (Test-Path "env:\$name")) {
        Set-Item "env:\$name" $val
    }
}

Start-Process -FilePath $python -ArgumentList "-m","tinyctx.proxy" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "$logDir\proxy-err.log"

Write-Host "[tinyctx] proxy started in background, log: $logFile" -ForegroundColor Cyan
