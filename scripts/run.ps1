$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

if (-not $env:ASR_MODEL) {
    $env:ASR_MODEL = "small"
}

if (-not $env:ASR_BACKEND) {
    $env:ASR_BACKEND = "whisper"
}

if (-not $env:ASR_DEVICE) {
    $env:ASR_DEVICE = "cpu"
}

if (-not $env:ASR_COMPUTE_TYPE) {
    $env:ASR_COMPUTE_TYPE = "int8"
}

& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
