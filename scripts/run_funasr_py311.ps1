$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv311\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python 3.11 virtual environment not found. Run .\scripts\setup_funasr_py311.ps1 first."
}

if (-not $env:ASR_BACKEND) {
    $env:ASR_BACKEND = "funasr"
}
if (-not $env:ASR_MODEL) {
    $env:ASR_MODEL = "FunAudioLLM/SenseVoiceSmall"
}
if (-not $env:ASR_DEVICE) {
    $env:ASR_DEVICE = "cpu"
}

& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
