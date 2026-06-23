$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

& $python -m pip install -r (Join-Path $root "requirements-datasets.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Dataset tooling install failed. Python 3.11 is recommended if Python 3.13 wheels are unavailable."
    exit $LASTEXITCODE
}

