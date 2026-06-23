$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

& $python (Join-Path $root "scripts\download_small_dataset.py") @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

