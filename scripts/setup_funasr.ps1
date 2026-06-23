$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

& $python -m pip install -r (Join-Path $root "requirements-funasr.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FunASR install failed. On Windows, Python 3.11 is recommended. Python 3.13 may require Microsoft C++ Build Tools for editdistance."
    exit $LASTEXITCODE
}
