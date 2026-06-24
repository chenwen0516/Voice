$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv311\Scripts\python.exe"
$manifest = Join-Path $root "samples\datasets\minds14-zh\manifest.csv"
$outputDir = Join-Path $root "samples\datasets\minds14-zh-sensevoice-small"

if (-not (Test-Path $python)) {
    throw "Python 3.11 virtual environment not found. Run .\scripts\setup_funasr_py311.ps1 first."
}

if (-not (Test-Path $manifest)) {
    throw "MInDS-14 manifest not found. Run .\scripts\download_small_dataset.ps1 --dataset minds14-zh --limit 100 --max-mb 100 first."
}

& $python (Join-Path $root "scripts\benchmark_manifest.py") `
    --manifest $manifest `
    --output-dir $outputDir `
    --backend funasr `
    --model-size "FunAudioLLM/SenseVoiceSmall" `
    --device cpu `
    --language zh `
    --funasr-use-itn true `
    --funasr-merge-vad true `
    --funasr-merge-length-s 30

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
