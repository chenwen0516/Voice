param(
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv311\Scripts\python.exe"
$manifest = Join-Path $root "samples\datasets\minds14-zh\manifest.csv"
$outputRoot = Join-Path $root "samples\datasets\minds14-zh-sensevoice-matrix"

if (-not (Test-Path $python)) {
    throw "Python 3.11 virtual environment not found. Run .\scripts\setup_funasr_py311.ps1 first."
}

if (-not (Test-Path $manifest)) {
    throw "MInDS-14 manifest not found. Run .\scripts\download_small_dataset.ps1 --dataset minds14-zh --limit 100 --max-mb 100 first."
}

$configs = @(
    @{ Name = "zh-itn-vad15"; Language = "zh"; UseItn = "true"; MergeVad = "true"; MergeLength = 15 },
    @{ Name = "auto-itn-vad15"; Language = "auto"; UseItn = "true"; MergeVad = "true"; MergeLength = 15 },
    @{ Name = "zh-noitn-vad15"; Language = "zh"; UseItn = "false"; MergeVad = "true"; MergeLength = 15 },
    @{ Name = "zh-itn-novad"; Language = "zh"; UseItn = "true"; MergeVad = "false"; MergeLength = 15 },
    @{ Name = "zh-itn-vad30"; Language = "zh"; UseItn = "true"; MergeVad = "true"; MergeLength = 30 }
)

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$summaryRows = @()

foreach ($config in $configs) {
    $out = Join-Path $outputRoot $config.Name
    Write-Host ""
    Write-Host "Running $($config.Name)..."

    $arguments = @(
        (Join-Path $root "scripts\benchmark_manifest.py"),
        "--manifest", $manifest,
        "--output-dir", $out,
        "--backend", "funasr",
        "--model-size", "FunAudioLLM/SenseVoiceSmall",
        "--device", "cpu",
        "--language", $config.Language,
        "--funasr-use-itn", $config.UseItn,
        "--funasr-merge-vad", $config.MergeVad,
        "--funasr-merge-length-s", [string]$config.MergeLength
    )
    if ($Limit -gt 0) {
        $arguments += @("--limit", [string]$Limit)
    }

    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $summary = Get-Content (Join-Path $out "dataset-summary.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    $summaryRows += [pscustomobject]@{
        name = $config.Name
        exact_rate = $summary.exact_rate
        clean_exact_rate = $summary.clean_exact_rate
        contains_reference_rate = $summary.contains_reference_rate
        average_cer = $summary.average_cer
        average_clean_cer = $summary.average_clean_cer
        average_contains_adjusted_clean_cer = $summary.average_contains_adjusted_clean_cer
        median_clean_cer = $summary.median_clean_cer
        average_asr_seconds = $summary.average_asr_seconds
    }
}

$matrixPath = Join-Path $outputRoot "matrix-summary.json"
$summaryRows |
    Sort-Object average_contains_adjusted_clean_cer, average_clean_cer, average_cer |
    ConvertTo-Json -Depth 4 |
    Set-Content -Encoding UTF8 $matrixPath

Write-Host ""
Write-Host "Matrix summary:"
$summaryRows |
    Sort-Object average_contains_adjusted_clean_cer, average_clean_cer, average_cer |
    Format-Table -AutoSize
Write-Host "Saved: $matrixPath"
