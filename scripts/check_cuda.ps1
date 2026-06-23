$ErrorActionPreference = "Stop"

$names = @("cublas64_12.dll", "cudnn64_9.dll", "cudnn64_8.dll")
$paths = $env:PATH -split ";"
$cudaRoots = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
)

foreach ($name in $names) {
    $found = $false
    foreach ($path in ($paths + $cudaRoots)) {
        if (-not $path) { continue }
        $candidate = Join-Path $path $name
        if (Test-Path $candidate) {
            Write-Host "$name`t$candidate"
            $found = $true
            break
        }
    }
    if (-not $found) {
        Write-Host "$name`tMISSING"
    }
}

Write-Host ""
Write-Host "If cublas64_12.dll is missing, keep ASR_DEVICE=cpu or install a CUDA 12 runtime/toolkit that places cuBLAS on PATH."

