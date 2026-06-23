# Local Voice Service

Windows local ASR + TTS test service.

- ASR backend 1: `faster-whisper`
- ASR backend 2: optional `SenseVoice / FunASR`
- TTS: Windows SAPI by default; optional Piper
- API/UI: FastAPI plus a local web page

## Install

```powershell
.\scripts\setup.ps1
```

Optional SenseVoice/FunASR backend:

```powershell
winget install --id Python.Python.3.11 -e
winget install --id Gyan.FFmpeg -e
.\scripts\setup_funasr_py311.ps1
```

On Windows, Python 3.11 is recommended for FunASR/SenseVoice. This project keeps it in `.venv311` because the main `.venv` may be Python 3.13.

Optional dataset tooling:

```powershell
.\scripts\setup_datasets.ps1
```

## Run

```powershell
.\scripts\run.ps1
```

Run the web service with SenseVoice/FunASR enabled:

```powershell
.\scripts\run_funasr_py311.ps1
```

Open:

```text
http://127.0.0.1:8000/
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## ASR

Whisper CPU baseline:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/asr?backend=whisper&model_size=small&device=cpu&compute_type=int8&language=zh" `
  -F "file=@C:\path\to\audio.wav"
```

SenseVoice/FunASR:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/asr?backend=funasr&model_size=FunAudioLLM/SenseVoiceSmall&device=cpu&language=auto" `
  -F "file=@C:\path\to\audio.wav"
```

## TTS

```powershell
$body = @{ text = "你好，这是本地语音合成测试。"; rate = 0; volume = 100 } | ConvertTo-Json -Compress
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-WebRequest -Uri "http://127.0.0.1:8000/tts" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $bytes `
  -OutFile output.wav
```

## Benchmarks

TTS -> ASR self-test:

```powershell
.\.venv\Scripts\python .\scripts\benchmark_tts_asr.py --count 100 --backend whisper --model-size small --device cpu --compute-type int8 --keep-wavs
```

External dataset manifest:

```powershell
.\.venv\Scripts\python .\scripts\benchmark_manifest.py --manifest .\samples\dataset.csv --backend whisper --model-size small --device cpu --compute-type int8
```

Small dataset export:

```powershell
.\scripts\download_small_dataset.ps1 --dataset minds14-zh --limit 100 --max-mb 100
```

SenseVoice on the small MInDS-14 zh-CN sample:

```powershell
.\scripts\benchmark_sensevoice_minds14.ps1
```

Dataset notes and current comparison numbers are in `docs/datasets.md`.

## CUDA Check

The current safe default is CPU:

```powershell
$env:ASR_DEVICE = "cpu"
```

Check CUDA runtime DLLs:

```powershell
.\scripts\check_cuda.ps1
```

If `cublas64_12.dll` is missing, `device=auto` will stay on CPU. After installing a CUDA 12 runtime/toolkit with cuBLAS on `PATH`, use:

```powershell
$env:ASR_DEVICE = "cuda"
$env:ASR_COMPUTE_TYPE = "int8_float16"
.\scripts\run.ps1
```
