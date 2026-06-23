# ASR Dataset Candidates

These are practical public datasets for the next round of ASR comparison.

| Dataset | Language / style | Why use it | Notes |
| --- | --- | --- | --- |
| AISHELL-1 | Mandarin, read speech | Clean baseline for Mandarin ASR accuracy | Large download; good first offline benchmark once cached |
| Mozilla Common Voice Chinese | Crowd-sourced Mandarin and accents | Better coverage of speakers and recording conditions | Versions change over time; use a pinned release |
| WenetSpeech | Large-scale Mandarin speech | Good stress test for real-world Mandarin | Much larger than AISHELL-1; use a small manifest subset first |

Recommended order:

1. Use AISHELL-1 test split for a clean Mandarin baseline.
2. Use a small Common Voice zh-CN sample for speaker/accent diversity.
3. Use WenetSpeech only after the benchmark flow is stable, because the dataset is much bigger.

`scripts/benchmark_manifest.py` accepts a CSV or JSONL manifest with these columns:

```text
audio,text
C:\path\to\sample.wav,这是一条转写文本
```

Alternative audio column names are also accepted: `audio_path`, `path`, `wav`, `file`.
Alternative text column names are also accepted: `transcript`, `sentence`, `normalized_text`.

Example:

```powershell
.\.venv\Scripts\python .\scripts\benchmark_manifest.py `
  --manifest .\samples\dataset.csv `
  --backend whisper `
  --model-size small `
  --device cpu `
  --compute-type int8
```

For SenseVoice/FunASR:

```powershell
.\scripts\setup_funasr.ps1
.\.venv\Scripts\python .\scripts\benchmark_manifest.py `
  --manifest .\samples\dataset.csv `
  --backend funasr `
  --model-size iic/SenseVoiceSmall `
  --device cpu
```

