# ASR Dataset Candidates

These are practical public datasets for the next round of ASR comparison.

| Dataset | Language / style | Why use it | Notes |
| --- | --- | --- | --- |
| PolyAI MInDS-14 zh-CN | Mandarin task-oriented user speech | Very small first real-speech benchmark | 100 WAV samples used about 5.92MB here |
| AISHELL-1 | Mandarin, read speech | Clean baseline for Mandarin ASR accuracy | Large download; good first offline benchmark once cached |
| Mozilla Common Voice Chinese | Crowd-sourced Mandarin and accents | Better coverage of speakers and recording conditions | Versions change over time; use a pinned release |
| WenetSpeech | Large-scale Mandarin speech | Good stress test for real-world Mandarin | Much larger than AISHELL-1; use a small manifest subset first |

Recommended order:

1. Use `PolyAI/minds14` zh-CN for a small first benchmark.
2. Use AISHELL-1 test split for a clean Mandarin baseline.
3. Use a small Common Voice zh-CN sample for speaker/accent diversity.
4. Use WenetSpeech only after the benchmark flow is stable, because the dataset is much bigger.

## Small Dataset: MInDS-14 zh-CN

Install the dataset tooling:

```powershell
.\scripts\setup_datasets.ps1
```

Download/export about 100 samples, capped near 100MB of WAV files:

```powershell
.\scripts\download_small_dataset.ps1 --dataset minds14-zh --limit 100 --max-mb 100
```

This creates:

```text
samples\datasets\minds14-zh\manifest.csv
samples\datasets\minds14-zh\audio\*.wav
```

On this machine, 100 exported MInDS-14 zh-CN WAV files used about 5.92MB. The full zh-CN subset has 502 rows, so it is still a small dataset; use `--limit 502` if you want the whole subset.

Run Whisper on the exported set:

```powershell
.\.venv\Scripts\python .\scripts\benchmark_manifest.py `
  --manifest .\samples\datasets\minds14-zh\manifest.csv `
  --backend whisper `
  --model-size small `
  --device cpu `
  --compute-type int8
```

Run SenseVoice/FunASR on the same set:

```powershell
.\scripts\benchmark_sensevoice_minds14.ps1
```

Current 100-sample results on this machine:

| Backend | Model | Device | Exact rate | Average CER | Median CER | Avg ASR time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Whisper | `tiny` | CPU int8 | 8% | 42.95% | 28.17% | 0.59s |
| Whisper | `small` | CPU int8 | 26% | 26.45% | 14.09% | 2.21s |
| SenseVoice / FunASR | `FunAudioLLM/SenseVoiceSmall` | CPU | 46% | 13.71% | 3.64% | 0.55s |

SenseVoice had the best accuracy in this small local run. Some high-CER rows appear to have loose or truncated reference transcripts, so this is a practical smoke benchmark rather than a clean leaderboard.

## SenseVoice Tuning Notes

The benchmark now reports three useful accuracy views:

- `cer`: strict character error rate after punctuation and whitespace normalization.
- `clean_cer`: CER after also ignoring common filler words such as `呃` and `嗯`.
- `contains_reference_rate`: how often the reference transcript appears inside the ASR output, which helps identify rows where the dataset transcript is shorter than the spoken audio.

Current 100-sample SenseVoice parameter sweep:

| Config | Exact | Clean exact | Contains reference | Avg CER | Avg clean CER | Contains-adjusted clean CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zh-itn-vad30` | 46% | 48% | 52% | 13.71% | 13.11% | 8.90% |
| `zh-itn-vad15` | 46% | 48% | 52% | 13.71% | 13.11% | 8.90% |
| `zh-itn-novad` | 46% | 48% | 52% | 13.71% | 13.11% | 8.90% |
| `auto-itn-vad15` | 48% | 50% | 54% | 14.08% | 13.46% | 9.12% |
| `zh-noitn-vad15` | 47% | 49% | 53% | 14.08% | 13.51% | 9.18% |

Recommendation from this run: keep Chinese language fixed (`zh`) and keep ITN enabled. `vad30` is the default because it matched the best CER group and was slightly faster in this local run.

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

For SenseVoice/FunASR with Python 3.11:

```powershell
.\scripts\setup_funasr_py311.ps1
.\.venv311\Scripts\python .\scripts\benchmark_manifest.py `
  --manifest .\samples\dataset.csv `
  --backend funasr `
  --model-size FunAudioLLM/SenseVoiceSmall `
  --device cpu
```
