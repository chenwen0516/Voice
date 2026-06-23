from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.asr import transcribe_audio
from app.metrics import character_error_rate, normalize_text
from app.tts import synthesize_speech


@dataclass
class BenchmarkRow:
    index: int
    expected: str
    actual: str
    normalized_expected: str
    normalized_actual: str
    cer: float
    edit_distance: int
    expected_chars: int
    exact: bool
    tts_seconds: float
    asr_seconds: float
    wav_path: str


def build_sentences(count: int) -> list[str]:
    subjects = [
        "本地语音服务",
        "会议记录助手",
        "智能客服系统",
        "字幕生成工具",
        "课堂笔记应用",
        "播客剪辑流程",
        "离线语音模型",
        "中文转写引擎",
        "自动配音方案",
        "桌面测试程序",
    ]
    verbs = [
        "正在测试",
        "可以处理",
        "需要识别",
        "已经生成",
        "准备保存",
        "正在比较",
        "可以回放",
        "需要校验",
        "已经完成",
        "准备上传",
    ]
    objects = [
        "一段清晰的普通话音频",
        "带有停顿的中文句子",
        "包含数字二零二六的样本",
        "比较自然的合成声音",
        "十秒以内的短语音",
        "多个不同长度的测试文本",
        "一组稳定可复现的结果",
        "本机离线运行的能力",
        "语音合成和语音识别链路",
        "后续接入应用的基础接口",
    ]
    tails = [
        "请记录准确率和耗时。",
        "结果会写入本地文件。",
        "我们关注错字和漏字。",
        "这条样本用于自动评估。",
        "如果识别稳定就继续优化。",
        "当前优先保证离线可用。",
        "测试完成后再看汇总。",
        "这个句子不依赖网络服务。",
        "模型加载后速度会更快。",
        "今天先看基础效果。",
    ]

    return [
        (
            f"{subjects[index % len(subjects)]}"
            f"{verbs[(index // 2) % len(verbs)]}"
            f"{objects[(index // 3) % len(objects)]}，"
            f"{tails[(index // 5) % len(tails)]}"
        )
        for index in range(count)
    ]


def run_benchmark(
    *,
    count: int,
    output_dir: Path,
    backend: str,
    model_size: str,
    device: str,
    compute_type: str,
    keep_wavs: bool,
) -> list[BenchmarkRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[BenchmarkRow] = []

    for index, expected in enumerate(build_sentences(count), start=1):
        wav_path = output_dir / f"benchmark-{index:03}.wav"

        tts_start = time.perf_counter()
        generated_path = synthesize_speech(
            text=expected,
            engine="sapi",
            voice=None,
            rate=0,
            volume=100,
        )
        generated_path.replace(wav_path)
        tts_seconds = time.perf_counter() - tts_start

        asr_start = time.perf_counter()
        result = transcribe_audio(
            wav_path,
            backend=backend,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language="zh",
            beam_size=5,
        )
        asr_seconds = time.perf_counter() - asr_start

        cer, distance, expected_chars = character_error_rate(expected, result.text)
        normalized_expected = normalize_text(expected)
        normalized_actual = normalize_text(result.text)
        rows.append(
            BenchmarkRow(
                index=index,
                expected=expected,
                actual=result.text,
                normalized_expected=normalized_expected,
                normalized_actual=normalized_actual,
                cer=cer,
                edit_distance=distance,
                expected_chars=expected_chars,
                exact=normalized_expected == normalized_actual,
                tts_seconds=tts_seconds,
                asr_seconds=asr_seconds,
                wav_path=str(wav_path),
            )
        )

        print(
            f"{index:03}/{count} cer={cer:.3f} "
            f"tts={tts_seconds:.2f}s asr={asr_seconds:.2f}s text={result.text}",
            flush=True,
        )

        if not keep_wavs:
            wav_path.unlink(missing_ok=True)

    return rows


def write_outputs(rows: list[BenchmarkRow], output_dir: Path) -> None:
    csv_path = output_dir / "benchmark-results.csv"
    json_path = output_dir / "benchmark-results.json"
    summary_path = output_dir / "benchmark-summary.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(row) for row in rows], file, ensure_ascii=False, indent=2)

    cers = [row.cer for row in rows]
    tts_times = [row.tts_seconds for row in rows]
    asr_times = [row.asr_seconds for row in rows]
    exact_count = sum(row.exact for row in rows)
    summary = {
        "count": len(rows),
        "exact_count": exact_count,
        "exact_rate": exact_count / len(rows),
        "average_cer": statistics.mean(cers),
        "median_cer": statistics.median(cers),
        "max_cer": max(cers),
        "average_tts_seconds": statistics.mean(tts_times),
        "average_asr_seconds": statistics.mean(asr_times),
        "total_tts_seconds": sum(tts_times),
        "total_asr_seconds": sum(asr_times),
        "worst_10": [
            asdict(row)
            for row in sorted(rows, key=lambda row: row.cer, reverse=True)[:10]
        ],
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "samples" / "benchmark")
    parser.add_argument("--backend", default="whisper")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--keep-wavs", action="store_true")
    args = parser.parse_args()

    rows = run_benchmark(
        count=args.count,
        output_dir=args.output_dir,
        backend=args.backend,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        keep_wavs=args.keep_wavs,
    )
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()

