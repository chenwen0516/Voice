from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.asr import transcribe_audio
from app.metrics import character_error_rate, normalize_text


def read_manifest(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def pick(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value
    raise KeyError(f"Missing one of columns: {', '.join(names)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "samples" / "dataset-benchmark")
    parser.add_argument("--backend", default="whisper")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-dir", type=Path)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, row in enumerate(rows, start=1):
        audio = Path(pick(row, ("audio", "audio_path", "path", "wav", "file")))
        if args.base_dir and not audio.is_absolute():
            audio = args.base_dir / audio
        expected = pick(row, ("text", "transcript", "sentence", "normalized_text"))

        start = time.perf_counter()
        result = transcribe_audio(
            audio,
            backend=args.backend,
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
            language="zh",
            beam_size=5,
        )
        seconds = time.perf_counter() - start
        cer, distance, expected_chars = character_error_rate(expected, result.text)
        output_rows.append(
            {
                "index": index,
                "audio": str(audio),
                "expected": expected,
                "actual": result.text,
                "normalized_expected": normalize_text(expected),
                "normalized_actual": normalize_text(result.text),
                "cer": cer,
                "edit_distance": distance,
                "expected_chars": expected_chars,
                "exact": normalize_text(expected) == normalize_text(result.text),
                "asr_seconds": seconds,
            }
        )
        print(f"{index:04}/{len(rows)} cer={cer:.3f} asr={seconds:.2f}s", flush=True)

    cers = [row["cer"] for row in output_rows]
    summary = {
        "count": len(output_rows),
        "exact_count": sum(row["exact"] for row in output_rows),
        "exact_rate": sum(row["exact"] for row in output_rows) / len(output_rows),
        "average_cer": statistics.mean(cers),
        "median_cer": statistics.median(cers),
        "max_cer": max(cers),
        "average_asr_seconds": statistics.mean(row["asr_seconds"] for row in output_rows),
        "worst_10": sorted(output_rows, key=lambda row: row["cer"], reverse=True)[:10],
    }

    csv_path = args.output_dir / "dataset-results.csv"
    json_path = args.output_dir / "dataset-results.json"
    summary_path = args.output_dir / "dataset-summary.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    json_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

