from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "samples" / "datasets" / "minds14-zh"


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return value or "sample"


def load_minds14_zh():
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Dataset dependencies are missing. Run .\\scripts\\setup_datasets.ps1 first."
        ) from exc

    dataset = load_dataset("PolyAI/minds14", "zh-CN", split="train")
    return dataset.cast_column("audio", Audio(decode=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["minds14-zh"], default="minds14-zh")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-mb", type=float, default=100.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"

    if args.dataset != "minds14-zh":
        raise SystemExit(f"Unsupported dataset: {args.dataset}")

    dataset = load_minds14_zh()
    rows: list[dict[str, str]] = []
    total_bytes = 0
    max_bytes = int(args.max_mb * 1024 * 1024)

    for index, example in enumerate(dataset, start=1):
        if len(rows) >= args.limit:
            break

        audio = example["audio"]
        text = str(example.get("transcription", "")).strip()
        if not text:
            continue

        sample_id = safe_name(str(example.get("path") or f"minds14-zh-{index:04}"))
        wav_path = audio_dir / f"{len(rows) + 1:04}-{sample_id}.wav"
        wav_path.write_bytes(audio["bytes"])
        size = wav_path.stat().st_size

        if rows and total_bytes + size > max_bytes:
            wav_path.unlink(missing_ok=True)
            break

        total_bytes += size
        rows.append(
            {
                "audio": str(wav_path.resolve()),
                "text": text,
                "source": "PolyAI/minds14 zh-CN",
                "id": str(example.get("path") or index),
            }
        )
        print(
            f"{len(rows):04} {size / 1024 / 1024:.2f}MB "
            f"total={total_bytes / 1024 / 1024:.2f}MB text={text}",
            flush=True,
        )

    if not rows:
        raise SystemExit("No rows were exported.")

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["audio", "text", "source", "id"])
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print(f"Exported rows: {len(rows)}")
    print(f"Audio size: {total_bytes / 1024 / 1024:.2f}MB")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
