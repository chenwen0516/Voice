from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "minds14-zh": {
        "output": PROJECT_ROOT / "samples" / "datasets" / "minds14-zh",
        "source": "PolyAI/minds14 zh-CN",
    },
    "fleurs-zh": {
        "output": PROJECT_ROOT / "samples" / "datasets" / "fleurs-zh",
        "source": "google/fleurs cmn_hans_cn",
    },
}


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return value or "sample"


def safe_console(value: str) -> str:
    return value.encode("gbk", errors="replace").decode("gbk")


def load_minds14_zh():
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Dataset dependencies are missing. Run .\\scripts\\setup_datasets.ps1 first."
        ) from exc

    dataset = load_dataset("PolyAI/minds14", "zh-CN", split="train")
    return dataset.cast_column("audio", Audio(decode=False))


def load_fleurs_zh(split: str):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Dataset dependencies are missing. Run .\\scripts\\setup_datasets.ps1 first."
        ) from exc

    dataset = load_dataset(
        "google/fleurs",
        "cmn_hans_cn",
        split=split,
        streaming=True,
    )
    return dataset.cast_column("audio", Audio(decode=False))


def load_dataset_rows(dataset_name: str, split: str):
    if dataset_name == "minds14-zh":
        return load_minds14_zh()
    if dataset_name == "fleurs-zh":
        return load_fleurs_zh(split)
    raise SystemExit(f"Unsupported dataset: {dataset_name}")


def pick_text(dataset_name: str, example: dict) -> str:
    if dataset_name == "minds14-zh":
        return str(example.get("transcription", "")).strip()
    if dataset_name == "fleurs-zh":
        text = str(example.get("transcription") or example.get("raw_transcription") or "")
        return "".join(text.split())
    return ""


def pick_id(dataset_name: str, example: dict, index: int) -> str:
    if dataset_name == "minds14-zh":
        return str(example.get("path") or f"minds14-zh-{index:04}")
    if dataset_name == "fleurs-zh":
        return str(example.get("id") or example.get("path") or f"fleurs-zh-{index:04}")
    return str(index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="minds14-zh")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-mb", type=float, default=100.0)
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = DATASETS[args.dataset]["output"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"

    dataset = load_dataset_rows(args.dataset, args.split)
    rows: list[dict[str, str]] = []
    total_bytes = 0
    max_bytes = int(args.max_mb * 1024 * 1024)

    for index, example in enumerate(dataset, start=1):
        if len(rows) >= args.limit:
            break

        audio = example["audio"]
        text = pick_text(args.dataset, example)
        if not text:
            continue

        sample_id = safe_name(pick_id(args.dataset, example, index))
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
                "source": DATASETS[args.dataset]["source"],
                "id": pick_id(args.dataset, example, index),
            }
        )
        print(
            f"{len(rows):04} {size / 1024 / 1024:.2f}MB "
            f"total={total_bytes / 1024 / 1024:.2f}MB text={safe_console(text)}",
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
