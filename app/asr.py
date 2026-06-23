from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    text: str
    backend: str = "whisper"
    model: str | None = None
    device: str | None = None
    language: str | None
    language_probability: float | None
    duration: float | None
    segments: list[Segment]


def _has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False

    return result.returncode == 0 and bool(result.stdout.strip())


def _find_runtime_dll(name: str) -> str | None:
    for directory in [
        *[Path(path) for path in os_environ_path()],
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin"),
    ]:
        candidate = directory / name
        if candidate.exists():
            return str(candidate)
    return None


def os_environ_path() -> list[str]:
    import os

    return [path for path in os.getenv("PATH", "").split(";") if path]


def has_cuda_runtime() -> bool:
    return bool(_find_runtime_dll("cublas64_12.dll"))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if _has_nvidia_gpu() and has_cuda_runtime() else "cpu"


@lru_cache(maxsize=4)
def _load_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    resolved_device = _resolve_device(device)
    try:
        return WhisperModel(model_size, device=resolved_device, compute_type=compute_type)
    except Exception:
        if device != "auto" or resolved_device != "cuda":
            raise

        return WhisperModel(model_size, device="cpu", compute_type=_cpu_compute_type(compute_type))


def _cpu_compute_type(compute_type: str) -> str:
    if "float16" in compute_type:
        return "int8"
    return compute_type


def _is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda" in message or "cublas" in message or "cudnn" in message


def _run_transcription(
    model,
    audio_path: Path,
    *,
    language: str | None,
    beam_size: int,
):
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
    )
    segments = [
        Segment(start=segment.start, end=segment.end, text=segment.text.strip())
        for segment in segments_iter
    ]
    return segments, info


@lru_cache(maxsize=2)
def _load_funasr_model(model_name: str, device: str):
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "FunASR is not installed. Run .\\scripts\\setup_funasr.ps1 first."
        ) from exc

    resolved_device = _resolve_device(device)
    funasr_device = "cuda:0" if resolved_device == "cuda" else "cpu"
    return AutoModel(
        model=model_name,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=funasr_device,
    )


def _extract_funasr_text(result: Any) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("text", "")).strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def _transcribe_with_funasr(
    audio_path: Path,
    *,
    model_size: str,
    device: str,
    language: str | None,
) -> TranscriptionResult:
    model_name = model_size if model_size else "iic/SenseVoiceSmall"
    model = _load_funasr_model(model_name, device)
    resolved_device = _resolve_device(device)
    result = model.generate(
        input=str(audio_path),
        cache={},
        language=language or "auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    text = _extract_funasr_text(result)
    return TranscriptionResult(
        text=text,
        backend="funasr",
        model=model_name,
        device=resolved_device,
        language=language,
        language_probability=None,
        duration=None,
        segments=[Segment(start=0.0, end=0.0, text=text)] if text else [],
    )


def transcribe_audio(
    audio_path: Path,
    *,
    backend: str = "whisper",
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
    beam_size: int,
) -> TranscriptionResult:
    selected_backend = backend.lower()
    if selected_backend in {"funasr", "sensevoice"}:
        return _transcribe_with_funasr(
            audio_path,
            model_size=model_size or "iic/SenseVoiceSmall",
            device=device,
            language=language,
        )
    if selected_backend not in {"whisper", "faster-whisper"}:
        raise ValueError(f"Unsupported ASR backend: {backend}")

    model = _load_model(model_size, device, compute_type)
    resolved_device = _resolve_device(device)
    try:
        segments, info = _run_transcription(
            model,
            audio_path,
            language=language,
            beam_size=beam_size,
        )
    except Exception as exc:
        if device != "auto" or not _is_cuda_runtime_error(exc):
            raise

        model = _load_model(model_size, "cpu", _cpu_compute_type(compute_type))
        resolved_device = "cpu"
        segments, info = _run_transcription(
            model,
            audio_path,
            language=language,
            beam_size=beam_size,
        )

    return TranscriptionResult(
        text="".join(segment.text for segment in segments).strip(),
        backend="whisper",
        model=model_size,
        device=resolved_device,
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        duration=getattr(info, "duration", None),
        segments=segments,
    )
