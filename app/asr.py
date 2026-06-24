from __future__ import annotations

import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

DEFAULT_FUNASR_MODEL = "FunAudioLLM/SenseVoiceSmall"


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


def _prepend_path(directory: Path) -> None:
    import os

    directory_text = str(directory)
    paths = os_environ_path()
    if directory.exists() and directory_text not in paths:
        os.environ["PATH"] = directory_text + ";" + os.getenv("PATH", "")


def _ensure_runtime_paths() -> None:
    _prepend_path(Path(sys.executable).resolve().parent)

    local_app_data = Path.home() / "AppData" / "Local"
    package_root = local_app_data / "Microsoft" / "WinGet" / "Packages"
    if package_root.exists() and not shutil.which("ffmpeg"):
        for ffmpeg in package_root.rglob("ffmpeg.exe"):
            _prepend_path(ffmpeg.parent)
            break


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
    _ensure_runtime_paths()
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "FunASR is not installed. Run .\\scripts\\setup_funasr_py311.ps1 first."
        ) from exc

    if model_name.startswith("FunAudioLLM/"):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for FunAudioLLM models.") from exc
        model_name = snapshot_download(model_name)

    resolved_device = _resolve_device(device)
    funasr_device = "cuda:0" if resolved_device == "cuda" else "cpu"
    kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "disable_update": True,
        "device": funasr_device,
    }
    if not Path(model_name).exists():
        kwargs["vad_model"] = "fsmn-vad"
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    return AutoModel(**kwargs)


def _resolve_funasr_model_name(model_size: str) -> str:
    if not model_size or model_size.lower() in {"small", "sensevoicesmall"}:
        return DEFAULT_FUNASR_MODEL
    return model_size


def _extract_funasr_text(result: Any) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return _clean_funasr_text(str(first.get("text", "")))
    if isinstance(result, dict):
        return _clean_funasr_text(str(result.get("text", "")))
    return _clean_funasr_text(str(result))


def _clean_funasr_text(text: str) -> str:
    import re

    return re.sub(r"<\|[^|]+?\|>", "", text).strip()


def _transcribe_with_funasr(
    audio_path: Path,
    *,
    model_size: str,
    device: str,
    language: str | None,
    use_itn: bool = True,
    merge_vad: bool = True,
    merge_length_s: int = 30,
) -> TranscriptionResult:
    model_name = _resolve_funasr_model_name(model_size)
    model = _load_funasr_model(model_name, device)
    resolved_device = _resolve_device(device)
    result = model.generate(
        input=str(audio_path),
        cache={},
        language=language or "auto",
        use_itn=use_itn,
        batch_size_s=60,
        merge_vad=merge_vad,
        merge_length_s=merge_length_s,
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
    funasr_use_itn: bool = True,
    funasr_merge_vad: bool = True,
    funasr_merge_length_s: int = 30,
) -> TranscriptionResult:
    selected_backend = backend.lower()
    if selected_backend in {"funasr", "sensevoice"}:
        return _transcribe_with_funasr(
            audio_path,
            model_size=model_size,
            device=device,
            language=language,
            use_itn=funasr_use_itn,
            merge_vad=funasr_merge_vad,
            merge_length_s=funasr_merge_length_s,
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
