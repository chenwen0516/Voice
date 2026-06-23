from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from app.asr import TranscriptionResult, has_cuda_runtime, transcribe_audio
from app.tts import synthesize_speech


app = FastAPI(title="Local Voice Service", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _delete_file(path: Path) -> None:
    path.unlink(missing_ok=True)


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    engine: str | None = Field(default=None, description="sapi or piper")
    voice: str | None = Field(default=None, description="Optional SAPI voice name")
    rate: int = Field(default=0, ge=-10, le=10)
    volume: int = Field(default=100, ge=0, le=100)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.head("/", include_in_schema=False)
def index_head() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "tts_engine": os.getenv("TTS_ENGINE", "sapi"),
        "asr_default_backend": os.getenv("ASR_BACKEND", "whisper"),
        "asr_default_model": os.getenv("ASR_MODEL", "small"),
        "asr_default_device": os.getenv("ASR_DEVICE", "cpu"),
        "cuda_runtime_available": has_cuda_runtime(),
    }


@app.post("/asr", response_model=TranscriptionResult)
async def asr(
    file: Annotated[UploadFile, File(...)],
    backend: str = Query(default_factory=lambda: os.getenv("ASR_BACKEND", "whisper")),
    model_size: str = Query(default_factory=lambda: os.getenv("ASR_MODEL", "small")),
    device: str = Query(default_factory=lambda: os.getenv("ASR_DEVICE", "auto")),
    compute_type: str = Query(default_factory=lambda: os.getenv("ASR_COMPUTE_TYPE", "int8_float16")),
    language: str | None = Query(default=None),
    beam_size: int = Query(default=5, ge=1, le=10),
) -> TranscriptionResult:
    suffix = Path(file.filename or "audio").suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_path = Path(temp_audio.name)
        while chunk := await file.read(1024 * 1024):
            temp_audio.write(chunk)

    try:
        return transcribe_audio(
            temp_path,
            backend=backend,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/tts")
def tts(request: TtsRequest) -> FileResponse:
    try:
        output_path = synthesize_speech(
            text=request.text,
            engine=request.engine,
            voice=request.voice,
            rate=request.rate,
            volume=request.volume,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="speech.wav",
        background=BackgroundTask(_delete_file, output_path),
    )
