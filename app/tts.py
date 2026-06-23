from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAPI_SCRIPT = PROJECT_ROOT / "scripts" / "sapi_tts.ps1"


def synthesize_speech(
    *,
    text: str,
    engine: str | None,
    voice: str | None,
    rate: int,
    volume: int,
) -> Path:
    selected_engine = (engine or os.getenv("TTS_ENGINE", "sapi")).lower()
    if selected_engine == "piper":
        return _synthesize_with_piper(text)
    if selected_engine == "sapi":
        return _synthesize_with_sapi(text=text, voice=voice, rate=rate, volume=volume)

    raise ValueError(f"Unsupported TTS engine: {selected_engine}")


def _synthesize_with_sapi(*, text: str, voice: str | None, rate: int, volume: int) -> Path:
    output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    text_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name)
    text_path.write_text(text, encoding="utf-8")

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SAPI_SCRIPT),
        "-TextFile",
        str(text_path),
        "-OutputFile",
        str(output_path),
        "-Rate",
        str(rate),
        "-Volume",
        str(volume),
    ]

    if voice:
        command.extend(["-Voice", voice])

    try:
        _run(command)
        return output_path
    finally:
        text_path.unlink(missing_ok=True)


def _synthesize_with_piper(text: str) -> Path:
    piper_exe = os.getenv("PIPER_EXE")
    piper_model = os.getenv("PIPER_MODEL")
    piper_config = os.getenv("PIPER_CONFIG")

    if not piper_exe or not piper_model:
        raise ValueError("PIPER_EXE and PIPER_MODEL must be set when TTS_ENGINE=piper")

    output_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    command = [piper_exe, "--model", piper_model, "--output_file", str(output_path)]
    if piper_config:
        command.extend(["--config", piper_config])

    _run(command, input_text=text)
    return output_path


def _run(command: list[str], input_text: str | None = None) -> None:
    result = subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "Command failed"
        raise RuntimeError(message)

