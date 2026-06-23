const statusEl = document.querySelector("#status");
const transcribeBtn = document.querySelector("#transcribeBtn");
const speakBtn = document.querySelector("#speakBtn");
const asrResult = document.querySelector("#asrResult");
const audio = document.querySelector("#ttsAudio");
const downloadLink = document.querySelector("#downloadLink");

async function loadHealth() {
  const response = await fetch("/health");
  const health = await response.json();
  statusEl.textContent = `ASR ${health.asr_default_backend}/${health.asr_default_model} on ${health.asr_default_device}. CUDA runtime: ${health.cuda_runtime_available ? "ready" : "missing"}.`;
}

transcribeBtn.addEventListener("click", async () => {
  const file = document.querySelector("#audioFile").files[0];
  if (!file) {
    asrResult.textContent = "Please choose an audio file.";
    return;
  }

  transcribeBtn.disabled = true;
  asrResult.textContent = "Transcribing...";

  const params = new URLSearchParams({
    backend: document.querySelector("#asrBackend").value,
    model_size: document.querySelector("#asrModel").value,
    device: document.querySelector("#asrDevice").value,
    compute_type: document.querySelector("#asrCompute").value,
    language: "zh",
  });
  const form = new FormData();
  form.append("file", file);

  try {
    const response = await fetch(`/asr?${params.toString()}`, {
      method: "POST",
      body: form,
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "ASR request failed.");
    }
    asrResult.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    asrResult.textContent = error.message;
  } finally {
    transcribeBtn.disabled = false;
  }
});

speakBtn.addEventListener("click", async () => {
  speakBtn.disabled = true;
  downloadLink.style.display = "none";

  const body = {
    text: document.querySelector("#ttsText").value,
    engine: document.querySelector("#ttsEngine").value,
    rate: Number(document.querySelector("#ttsRate").value),
    volume: Number(document.querySelector("#ttsVolume").value),
  };

  try {
    const response = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.detail || "TTS request failed.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    audio.src = url;
    downloadLink.href = url;
    downloadLink.style.display = "inline-block";
  } catch (error) {
    alert(error.message);
  } finally {
    speakBtn.disabled = false;
  }
});

loadHealth().catch(() => {
  statusEl.textContent = "Service health check failed.";
});

