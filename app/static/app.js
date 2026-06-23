const statusEl = document.querySelector("#status");
const transcribeBtn = document.querySelector("#transcribeBtn");
const speakBtn = document.querySelector("#speakBtn");
const asrResult = document.querySelector("#asrResult");
const audio = document.querySelector("#ttsAudio");
const downloadLink = document.querySelector("#downloadLink");
const asrBackend = document.querySelector("#asrBackend");
const asrModel = document.querySelector("#asrModel");
const asrCompute = document.querySelector("#asrCompute");
const asrDevice = document.querySelector("#asrDevice");

async function loadHealth() {
  const response = await fetch("/health");
  const health = await response.json();
  statusEl.textContent = `ASR ${health.asr_default_backend}/${health.asr_default_model} on ${health.asr_default_device}. CUDA runtime: ${health.cuda_runtime_available ? "ready" : "missing"}.`;
  asrBackend.value = health.asr_default_backend;
  asrModel.value = health.asr_default_model;
  asrDevice.value = health.asr_default_device;
  asrBackend.dispatchEvent(new Event("change"));
}

asrBackend.addEventListener("change", () => {
  if (asrBackend.value === "funasr" && asrModel.value === "small") {
    asrModel.value = "FunAudioLLM/SenseVoiceSmall";
  }
  if (asrBackend.value === "whisper" && asrModel.value === "FunAudioLLM/SenseVoiceSmall") {
    asrModel.value = "small";
  }
  asrCompute.disabled = asrBackend.value === "funasr";
});

transcribeBtn.addEventListener("click", async () => {
  const file = document.querySelector("#audioFile").files[0];
  if (!file) {
    asrResult.textContent = "Please choose an audio file.";
    return;
  }

  transcribeBtn.disabled = true;
  asrResult.textContent = "Transcribing...";

  const params = new URLSearchParams({
    backend: asrBackend.value,
    model_size: asrModel.value,
    device: asrDevice.value,
    compute_type: asrCompute.value,
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

asrBackend.dispatchEvent(new Event("change"));
