from faster_whisper import WhisperModel
import numpy as np

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

def transcribe(audio: np.ndarray, sample_rate: int = 16_000) -> str:

    # Normalise to float32 in [-1, 1]
    audio_f32 = audio.astype(np.float32) / 32768.0

    segments, _ = model.transcribe(
        audio_f32,
        language = "en",
        beam_size = 5,
        vad_filter = True,         
        vad_parameters = dict(
            min_silence_duration_ms = 500,
        ),
    )

    return " ".join(seg.text.strip() for seg in segments).strip()