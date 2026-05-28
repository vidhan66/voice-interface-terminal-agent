import numpy as np
from src.config import load_config

CONFIG = load_config()

SPEECH_RMS_THRESHOLD = CONFIG.get("vad_threshold")   # tune for your mic; lower = more sensitive

def is_speech(frame: np.ndarray, threshold: int = SPEECH_RMS_THRESHOLD) -> bool:
    """Return True if the frame likely contains speech (energy > threshold)."""
    rms = int(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
    return rms > threshold

class SilenceDetector:
    def __init__(
        self,
        sample_rate:int = 16_000,
        silence_cutoff:float = CONFIG.get("silence_cutoff"),
        frame_samples:int = 512,
    ):
        frames_per_second = sample_rate / frame_samples
        self._silent_needed = int(silence_cutoff * frames_per_second)
        self._silent_count = 0
        self._speech_seen = False

    def update(self, frame: np.ndarray) -> bool:
        
        if is_speech(frame):
            self._speech_seen = True
            self._silent_count = 0
        else:
            self._silent_count += 1

        if self._speech_seen and self._silent_count >= self._silent_needed:
            return True

        return False
