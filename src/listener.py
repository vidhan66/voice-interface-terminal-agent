import time
import threading
import numpy as np
from src.mic import MicStream, SAMPLE_RATE, FRAME_SAMPLES
from src.stt import transcribe
from src.display import Display
from src.vad import is_speech, SilenceDetector
from vosk import Model, KaldiRecognizer
import json, os

def load_wake_word_detector():
    try:
        return VoskKWSDetector()
    except Exception:
        pass
    return EnergyKeywordFallback()

class VoskKWSDetector:
    """
    Vosk keyword spotting — fully offline, no API key needed.
    Requires:  pip install vosk
    Model:     download from https://alphacephei.com/vosk/models
               e.g. vosk-model-small-en-us-0.15, place in ./models/vosk/
    """

    KEYWORDS = ["hey voker"]

    def __init__(self):
        model_path = os.environ.get("VOSK_MODEL_PATH", "models/vosk")
        self._model = Model(model_path)

        grammar = json.dumps(self.KEYWORDS + ["[unk]"])
        self._rec = KaldiRecognizer(self._model, SAMPLE_RATE, grammar)

    def process(self, frame: np.ndarray) -> bool:
        data = frame.tobytes()
        if self._rec.AcceptWaveform(data):
            result = json.loads(self._rec.Result())
            text = result.get("text", "")
            return any(kw in text for kw in self.KEYWORDS)
        return False

class EnergyKeywordFallback:

    THRESHOLD = 2000  # RMS amplitude

    def process(self, frame: np.ndarray) -> bool:
        rms = int(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        return rms > self.THRESHOLD

class Listener:

    SILENCE_CUTOFF = 1.8
    MAX_UTTERANCE = 30.0
    MIN_UTTERANCE = 0.4

    def __init__(
        self,
        mic: MicStream,
        prompt_lock: threading.Lock,
        cancel_event: threading.Event,
        agent_busy: threading.Event,
        prompt_queue: list,
        display: Display,
    ):
        self._mic = mic
        self._lock = prompt_lock
        self._cancel = cancel_event
        self._busy = agent_busy
        self._queue = prompt_queue
        self._display = display
        self._detector = load_wake_word_detector()

    def run(self):
        self._display.info("Listener ready — say 'Hey Voker' to begin")
        while True:
            if self._busy.is_set():
                self.watch_for_interrupt()
            else:
                self.watch_for_wake_word()

    def watch_for_wake_word(self):
        frame = self._mic.read_frame()
        triggered = self._detector.process(frame)
        if triggered:
            self._display.wake()
            audio = self.capture_utterance()
            if audio is not None:
                text = transcribe(audio)
                if text.strip():
                    self._display.user_said(text)
                    self.enqueue(text)

    def watch_for_interrupt(self):

        frame = self._mic.read_frame()
        if is_speech(frame):

            self._display.interrupt()
            self._cancel.set()           
            audio = self.capture_utterance(pre_frame=frame)
            self._cancel.clear()
            if audio is not None:
                text = transcribe(audio)
                if text.strip():
                    self._display.user_said(text)
                    self.enqueue(text)

    def capture_utterance(self, pre_frame: np.ndarray | None = None) -> np.ndarray | None:

        self._mic.drain_frames()  

        silence = SilenceDetector(
            sample_rate = SAMPLE_RATE,
            silence_cutoff = self.SILENCE_CUTOFF,
            frame_samples = FRAME_SAMPLES,
        )

        chunks = []
        if pre_frame is not None:
            chunks.append(pre_frame)
            silence.update(pre_frame)

        max_frames = int(self.MAX_UTTERANCE * SAMPLE_RATE / FRAME_SAMPLES)

        for _ in range(max_frames):
            frame = self._mic.read_frame()
            chunks.append(frame)
            if silence.update(frame):
                break   

        audio = np.concatenate(chunks)

        if len(audio) / SAMPLE_RATE < self.MIN_UTTERANCE:
            return None

        return audio

    def enqueue(self, text: str):
        with self._lock:
            self._queue.append(text)
