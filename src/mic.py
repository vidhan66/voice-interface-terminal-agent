import queue
import threading
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000   
FRAME_SAMPLES = 512    
CHANNELS = 1
DTYPE = "int16"

class MicStream:
    
    def __init__(self, maxbuf: int = 400):
        # 400 × 512 samples = ~12.8 s of audio buffer before overwrite
        self._q: queue.Queue = queue.Queue(maxsize=maxbuf)
        self._stream = None
        self._lock   = threading.Lock()

    def _callback(self, indata, frames, time_info, status):

        chunk = indata[:, 0].copy().astype(np.int16)
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(chunk)

    def start(self):
        self._stream = sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels = CHANNELS,
            dtype = DTYPE,
            blocksize = FRAME_SAMPLES,
            callback = self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def read_frame(self) -> np.ndarray:
        return self._q.get()

    def drain_frames(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def read_seconds(self, seconds: float) -> np.ndarray:
        frames_needed = int(seconds * SAMPLE_RATE / FRAME_SAMPLES)
        chunks = []
        for _ in range(frames_needed):
            chunks.append(self.read_frame())
        return np.concatenate(chunks)
