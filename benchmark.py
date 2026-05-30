import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from src.config import load_config
from src.vad import SilenceDetector, is_speech
from src.mic import MicStream
from src.stt import transcribe
from src.agent import Agent, detect_model
from src.intent import classify

SAMPLE_RATE   = 16_000
FRAME_SAMPLES = 512
CONFIG = load_config()

# ── 1. STT latency + accuracy ─────────────────────────────────────────────────
def benchmark_stt(runs: int = 5):
    print("\n=== STT Benchmark ===\n")
    print(f"Speak a phrase {runs} times when prompted.\n")

    model = WhisperModel("small", device="cpu", compute_type="int8")
    latencies = []

    for i in range(runs):
        input(f"Run {i+1}/{runs} — press enter then speak...")
        audio = sd.rec(
            int(5 * SAMPLE_RATE),
            samplerate = SAMPLE_RATE,
            channels = 1,
            dtype = "int16",
        )
        sd.wait()
        audio = audio[:, 0]

        start = time.perf_counter()
        result = transcribe(audio)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        print(f" transcript : {result}")
        print(f" latency : {elapsed:.3f}s")

    print(f"\n  avg latency : {np.mean(latencies):.3f}s")
    print(f"  min : {np.min(latencies):.3f}s")
    print(f"  max : {np.max(latencies):.3f}s")


# ── 2. Intent classification latency ─────────────────────────────────────────
def benchmark_intent(runs: int = 10):
    print("\n=== Intent Classification Benchmark ===")

    test_prompts = [
        "list all files in this folder",
        "what does the agent file do",
        "how are you doing today",
        "run the main script",
        "explain the architecture of this project",
        "tell me a joke",
        "fix the bug in tools.py",
        "create a new folder called test",
        "what is the purpose of vad.py",
        "show current directory",
    ]

    latencies = []
    for prompt in test_prompts:
        start  = time.perf_counter()
        intent = classify(prompt)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        print(f" [{intent:<10}] {elapsed:.3f}s  —  {prompt}")

    print(f"\n avg latency : {np.mean(latencies):.3f}s")
    print(f" min : {np.min(latencies):.3f}s")
    print(f" max : {np.max(latencies):.3f}s")


# ── 3. Time to first token ────────────────────────────────────────────────────
def benchmark_ttft(runs: int = 3):
    print("\n=== Time to First Token Benchmark ===")

    test_prompts = [
        "what is 2 plus 2",
        "write a hello world in python",
        "list files in current directory",
    ]

    agent = Agent()

    for prompt in test_prompts:
        start = time.perf_counter()
        first_token = None

        for token in agent.ask_stream(prompt):
            if first_token is None:
                first_token = time.perf_counter() - start
            break   

        print(f"  TTFT: {first_token:.3f}s  —  {prompt}")

# ── 4. VAD silence cutoff accuracy ───────────────────────────────────────────
def benchmark_vad(runs: int = 5):
    print("\n=== VAD Silence Cutoff Benchmark ===")
    print("Speak then stop — measures how long after silence it takes to cut off.\n")

    mic = MicStream()
    mic.start()

    configured = CONFIG.get("silence_cutoff", 1.8)
    print(f" configured silence_cutoff: {configured}s")

    cutoffs = []

    for i in range(runs):
        input(f"Run {i+1}/{runs} — press enter, speak, then stop...")
        silence = SilenceDetector()
        chunks = []
        last_speech_t = None

        while True:
            frame = mic.read_frame()
            chunks.append(frame)
            if is_speech(frame):
                last_speech_t = time.perf_counter()
            if silence.update(frame):
                cutoff_t = time.perf_counter()
                break

        actual = cutoff_t - last_speech_t if last_speech_t else 0
        cutoffs.append(actual)
        print(f"  silence cutoff after: {actual:.3f}s (configured: {configured}s)")

    mic.stop()
    print(f"\n avg actual cutoff : {np.mean(cutoffs):.3f}s")
    print(f" configured : {configured}s")
    print(f" drift : {np.mean(cutoffs) - configured:+.3f}s")


# ── 5. End to end latency ─────────────────────────────────────────────────────
def benchmark_e2e(runs: int = 3):
    print("\n=== End to End Latency Benchmark ===")
    print("Measures: STT + intent + time to first token\n")

    mic = MicStream()
    mic.start()

    agent = Agent()

    for i in range(runs):
        input(f"Run {i+1}/{runs} — press enter then speak your prompt...")

        # record
        silence = SilenceDetector()
        chunks = []
        while True:
            frame = mic.read_frame()
            chunks.append(frame)
            if silence.update(frame):
                break

        audio = np.concatenate(chunks)

        # STT
        t0 = time.perf_counter()
        text = transcribe(audio)
        t_stt = time.perf_counter() - t0
        print(f" transcript : {text}")
        print(f" stt : {t_stt:.3f}s")

        # intent
        t1 = time.perf_counter()
        intent = classify(text)
        t_intent = time.perf_counter() - t1
        print(f" intent : {intent} ({t_intent:.3f}s)")

        # TTFT
        t2 = time.perf_counter()
        for token in agent.ask_stream(text):
            t_ttft = time.perf_counter() - t2
            print(f" ttft : {t_ttft:.3f}s")
            break

        total = t_stt + t_intent + t_ttft
        print(f" total e2e : {total:.3f}s\n")

    mic.stop()


# ── runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Voker — Pipeline Benchmark")
    print("==========================")
    print("Backend:", detect_model())

    benchmark_intent()
    benchmark_ttft()
    benchmark_stt()
    benchmark_vad()
    benchmark_e2e()