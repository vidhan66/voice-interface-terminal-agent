import threading
from src.listener import Listener
from src.mic import MicStream
from src.vad import SilenceDetector
from src.stt import transcribe
from src.agent import Agent
from src.display import Display
import numpy as np
import time

def record_until_silence(mic: MicStream, display: Display) -> np.ndarray:
    display.info("Listening... speak now")
    
    silence = SilenceDetector()
    chunks  = []

    while True:
        frame = mic.read_frame()
        chunks.append(frame)
        if silence.update(frame):
            break

    display.info("Got it, processing...")
    return np.concatenate(chunks)

def main():
    display = Display()
    display.banner()

    mic = MicStream()
    mic.start()

    agent = Agent()

    prompt_queue = []
    prompt_lock  = threading.Lock()

    listener = Listener(
        mic = mic,
        prompt_lock = prompt_lock,
        prompt_queue = prompt_queue,
        display = display,
    )
    listener_thread = threading.Thread(target=listener.run, daemon=True)
    listener_thread.start()

    display.info("Say 'Hey Voker' to begin")

    while True:
        with prompt_lock:
            if prompt_queue:
                text = prompt_queue.pop(0)
            else:
                text = None

        if text is None:
            time.sleep(0.05)
            continue

        if text.lower().strip() in ["close", "quit", "stop", "exit"]:
            display.info("Goodbye!")
            break

        display.agent_start()
        for token in agent.ask_stream(text):
            display.token(token)
        display.agent_stop()

if __name__ == "__main__":
    main()