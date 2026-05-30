import threading
from src.listener import Listener
from src.mic import MicStream
from src.vad import SilenceDetector
from src.stt import transcribe
from src.agent import Agent
from src.display import Display
from src.runner import AgentRunner
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

    cancel_event = threading.Event()   
    prompt_queue = []                  
    prompt_lock = threading.Lock()
    agent_busy = threading.Event()

    mic = MicStream()
    mic.start()

    agent = Agent()

    prompt_queue = []
    prompt_lock  = threading.Lock()

    listener = Listener(
        mic = mic,
        cancel_event = cancel_event,
        prompt_lock = prompt_lock,
        prompt_queue = prompt_queue,
        agent_busy = agent_busy,
        display = display,
    )
    runner = AgentRunner(
        cancel_event = cancel_event,
        prompt_lock = prompt_lock,
        prompt_queue = prompt_queue,
        agent_busy = agent_busy,
        display = display,
    )

    listener_thread = threading.Thread(target=listener.run, daemon=True)
    runner_thread = threading.Thread(target=runner.run, daemon=True)
    listener_thread.start()
    runner_thread.start()

    display.info("Say 'Hey Voker' to begin")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()