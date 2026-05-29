from src.mic import MicStream
from src.vad import SilenceDetector
from src.stt import transcribe
from src.agent import Agent
from src.display import Display
import numpy as np

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
    while True:
        input("Press enter to speak...")
        audio = record_until_silence(mic, display)
        #text = "list all files, find hello.js and write binary search code in it and run the file"
        text = transcribe(audio)

        display.user_said(text)
        
        if(text.lower() in ["close", "quit", "stop", "exit"]):
            display.info("Goodbye!")
            break

        display.agent_start()
        for token in agent.ask_stream(text):
            display.token(token)
        display.agent_stop()

if __name__ == "__main__":
    main()