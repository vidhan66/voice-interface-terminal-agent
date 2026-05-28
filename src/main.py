from src.audio import record_audio
from src.stt import transcribe
from src.agent import Agent

def main():

    agent = Agent()
    while True:
        #input("Press enter to record audio...")
        #audio_path = record_audio()
        text = "list all files, find hello.js and write binary search code in it and run the file"
        #text = transcribe(audio_path)

        print(f"User: {text}")
        if(text.lower() in ["close", "quit", "stop", "exit"]):
            print("Closing...")
            break
        response = agent.ask(text)

        print("\nAgent Response:\n")
        print(response)


if __name__ == "__main__":
    main()