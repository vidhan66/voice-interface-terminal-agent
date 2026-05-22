from audio import record_audio
from stt import transcribe
from agent import ask_agent


def main():

    audio_path = record_audio()

    text = transcribe(audio_path)

    print("\nYou said:")
    print(text)

    response = ask_agent(text)

    print("\nAgent Response:\n")
    print(response)


if __name__ == "__main__":
    main()