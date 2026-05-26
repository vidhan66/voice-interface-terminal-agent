import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(
    filename="temp.wav",
    duration=10,
    sample_rate=16000
):

    print("Recording... Speak now")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(filename, sample_rate, audio)

    print("Recording complete")

    return filename