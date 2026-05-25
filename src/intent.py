import ollama
from prompts.loader import load
def classify(text):
    PROMPTS = load("prompts/prompt.yaml")
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content": PROMPTS["classify_intent"]
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )
    label = response["message"]["content"].strip()
    return label == "TERMINAL"