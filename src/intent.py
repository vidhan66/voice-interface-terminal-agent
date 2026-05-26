import ollama
from prompts.loader import load
def classify(text):
    PROMPTS = load("prompts/prompt.yaml")
    response = ollama.chat(
        model="qwen2.5:7b",
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
    return response["message"]["content"].strip().upper()