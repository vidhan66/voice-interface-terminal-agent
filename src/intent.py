import ollama
from prompts.loader import load
from src.config import load_config

CONFIG = load_config()

def classify(text):
    PROMPTS = load("prompts/prompt.yaml")
    response = ollama.chat(
        model=CONFIG["ollama_chat_model"],
        messages=[
            {"role": "system", "content": PROMPTS["classify_intent"]},
            {"role": "user", "content": text}
        ]
    )
    return response["message"]["content"].strip().upper()