import ollama
def classify(text):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content":(
                    """You are an intent classifier.
                    Task:
                    Determine whether the user request is related to
                    a terminal/shell/system command.

                    Rules:
                    Return ONLY one word: TERMINAL or CHAT

                    Examples:
                    "list files"
                    -> TERMINAL
                    "show current directory"
                    -> TERMINAL
                    "what is recursion"
                    -> CHAT
                    "explain transformers"
                    -> CHAT
                """)},
                {
                    "role": "user",
                    "content": text
                }
        ]
    )
    label = response["message"]["content"].strip()
    return label == "TERMINAL"