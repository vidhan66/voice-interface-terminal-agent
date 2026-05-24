from click import prompt
import ollama

class Agent:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": """You are a terminal coding assistant. Be concise.
                    Focus on coding, debugging, shell commands,system design, and 
                    engineering help.
                """
            }
        ]
        
    def ask(self,prompt):
        self.messages.append({
            "role": "user",
            "content": prompt
        })

        response = ollama.chat(
            model="qwen2.5:7b",
            messages=self.messages
        )
        print(response)
        ans =  response["message"]["content"]   
        self.messages.append({
            "role": "assistant",
            "content": ans
        })
        return ans
