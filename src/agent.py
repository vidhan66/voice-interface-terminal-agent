# import wexpect
# import time
# import sys

# class ClaudeAgent:
#     def __init__(self):
#         print("Starting Claude agent...")

#         self.child = wexpect.spawn('ollama launch claude --model qwen2.5:7b', timeout=120)
#         self.child.expect("❯")
#         self.child.logfile = sys.stdout
#         print("Claude ready")

#     def ask(self, prompt):
#         print(f"\nSending: {prompt}")
#         self.child.send(prompt)
#         self.child.send("\r")   # explicit enter

#         self.child.expect("❯", timeout=120)
#         print("\nBEFORE:")
#         print(repr(self.child.before))

#         print("\nAFTER:")
#         print(repr(self.child.after))
            
#         response = self.child.before.strip()
#         return response

#     def close(self):
#         print("Closing Claude agent...")
#         self.child.terminate()
#         print("Claude agent closed.")

# if __name__ == "__main__":
#     agent = ClaudeAgent()

#     response = agent.ask(
#         "2+2 equals ?"
#     )

#     print("\nRESPONSE:")
#     print(response)

#     agent.close()

import ollama
def ask_agent(prompt):

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]