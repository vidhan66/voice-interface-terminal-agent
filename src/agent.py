import ollama
import subprocess
from src.intent import classify
from prompts.loader import load

class Agent:
    def __init__(self):
        self.PROMPTS = load("prompts/prompt.yaml")
        self.messages = [
            {
                "role": "system",
                "content": self.PROMPTS["system_prompt"]
            }
        ]

    def run_command(self,command):
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )

            output = result.stdout or ""

            if result.stderr:
                output += ("\n" + result.stderr).strip()

            return output.strip()

        except Exception as e:
            return str(e)
    
    def get_shell_command(self, prompt):
        messages = [
            {
                "role": "system",
                "content": self.PROMPTS["shell_command"]
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = ollama.chat(
            model="qwen2.5-coder:7b",
            messages=messages
        )

        command = response["message"]["content"].strip()

        # cleanup
        command = command.replace("```powershell", "")
        command = command.replace("```bash", "")
        command = command.replace("```", "")
        command = command.strip()

        # keep only first line
        command = command.split("\n")[0]

        return command

    def ask(self,prompt):
        self.messages.append({
            "role": "user",
            "content": prompt
        })
        intent = classify(prompt)
        print(f"intent: {intent}")
        if(intent == True):
            shell_command = self.get_shell_command(prompt)
            print(f"Executing command: {shell_command}")
            confirm = input("Do you want to execute this command? (y/n): ")
            if(confirm.lower()=='y'):
                output = self.run_command(shell_command)
                answer = f"Command: {shell_command}\nOutput: {output}"
                print(answer)
                self.messages.append({"role": "assistant", "content": answer})
                return answer
            else:
                self.messages.append({
                    "role": "assistant",
                    "content": self.PROMPTS["cancelled_command"] 
                })
                return "Command execution cancelled by user."
        else :
            response = ollama.chat(
                model="qwen2.5:3b",
                messages=self.messages
            )
            print(response)
            ans =  response["message"]["content"]   
            self.messages.append({
                "role": "assistant",
                "content": ans
            })
            return ans
        