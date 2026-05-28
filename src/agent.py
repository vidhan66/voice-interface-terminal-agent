import ollama
import subprocess
from src.intent import classify
from prompts.loader import load
from src.tools import list_files, read_file
import os
from src.config import load_config

CONFIG = load_config()

def detect_model(task: str = "chat") -> tuple[str, str]:

    backend = (
        os.environ.get("LLM_BACKEND")        
        or CONFIG.get("llm_backend")        
        or "auto"                            
    ).lower()

    if backend == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            backend = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            backend = "openai"
        elif os.environ.get("GEMINI_API_KEY"):
            backend = "gemini"
        else:
            backend = "ollama"
            
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    if os.environ.get("OPENAI_API_KEY"):
        return "openai", os.environ.get("OPENAI_MODEL", "gpt-4o")

    if os.environ.get("GEMINI_API_KEY"):
        return "gemini", os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    if task == "code":
        return "ollama", os.environ.get(
                    "OLLAMA_CODE_MODEL",
                    CONFIG.get("ollama_code_model", "qwen2.5-coder:7b")
                )
    
    return "ollama", os.environ.get(
                "OLLAMA_CHAT_MODEL",
                CONFIG.get("ollama_chat_model", "qwen2.5:7b")
            )

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
        if(intent == "TERMINAL"):
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
        
        elif (intent == "REPO"):

            ans = self.repo_aware(prompt)
            self.messages.append({
                "role": "assistant",
                "content": ans
            })
            return ans

        else :
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
        
    def repo_aware(self, query):
        messages = [
            {
                "role": "system",
                "content": self.PROMPTS["analyse_repo"]
            },
            {
                "role": "user",
                "content": query
            }
        ]

        for _ in range(5):
            response = ollama.chat(
                model="qwen2.5-coder:7b",
                messages=messages
            )
            reply = response["message"]["content"]

            print("LLM:", reply)
            # tool: list_files
            if "TOOL:list_files" in reply:
                files = list_files()
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": "\n".join(files)
                    }
                )
            # tool: read_file
            elif "TOOL:read_file" in reply:
                file_path = (
                    reply
                    .split("TOOL:read_file(")[1]
                    .split(")")[0]
                    .strip()
                )
                content = read_file(file_path)
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content":
                        f"FILE: {file_path}\n\n{content}"
                    }
                )
            else:
                return reply

        return "Could not analyze repository."
                