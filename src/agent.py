import ollama
import subprocess
from intent import classify
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
                "content": """
                    You are a shell command generator.

                    Your task:
                    Convert user requests into EXACTLY ONE shell command.

                    STRICT RULES:
                    - Return ONLY the command.
                    - No explanations.
                    - No markdown.
                    - No code blocks.
                    - No extra text.
                    - No quotes around command.
                    - Assume Windows PowerShell.

                    Examples:
                    User: list files
                    Output:
                    dir

                    User: show current directory
                    Output:
                    pwd

                    User: create a folder named test
                    Output:
                    mkdir test

                    User: run app.py
                    Output:
                    python app.py
                    """
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = ollama.chat(
            model="qwen2.5:3b",
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
            output = self.run_command(shell_command)
            answer = f"Command: {shell_command}\nOutput: {output}"
            print(answer)
            self.messages.append({"role": "assistant", "content": answer})
            return answer
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
        