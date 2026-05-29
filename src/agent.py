import ollama
import subprocess
from src.intent import classify
from typing import Generator, Callable
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
            
    if backend == "anthropic":
        return "anthropic", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    if backend == "openai":
        return "openai", os.environ.get("OPENAI_MODEL", "gpt-4o")

    if backend == "gemini":
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
    def __init__(self, confirm_callback: Callable[[str], bool] | None = None):
        self.PROMPTS = load("prompts/prompt.yaml")
        self.messages = [
            {"role": "system","content": self.PROMPTS["system_prompt"]}
        ]
        self._confirm = confirm_callback or (lambda cmd: True)

    def chat_once(self, messages: list, task: str = "chat") -> str:
        """
        Non-streaming — returns the full reply as a string.
        Used in: repo tool-call loop, _get_shell_command
        """
        backend, model = detect_model(task)

        if backend == "anthropic":
            from anthropic import Anthropic
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            msgs   = [m for m in messages if m["role"] != "system"]
            resp   = Anthropic().messages.create(
                model      = model,
                max_tokens = 8096,
                system     = system,
                messages   = msgs,
            )
            return resp.content[0].text

        if backend == "openai":
            from openai import OpenAI
            resp = OpenAI().chat.completions.create(
                model    = model,
                messages = messages,
            )
            return resp.choices[0].message.content
        
        if backend == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in messages if m["role"] != "system"
            )
            return genai.GenerativeModel(model).generate_content(prompt).text

        # ollama
        resp = ollama.chat(model=model, messages=messages)
        return resp["message"]["content"]
                
    def stream_chat(
        self, messages:list, track_reply:bool = False, task:str  = "chat",
    ) -> Generator[str, None, None]:

        backend, model = detect_model(task)
        full_reply = []

        if backend == "anthropic":
            from anthropic import Anthropic
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            msgs   = [m for m in messages if m["role"] != "system"]
            with Anthropic().messages.stream(
                model      = model,
                max_tokens = 8096,
                system     = system,
                messages   = msgs,
            ) as stream:
                for token in stream.text_stream:
                    full_reply.append(token)
                    yield token

        elif backend == "openai":
            from openai import OpenAI
            for chunk in OpenAI().chat.completions.create(
                model    = model,
                messages = messages,
                stream   = True,
            ):
                token = chunk.choices[0].delta.content or ""
                full_reply.append(token)
                yield token

        elif backend == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in messages if m["role"] != "system"
            )
            for chunk in genai.GenerativeModel(model).generate_content(prompt, stream=True):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

        else:  # ollama
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                token = chunk["message"]["content"]
                full_reply.append(token)
                yield token

        if track_reply:
            self.messages.append({
                "role":    "assistant",
                "content": "".join(full_reply),
            })

    def run_command(self,command):
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True
            )
            output = result.stdout or ""
            if result.stderr:
                output += ("\n" + result.stderr).strip()
            return output
        except Exception as e:
            return str(e)
    
    def get_shell_command(self, prompt):
        messages = [
            {"role": "system", "content": self.PROMPTS["shell_command"]},
            {"role": "user", "content": prompt}
        ]
        command = self.chat_once(messages, task="code")
        for tag in ["```powershell", "```bash", "```"]:
            command = command.replace(tag, "")
        return command.strip().split("\n")[0]

    def handle_terminal(self, prompt: str) -> Generator[str, None, None]:
        shell_command = self.get_shell_command(prompt)
        yield f"\nExecuting command: {shell_command}\n"
        confirmed = self._confirm(shell_command)
        if confirmed:
            output = self.run_command(shell_command)
            answer = f"[Output]\n{output}"
            self.messages.append({"role": "assistant", "content": answer})
            yield answer
        else:
            msg = self.PROMPTS["cancelled_command"]
            self.messages.append({"role": "assistant", "content": msg})
            yield msg
    
    def repo_aware(self, prompt: str) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": self.PROMPTS["analyse_repo"]},
            {"role": "user", "content": prompt}
        ]
        for _ in range(5):
            reply = self.chat_once(messages, task="code")
            print("LLM:", reply)
            # tool: list_files
            if "TOOL:list_files" in reply:
                files = list_files()
                messages.append(
                    {"role": "assistant", "content": reply}
                )
                messages.append(
                    {"role": "tool", "content": "\n".join(files)}
                )
                yield f"[tool] listing files…\n"

            # tool: read_file
            elif "TOOL:read_file" in reply:
                file_path = reply.split("TOOL:read_file(")[1].split(")")[0].strip()
                content   = read_file(file_path)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "tool", "content": f"FILE: {file_path}\n\n{content}"})
                yield f"[tool] reading {file_path}…\n"

            else:
                # Stream the final answer token by token
                for token in self.stream_chat(messages, task="code"):
                    yield token
                self.messages.append({"role": "assistant", "content": reply})
                return
            
        yield "Could not analyze repository."
            
    def handle_chat(self) -> Generator[str, None, None]:
        for token in self.stream_chat(self.messages, track_reply=True, task="chat"):
            yield token

    def ask_stream(self,prompt):
        self.messages.append({"role": "user","content": prompt})
        intent = classify(prompt)
        print(f"intent: {intent}")
        if(intent == "TERMINAL"):
            yield from self.handle_terminal(prompt)
        
        elif (intent == "REPO"):
            yield from self.repo_aware(prompt)

        else :
            yield from self.handle_chat()
