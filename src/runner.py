import os
import time
import threading
import subprocess
from src.display import Display
from src.agent   import Agent
from src.config import load_config

CONFIG = load_config()
VALID_MODES = {"custom", "aider"}

AGENT_MODE = os.environ.get("AGENT_MODE", CONFIG.get("agent_mode", "custom")).lower()
if AGENT_MODE not in VALID_MODES:
    print(f"Invalid AGENT_MODE '{AGENT_MODE}', falling back to custom")
    AGENT_MODE = "custom"

class AgentRunner:
    POLL = 0.05   # seconds between prompt-queue checks

    def __init__(
        self,
        cancel_event: threading.Event,
        prompt_lock: threading.Lock,
        prompt_queue: list,
        agent_busy: threading.Event,
        display: Display,
    ):
        self._cancel = cancel_event
        self._lock = prompt_lock
        self._queue = prompt_queue
        self._busy = agent_busy
        self._display = display

        if AGENT_MODE == "custom":
            self._backend = CustomBackend(display)

        elif AGENT_MODE == "aider":
            self._backend = AiderBackend(display)

        else:
            self._display.warn(f"Unknown AGENT_MODE '{AGENT_MODE}', falling back to custom")
            self._backend = CustomBackend(display)

        self._display.info(f"Agent mode: {AGENT_MODE}")

    def run(self):
        while True:
            prompt = self.dequeue()
            if prompt is None:
                time.sleep(self.POLL)
                continue

            self._busy.set()
            self._cancel.clear()

            try:
                self._backend.ask(prompt, self._cancel)
            finally:
                self._busy.clear()
                if self._cancel.is_set():
                    self._display.cancelled()

    def dequeue(self) -> str | None:
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
        return None

class CustomBackend:
    """Wraps the existing CustomAgent with streaming + cancellation."""

    def __init__(self, display: Display):
        self._agent   = Agent()
        self._display = display

    def ask(self, prompt: str, cancel: threading.Event):
        self._display.agent_start()

        for token in self._agent.ask_stream(prompt):
            if cancel.is_set():
                self._display.agent_stop()
                return
            self._display.token(token)

        self._display.agent_stop()

class AiderBackend:

    OLLAMA_BASE      = "http://localhost:11434"
    AIDER_MODEL      = "ollama/qwen2.5-coder:7b"
    FIRST_TOKEN_WAIT = 60    # seconds to wait for first output
    POST_OUTPUT_WAIT = 5.0   # seconds of silence after last output = done

    def __init__(self, display: Display):
        self._display = display

    def ask(self, prompt: str, cancel: threading.Event):
        self._display.agent_start()

        env = os.environ.copy()
        env["OLLAMA_API_BASE"] = self.OLLAMA_BASE

        proc = subprocess.Popen(
            [
                "aider",
                "--no-pretty",
                "--yes",
                "--no-show-model-warnings",
                "--no-fancy-input",
                "--no-git",
                "--stream",
                "--model", self.AIDER_MODEL,
                "--message", prompt,
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            stdin  = subprocess.DEVNULL,
            text   = True,
            bufsize= 1,
            env    = env,
        )

        output_lines = []
        last_output  = time.time()
        got_output   = False

        def read():
            nonlocal got_output, last_output
            for line in iter(proc.stdout.readline, ""):
                stripped = line.rstrip()
                if stripped and stripped.strip() != ">":
                    output_lines.append(stripped)
                    got_output  = True
                    last_output = time.time()

        reader = threading.Thread(target=read, daemon=True)
        reader.start()

        deadline = time.time() + self.FIRST_TOKEN_WAIT
        while not got_output:
            if cancel.is_set():
                proc.terminate()
                self._display.agent_stop()
                return
            if time.time() > deadline:
                proc.terminate()
                self._display.warn("Aider timed out")
                self._display.agent_stop()
                return
            time.sleep(0.1)

        while True:
            if cancel.is_set():
                proc.terminate()
                self._display.agent_stop()
                return

            while output_lines:
                self._display.line(output_lines.pop(0))

            if time.time() - last_output > self.POST_OUTPUT_WAIT:
                break

            time.sleep(0.1)

        while output_lines:
            self._display.line(output_lines.pop(0))

        proc.wait()
        self._display.agent_stop()

    def close(self):
        pass