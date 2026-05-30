# Voker — Hands-Free Terminal Coding Assistant

Say **"Hey Voker"** and start talking. No keypresses needed.

---

## What it does

- **Always-on wake word** — "Hey Voker" triggers listening (powered by Vosk, runs offline)
- **VAD-gated capture** — records until you stop talking; no fixed duration
- **Interruptible generation** — speak while the agent is responding and it stops immediately, transcribes your new command, and continues from there
- **Dual mode** — use your own ollama-based agent or plug in Aider
- **Streaming output** — responses appear word-by-word in the terminal

---

## Architecture

```
┌──────────────┐   audio   ┌──────────────────┐   prompt   ┌──────────────┐
│  MicThread   │ ────────► │  ListenerThread  │ ─────────► │ AgentThread  │
│ ring buffer  │           │ wake + VAD + STT │            │  streaming   │
└──────────────┘           └──────────────────┘            └──────────────┘
                                    │                              │
                                    │  cancel_event (interrupt)    │
                                    └──────────────────────────────┘
```

Three OS threads share a single `threading.Event` (`cancel_event`).
When you speak during generation, ListenerThread sets the event,
AgentThread stops streaming, and the new utterance replaces the old prompt.

## Tech Stack

| Component | Technology |
|---|---|
| Wake word detection | [Vosk](https://alphacephei.com/vosk/) — offline, Apache 2.0 |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local Whisper, MIT |
| Voice activity detection | Energy-based RMS threshold (custom, zero deps) |
| LLM — local | [Ollama](https://ollama.com) + qwen2.5:7b / qwen2.5-coder:7b |
| LLM — cloud | Anthropic Claude / OpenAI GPT / Google Gemini (optional, via API key) |
| External agent | [Aider](https://aider.chat) — Apache 2.0 |
| Audio capture | [sounddevice](https://python-sounddevice.readthedocs.io) |
| Language | Python 3.12 |

## Demo Video
https://github.com/user-attachments/assets/16f8a14d-d009-4570-9f41-15421445a992
## LLM Backend

Runs fully offline by default using Ollama. Export any one API key to
switch to a cloud model — no code changes needed.

| Provider  | Env var           | Default model              | Task routing        |
|-----------|-------------------|----------------------------|---------------------|
| Anthropic | ANTHROPIC_API_KEY | claude-sonnet-4-20250514   | all tasks           |
| OpenAI    | OPENAI_API_KEY    | gpt-4o                     | all tasks           |
| Gemini    | GEMINI_API_KEY    | gemini-2.0-flash           | all tasks           |
| Ollama    | (none needed)     | qwen2.5:7b + qwen2.5-coder | chat vs code tasks  |

Priority order: Anthropic → OpenAI → Gemini → Ollama

When using Ollama, the agent automatically picks the right model per task:
- General conversation and questions → qwen2.5:7b
- Code, shell commands, repo analysis → qwen2.5-coder:7b

Cloud providers use one model for everything since they handle both equally well.

## Known Limitations

**Aider integration:** Aider receives the prompt and processes it correctly
but the response is written to a chat history file instead of appearing in
the terminal. The custom agent mode (`agent_mode: custom`) is the recommended mode and works fully.

**Wake word accuracy.** Vosk KWS matches phonetically against a small grammar, so accents and background noise can cause false negatives. The wake word "Hey Voker" may need to be repeated in noisy environments.

**Single language.** The STT model is configured for English only. Non-English prompts will be transcribed inaccurately.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama + models

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

### 3. Wake word setup 

**Option A — Vosk (no API key, fully offline)**

1. Download a model: https://alphacephei.com/vosk/models
   Recommended: `vosk-model-small-en-us-0.4`
2. Place it at `./models/vosk/`
3. In `config.yaml`, set `wake_word_backend: vosk`

**Option B — Energy fallback (no setup, least accurate)**

In `config.yaml`, set `wake_word_backend: energy`.
Triggers on any loud sound. Useful for testing the pipeline.

### 4. Agent mode (optional)

Edit `config.yaml` or set the env var:

```bash
# Use built-in ollama agent (default)
export AGENT_MODE=custom

# Use Aider
export AGENT_MODE=aider

```

---

## Running

```bash
python -m src.main
```

---

## Project structure

```
voice_agent/    
├── config.yaml          ← all tunable settings
├── requirements.txt
├── src/
│   ├── mic.py           ← always-on ring buffer mic stream
│   ├── listener.py      ← wake word + VAD + STT + interrupt detection
│   ├── vad.py           ← voice activity detection helpers
│   ├── stt.py           ← faster-whisper transcription
│   ├── runner.py        ← agent thread, cancellable streaming
│   ├── agent.py         ← custom ollama agent (your original, upgraded)
│   ├── display.py       ← all terminal output (ANSI colours)
│   ├── intent.py        ← TERMINAL / REPO / CHAT classifier
│   ├── tools.py         ← list_files, read_file
│   └── main.py          ← entry point, wires threads together
├── prompts/
│   ├── loader.py
│   └── prompt.yaml
└── models/
    └── vosk/
        └── vosk-model-small-en-in-0.4
    
```

---

## Tuning

**Too many false wake-word triggers** → raise `vad_threshold` in `config.yaml`

**Misses soft speech** → lower `vad_threshold`

**Utterance cuts off too early** → raise `silence_cutoff` (default 1.8 s)

**STT too slow** → switch to `stt_model: tiny` in `config.yaml`

**Want better accuracy** → switch to `stt_model: medium` or `large-v3`

---

## License
Apache 2.0 — see `LICENSE`.
