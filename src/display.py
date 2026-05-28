# ANSI colour codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_MAG    = "\033[35m"


class Display:

    def banner(self):
        print(f"""
{_BOLD}{_CYAN}
  ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗
  ██║   ██║██╔═══██╗██║██╔════╝██╔════╝
  ██║   ██║██║   ██║██║██║     █████╗  
  ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  
   ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗
    ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝
{_RESET}
  {_DIM}Say {_RESET}{_BOLD}"Hey Voker"{_RESET}{_DIM} to start · speak to interrupt{_RESET}
""")

    def info(self, msg: str):
        print(f"{_DIM}  {msg}{_RESET}", flush=True)

    def warn(self, msg: str):
        print(f"{_YELLOW}  ⚠  {msg}{_RESET}", flush=True)

    def wake(self):
        print(f"\n{_BOLD}{_GREEN}  ◉  Wake word detected — listening…{_RESET}", flush=True)

    def user_said(self, text: str):
        print(f"\n{_BOLD}{_BLUE}  You ▶{_RESET}  {text}\n", flush=True)

    def interrupt(self):
        print(f"\n{_BOLD}{_YELLOW}  ↩  Interrupting — listening for new command…{_RESET}", flush=True)

    def cancelled(self):
        print(f"\n{_DIM}  [generation cancelled]{_RESET}", flush=True)

    def agent_start(self):
        print(f"\n{_BOLD}{_MAG}  Agent ▶{_RESET}  ", end="", flush=True)

    def token(self, tok: str):
        """Print a single streaming token — no newline."""
        print(tok, end="", flush=True)

    def line(self, text: str):
        """Print a full line (for subprocess backends)."""
        print(f"  {text}", flush=True)

    def agent_stop(self):
        print(f"\n{_DIM}  ─────────────────────────────{_RESET}\n", flush=True)
