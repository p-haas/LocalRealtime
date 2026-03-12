from __future__ import annotations

from datetime import datetime


class TerminalUI:
    def __init__(self) -> None:
        self._state = "idle"

    def set_state(self, state: str, detail: str | None = None) -> None:
        self._state = state
        suffix = f" | {detail}" if detail else ""
        self._print(f"[{state.upper()}]{suffix}")

    def partial_transcript(self, text: str) -> None:
        self._print(f"partial: {text}")

    def final_user(self, text: str) -> None:
        self._print(f"user: {text}")

    def final_assistant(self, text: str) -> None:
        self._print(f"assistant: {text}")

    def error(self, text: str) -> None:
        self._print(f"error: {text}")

    def info(self, text: str) -> None:
        self._print(text)

    def _print(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"{stamp} {text}", flush=True)
