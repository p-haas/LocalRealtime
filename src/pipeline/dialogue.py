from __future__ import annotations

from collections import deque

from src.core.runtime_types import ConversationTurn


class ConversationMemory:
    def __init__(self, max_turn_pairs: int, system_prompt: str) -> None:
        self._max_messages = max_turn_pairs * 2
        self._system_prompt = system_prompt
        self._history: deque[ConversationTurn] = deque(maxlen=self._max_messages)

    def build_messages(self, user_text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(
            {"role": turn.role, "content": turn.content} for turn in self._history
        )
        messages.append({"role": "user", "content": user_text})
        return messages

    def commit_turn(self, user_text: str, assistant_text: str) -> None:
        self._history.append(ConversationTurn(role="user", content=user_text))
        self._history.append(ConversationTurn(role="assistant", content=assistant_text))

    def snapshot(self) -> list[ConversationTurn]:
        return list(self._history)
