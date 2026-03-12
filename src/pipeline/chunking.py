from __future__ import annotations

import time

_ABBREVS: frozenset[str] = frozenset({
    # Titles
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "rev", "gen", "sgt", "cpl",
    # Business / legal
    "inc", "ltd", "co", "corp", "dept", "est", "approx", "vs",
    # Common Latin / shorthand
    "etc", "vol", "no", "fig", "pp", "ed", "op",
    # Months
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # Address
    "st", "ave", "blvd", "rd", "ln",
})


class SentenceChunker:
    def __init__(
        self,
        stall_ms: int = 350,
        char_threshold: int = 80,
        first_chunk_stall_ms: int | None = None,
        first_chunk_chars: int | None = None,
    ) -> None:
        self._stall_ms = stall_ms
        self._char_threshold = char_threshold
        self._first_chunk_stall_ms = first_chunk_stall_ms if first_chunk_stall_ms is not None else stall_ms
        self._first_chunk_chars = first_chunk_chars if first_chunk_chars is not None else char_threshold
        self._buffer = ""
        self._last_emit = time.monotonic()
        self._first_chunk_emitted = False

    def push(self, text: str) -> list[str]:
        self._buffer += text
        chunks: list[str] = []

        while True:
            index = self._find_boundary(self._buffer)
            if index == -1:
                break
            chunk = self._buffer[: index + 1].strip()
            self._buffer = self._buffer[index + 1 :].lstrip()
            if chunk:
                chunks.append(chunk)
                self._last_emit = time.monotonic()
                self._first_chunk_emitted = True

        char_threshold = self._first_chunk_chars if not self._first_chunk_emitted else self._char_threshold
        if (
            self._buffer
            and len(self._buffer) >= char_threshold
            and self._buffer.endswith(" ")
        ):
            chunk = self._buffer.strip()
            self._buffer = ""
            chunks.append(chunk)
            self._last_emit = time.monotonic()
            self._first_chunk_emitted = True

        return chunks

    def flush_due_to_stall(self) -> str | None:
        if not self._buffer:
            return None
        stall_ms = self._first_chunk_stall_ms if not self._first_chunk_emitted else self._stall_ms
        elapsed_ms = (time.monotonic() - self._last_emit) * 1000.0
        if elapsed_ms < stall_ms:
            return None
        chunk = self._buffer.strip()
        self._buffer = ""
        self._last_emit = time.monotonic()
        if chunk:
            self._first_chunk_emitted = True
        return chunk or None

    def finalize(self) -> str | None:
        chunk = self._buffer.strip()
        self._buffer = ""
        return chunk or None

    @staticmethod
    def _find_boundary(buffer: str) -> int:
        n = len(buffer)
        for i, char in enumerate(buffer):
            if char not in ".?!":
                continue

            # Require whitespace (or end of string) after the punctuation.
            next_char = buffer[i + 1] if i + 1 < n else " "
            if not next_char.isspace():
                continue

            # '?' and '!' are unambiguous sentence-enders.
            if char in "?!":
                return i

            # For '.' apply extra heuristics to skip non-sentence boundaries.

            # Skip decimal numbers and ordinals: "3.5", "No. 4".
            if i > 0 and buffer[i - 1].isdigit():
                continue

            # Extract the word token immediately preceding the period.
            word_end = i
            word_start = word_end
            while word_start > 0 and not buffer[word_start - 1].isspace():
                word_start -= 1
            word = buffer[word_start:word_end]

            # Skip single characters (initials: "J.", "U.").
            if len(word) <= 1:
                continue

            # Skip abbreviation chains that already contain a period ("U.S", "e.g").
            if "." in word:
                continue

            # Skip known abbreviations.
            if word.lower() in _ABBREVS:
                continue

            return i

        return -1
