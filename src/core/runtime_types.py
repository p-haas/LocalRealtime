from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    timestamp: float


@dataclass(slots=True)
class Utterance:
    utterance_id: str
    samples: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    wav_path: Path

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / float(self.sample_rate)


@dataclass(slots=True)
class TranscriptEvent:
    utterance_id: str
    text: str
    is_final: bool


@dataclass(slots=True)
class AssistantChunk:
    turn_id: str
    text: str
    is_final: bool = False


@dataclass(slots=True)
class PlaybackItem:
    turn_id: str
    samples: np.ndarray
    sample_rate: int


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(slots=True)
class TurnContext:
    turn_id: str
    utterance_id: str
    transcript: str
    cancelled: bool = False
    reply_text: str = ""
    tts_chunks: list[str] = field(default_factory=list)
    generation_done: bool = False
    pending_playback: int = 0
    interruptible: bool = False
