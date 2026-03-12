from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
import uuid
import wave

import numpy as np

from src.core.config import AppConfig
from src.core.runtime_types import AudioChunk, Utterance


@dataclass(slots=True)
class TurnDetectorEvent:
    kind: str
    utterance: Utterance | None = None
    samples: np.ndarray | None = None
    sample_rate: int | None = None


class TurnDetector:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._pre_roll_frames = max(1, config.pre_roll_ms // config.frame_ms)
        self._min_frames = max(1, config.min_utterance_ms // config.frame_ms)
        self._default_end_silence_frames = max(1, config.vad_end_ms // config.frame_ms)
        self._end_silence_frames = self._default_end_silence_frames
        self._max_frames = int(
            config.max_utterance_seconds * 1000 / config.frame_ms
        )
        self._snapshot_interval_frames = max(1, config.stt_snapshot_interval_ms // config.frame_ms)
        self._preroll: deque[np.ndarray] = deque(maxlen=self._pre_roll_frames)
        self._frames: list[np.ndarray] = []
        self._speech_started = False
        self._speech_started_at = 0.0
        self._silence_run = 0
        self._voiced_run = 0
        self._frames_since_last_snapshot = 0
        self._vad = self._build_vad()

    def set_dynamic_end_ms(self, end_ms: int) -> None:
        """Temporarily adjust the silence duration required to end an utterance."""
        self._end_silence_frames = max(1, end_ms // self._config.frame_ms)

    def feed(self, chunk: AudioChunk) -> list[TurnDetectorEvent]:
        samples = chunk.samples.astype(np.int16, copy=False)
        if not self._speech_started:
            self._preroll.append(samples.copy())
        voiced = self._is_speech(samples)
        events: list[TurnDetectorEvent] = []

        if not self._speech_started:
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= 2:
                self._speech_started = True
                self._speech_started_at = chunk.timestamp - (
                    len(self._preroll) * self._config.frame_ms / 1000.0
                )
                self._frames = list(self._preroll)
                self._silence_run = 0
                self._frames_since_last_snapshot = 0
                events.append(TurnDetectorEvent(kind="speech_start"))
            return events

        self._frames.append(samples.copy())
        self._frames_since_last_snapshot += 1
        if voiced:
            self._silence_run = 0
        else:
            self._silence_run += 1

        if len(self._frames) >= self._max_frames:
            utterance = self._finalize(chunk.timestamp)
            if utterance is not None:
                events.append(TurnDetectorEvent(kind="utterance", utterance=utterance))
            return events

        if self._silence_run >= self._end_silence_frames and len(self._frames) >= self._min_frames:
            utterance = self._finalize(chunk.timestamp)
            if utterance is not None:
                events.append(TurnDetectorEvent(kind="utterance", utterance=utterance))
            return events

        if self._frames_since_last_snapshot >= self._snapshot_interval_frames:
            self._frames_since_last_snapshot = 0
            all_samples = np.concatenate(self._frames).astype(np.int16, copy=False)
            events.append(TurnDetectorEvent(
                kind="speech_frames",
                samples=all_samples,
                sample_rate=self._config.sample_rate,
            ))

        return events

    def _is_speech(self, samples: np.ndarray) -> bool:
        frame_bytes = samples.tobytes()
        if self._vad is not None:
            return bool(self._vad.is_speech(frame_bytes, self._config.sample_rate))
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        return rms > 900.0

    def _finalize(self, end_time: float) -> Utterance | None:
        if not self._frames:
            self._reset()
            return None
        all_samples = np.concatenate(self._frames).astype(np.int16, copy=False)
        utterance_id = uuid.uuid4().hex[:12]
        wav_path = Path("/tmp") / f"utterance-{utterance_id}.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self._config.sample_rate)
            handle.writeframes(all_samples.tobytes())

        utterance = Utterance(
            utterance_id=utterance_id,
            samples=all_samples,
            sample_rate=self._config.sample_rate,
            start_time=self._speech_started_at or time.monotonic(),
            end_time=end_time,
            wav_path=wav_path,
        )
        self._reset()
        return utterance

    def _reset(self) -> None:
        self._frames = []
        self._speech_started = False
        self._speech_started_at = 0.0
        self._silence_run = 0
        self._voiced_run = 0
        self._frames_since_last_snapshot = 0
        self._preroll.clear()
        self._end_silence_frames = self._default_end_silence_frames

    @staticmethod
    def _build_vad():
        try:
            import webrtcvad  # type: ignore
        except ImportError:
            return None
        vad = webrtcvad.Vad(2)
        return vad
