from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
import uuid
import wave

import numpy as np

from src.core.config import AppConfig
from src.core.runtime_types import TranscriptEvent, Utterance


class VoxtralSTT:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model = None

    async def warmup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_model)

    async def transcribe(
        self,
        utterance: Utterance,
        on_partial,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._transcribe_blocking,
            utterance,
            on_partial,
            should_stop or (lambda: False),
        )

    async def transcribe_snapshot(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> str:
        """Transcribe a numpy array of audio samples directly."""
        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._transcribe_snapshot_blocking,
            samples,
            sample_rate,
        )

    def _load_model(self):
        if self._model is None:
            from mlx_audio.stt.utils import load

            self._model = load(self._config.stt_model)
        return self._model

    def _transcribe_blocking(
        self,
        utterance: Utterance,
        on_partial,
        should_stop: Callable[[], bool],
    ) -> str:
        model = self._load_model()
        partial = ""
        for chunk in model.generate(
            str(utterance.wav_path),
            stream=True,
        ):
            if should_stop():
                break
            chunk_text = getattr(chunk, "text", None)
            partial += str(chunk_text if chunk_text is not None else chunk)
            if should_stop():
                break
            on_partial(TranscriptEvent(utterance.utterance_id, partial, is_final=False))
        return partial.strip()

    def _transcribe_snapshot_blocking(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> str:
        model = self._load_model()
        snapshot_id = uuid.uuid4().hex[:8]
        wav_path = Path("/tmp") / f"snapshot-{snapshot_id}.wav"
        try:
            int16_samples = samples.astype(np.int16, copy=False)
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(int16_samples.tobytes())

            result = model.generate(str(wav_path), stream=False)
            text = getattr(result, "text", None)
            if text is not None:
                return str(text).strip()
            return str(result).strip()
        finally:
            if wav_path.exists():
                wav_path.unlink()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
