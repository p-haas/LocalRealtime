from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.core.config import AppConfig
from src.core.runtime_types import AudioChunk


class AudioIO:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._capture_queue: asyncio.Queue[AudioChunk] | None = None
        self._input_stream: Any = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._play_lock = asyncio.Lock()

    async def start_capture(self, queue: asyncio.Queue[AudioChunk]) -> None:
        sounddevice = self._import_sounddevice()
        self._loop = asyncio.get_running_loop()
        self._capture_queue = queue
        self._input_stream = sounddevice.InputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._config.frame_samples,
            callback=self._capture_callback,
            device=self._config.input_device,
        )
        self._input_stream.start()

    async def stop(self) -> None:
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        sounddevice = self._import_sounddevice()
        output = samples.astype(np.float32, copy=False)
        async with self._play_lock:
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._play_blocking,
                sounddevice,
                output,
                sample_rate,
                self._config.output_device,
            )

    def stop_playback(self) -> None:
        sounddevice = self._import_sounddevice()
        sounddevice.stop()

    def _capture_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        del frames, status
        if self._loop is None or self._capture_queue is None:
            return
        chunk = AudioChunk(
            samples=np.copy(indata[:, 0]),
            sample_rate=self._config.sample_rate,
            timestamp=self._extract_timestamp(time_info),
        )
        self._loop.call_soon_threadsafe(self._enqueue_chunk, chunk)

    @staticmethod
    def _play_blocking(
        sounddevice: Any,
        samples: np.ndarray,
        sample_rate: int,
        output_device: str | None,
    ) -> None:
        sounddevice.play(
            samples,
            samplerate=sample_rate,
            device=output_device,
            blocking=True,
        )

    @staticmethod
    def _import_sounddevice():
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for audio I/O. Install project dependencies first."
            ) from exc
        return sd

    def _extract_timestamp(self, time_info: Any) -> float:
        if self._loop is None:
            return 0.0
        if isinstance(time_info, dict):
            return float(time_info.get("input_buffer_adc_time", self._loop.time()))

        value = getattr(time_info, "inputBufferAdcTime", None)
        if value is None:
            value = getattr(time_info, "input_buffer_adc_time", None)
        if value is None:
            return self._loop.time()
        return float(value)

    def _enqueue_chunk(self, chunk: AudioChunk) -> None:
        if self._capture_queue is None:
            return
        try:
            self._capture_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass
