from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.core.config import AppConfig
from src.core.mlx_guard import MLX_LOCK


class KokoroTTS:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model = None
        self._runtime_ready = False

    async def warmup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._warmup_blocking)

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._synthesize_blocking,
            text,
        )

    def _load_model(self):
        if self._model is None:
            self._patch_phonemizer_compat()
            try:
                from mlx_audio.tts.utils import load_model
            except ImportError as exc:
                raise RuntimeError(
                    "Kokoro TTS dependencies are missing. Install the project dependencies "
                    "again, including `misaki`."
                ) from exc

            try:
                self._model = load_model(self._config.tts_model)
            except ImportError as exc:
                message = str(exc)
                if "misaki" in message:
                    raise RuntimeError(
                        "Kokoro requires the `misaki` package. Run "
                        "`pip install -r requirements.txt` or `pip install misaki`."
                    ) from exc
                if "EspeakWrapper" in message or "set_data_path" in message:
                    raise RuntimeError(
                        "Kokoro's phonemizer stack is incompatible in this environment. "
                        "Install `phonemizer-fork` and prefer the repo requirements file."
                    ) from exc
                raise RuntimeError(
                    f"Failed to load Kokoro model `{self._config.tts_model}`: {message}"
                ) from exc
        return self._model

    def _warmup_blocking(self):
        with MLX_LOCK:
            model = self._load_model()
            self._ensure_runtime_assets(model)
        return model

    def _ensure_runtime_assets(self, model) -> None:
        if self._runtime_ready:
            return

        lang_code = self._resolve_lang_code(self._config.voice)
        get_pipeline = getattr(model, "_get_pipeline", None)
        if callable(get_pipeline):
            pipeline = get_pipeline(lang_code)
            load_voice = getattr(pipeline, "load_voice", None)
            if callable(load_voice):
                load_voice(self._config.voice)

        self._runtime_ready = True

    @staticmethod
    def _resolve_lang_code(voice: str) -> str:
        stem = Path(voice).stem
        prefix = stem.split("_", 1)[0]
        return prefix[:1] or "a"

    @staticmethod
    def _patch_phonemizer_compat() -> None:
        try:
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError:
            return

        if hasattr(EspeakWrapper, "set_data_path"):
            return

        def set_data_path(cls, data_path):
            cls._OVERRIDE_DATA_PATH = data_path

        original_fetch = EspeakWrapper._fetch_version_and_path

        def patched_fetch(self):
            original_fetch(self)
            override = getattr(type(self), "_OVERRIDE_DATA_PATH", None)
            if override:
                from pathlib import Path

                self._data_path = Path(override)

        EspeakWrapper.set_data_path = classmethod(set_data_path)
        EspeakWrapper._fetch_version_and_path = patched_fetch

    def _synthesize_blocking(self, text: str) -> tuple[np.ndarray, int]:
        model = self._load_model()
        self._ensure_runtime_assets(model)
        clips: list[np.ndarray] = []
        sample_rate = 24_000
        lang_code = self._resolve_lang_code(self._config.voice)
        with MLX_LOCK:
            for result in model.generate(text, voice=self._config.voice, lang_code=lang_code):
                clips.append(np.asarray(result.audio, dtype=np.float32))
                sample_rate = result.sample_rate
        if not clips:
            return np.zeros(0, dtype=np.float32), sample_rate
        merged = np.concatenate(clips).astype(np.float32, copy=False)
        return merged, sample_rate

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
