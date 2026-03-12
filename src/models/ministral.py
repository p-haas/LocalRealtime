from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Callable

from src.core.config import AppConfig


class MinistralLLM:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model = None
        self._tokenizer = None

    async def warmup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_model)

    async def stream_reply(
        self,
        messages: list[dict[str, str]],
        should_stop: Callable[[], bool] | None = None,
    ):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stop = should_stop or (lambda: False)

        def producer() -> None:
            model, tokenizer = self._load_model()
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            from mlx_lm import stream_generate

            try:
                for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
                    if stop():
                        break
                    if not response.text:
                        continue
                    loop.call_soon_threadsafe(queue.put_nowait, response.text)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            from mlx_lm import load

            try:
                self._model, self._tokenizer = load(self._config.llm_model)
            except Exception as exc:
                if "Oniguruma" not in str(exc):
                    raise
                self._patch_tokenizer_oniguruma(self._config.llm_model)
                self._model, self._tokenizer = load(self._config.llm_model)
        return self._model, self._tokenizer

    @staticmethod
    def _patch_tokenizer_oniguruma(model_id: str) -> None:
        """Remove (?s:...) inline-flag syntax from the cached tokenizer.json.

        tokenizers 0.22.x bundles an Oniguruma version that rejects the
        non-capturing group with inline flags syntax (e.g. ``(?s:...)``)
        when it appears inside a very long compiled pattern. The ``s`` flag
        (dotall) has no effect on the Mistral pre-tokenizer patterns because
        none of the alternatives contain ``.``, so replacing it with a plain
        non-capturing group is semantically equivalent.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return

        try:
            tok_path = hf_hub_download(model_id, "tokenizer.json")
        except Exception:
            return

        with open(tok_path) as fh:
            raw = fh.read()

        if "(?s:" not in raw:
            return

        with open(tok_path, "w") as fh:
            fh.write(raw.replace("(?s:", "(?:"))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
