from __future__ import annotations

import threading

# MLX's Metal backend is not thread-safe across Python threads. All blocking
# MLX inference calls (STT, LLM, TTS) must acquire this lock before touching
# the Metal device to prevent concurrent access segfaults.
MLX_LOCK = threading.Lock()
