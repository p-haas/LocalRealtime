from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    stt_model: str = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    llm_model: str = "mlx-community/Ministral-3-3B-Instruct-2512"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    voice: str = "af_heart"
    input_device: str | None = None
    output_device: str | None = None
    latency_preset: str = "low"
    context_turns: int = 4
    vad_end_ms: int = 500
    short_vad_end_ms: int = 250
    min_utterance_ms: int = 300
    max_utterance_seconds: float = 10.0
    transcription_delay_ms: int = 240
    sample_rate: int = 16_000
    frame_ms: int = 20
    pre_roll_ms: int = 300
    stt_snapshot_interval_ms: int = 300
    stall_chunk_ms: int = 350
    stall_chunk_chars: int = 80
    first_chunk_stall_ms: int = 150
    first_chunk_chars: int = 30
    enable_speculative_llm: bool = True
    speculate_after_ms: int = 200
    system_prompt: str = (
        "You are a concise local voice assistant. Respond in short spoken sentences, "
        "avoid markdown, and prefer practical answers."
    )

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


DEFAULT_CONFIG = AppConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local realtime voice assistant on macOS using MLX."
    )
    parser.add_argument("--stt-model", default=DEFAULT_CONFIG.stt_model)
    parser.add_argument("--llm-model", default=DEFAULT_CONFIG.llm_model)
    parser.add_argument("--tts-model", default=DEFAULT_CONFIG.tts_model)
    parser.add_argument("--voice", default=DEFAULT_CONFIG.voice)
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument(
        "--latency-preset",
        choices=("low", "balanced", "quality"),
        default=DEFAULT_CONFIG.latency_preset,
    )
    parser.add_argument("--context-turns", type=int, default=DEFAULT_CONFIG.context_turns)
    parser.add_argument("--vad-end-ms", type=int, default=DEFAULT_CONFIG.vad_end_ms)
    parser.add_argument(
        "--min-utterance-ms",
        type=int,
        default=DEFAULT_CONFIG.min_utterance_ms,
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=DEFAULT_CONFIG.max_utterance_seconds,
    )
    parser.add_argument(
        "--transcription-delay-ms",
        type=int,
        default=DEFAULT_CONFIG.transcription_delay_ms,
    )
    return parser


def config_from_args() -> AppConfig:
    args = build_parser().parse_args()
    return AppConfig(
        stt_model=args.stt_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        voice=args.voice,
        input_device=args.input_device,
        output_device=args.output_device,
        latency_preset=args.latency_preset,
        context_turns=args.context_turns,
        vad_end_ms=args.vad_end_ms,
        min_utterance_ms=args.min_utterance_ms,
        max_utterance_seconds=args.max_utterance_seconds,
        transcription_delay_ms=args.transcription_delay_ms,
    )
