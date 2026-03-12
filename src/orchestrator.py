from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
import uuid

import numpy as np

from src.audio.audio import AudioIO
from src.pipeline.chunking import SentenceChunker
from src.core.config import AppConfig
from src.pipeline.dialogue import ConversationMemory
from src.models.kokoro import KokoroTTS
from src.models.ministral import MinistralLLM
from src.core.runtime_types import PlaybackItem, TranscriptEvent, TurnContext, Utterance
from src.ui.terminal_ui import TerminalUI
from src.audio.vad import TurnDetector
from src.models.voxtral import VoxtralSTT


@dataclass(slots=True)
class ActiveTurn:
    context: TurnContext
    task: asyncio.Task[None]


@dataclass
class SpeculativeState:
    """Tracks speculative LLM generation during speech."""
    transcript: str = ""
    stable_since: float = 0.0
    task: asyncio.Task[None] | None = None
    cancelled: bool = False
    reply_parts: list[str] = field(default_factory=list)
    tts_items: list[tuple[str, np.ndarray, int]] = field(default_factory=list)
    chunker: SentenceChunker | None = None
    generation_done: bool = False


class RealtimeOrchestrator:
    def __init__(self, config: AppConfig, ui: TerminalUI) -> None:
        self._config = config
        self._ui = ui
        self._audio = AudioIO(config)
        self._detector = TurnDetector(config)
        self._stt = VoxtralSTT(config)
        self._llm = MinistralLLM(config)
        self._tts = KokoroTTS(config)
        self._memory = ConversationMemory(config.context_turns, config.system_prompt)
        self._capture_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._playback_queue: asyncio.Queue[PlaybackItem | None] = asyncio.Queue()
        self._playback_task: asyncio.Task[None] | None = None
        self._active_turn: ActiveTurn | None = None
        self._background_turns: set[asyncio.Task[None]] = set()
        self._pending_turn_id: str | None = None
        self._pending_partial: str = ""
        self._stt_task: asyncio.Task[str] | None = None
        self._speculative: SpeculativeState | None = None

    async def run(self) -> None:
        await self._warmup()
        await self._audio.start_capture(self._capture_queue)
        self._playback_task = asyncio.create_task(self._playback_loop())
        self._ui.set_state("listening", "speak into the microphone")

        try:
            while True:
                chunk = await self._capture_queue.get()
                for event in self._detector.feed(chunk):
                    if event.kind == "speech_start":
                        if self._should_interrupt_for_speech_start():
                            self._interrupt_active_turn()
                        self._on_speech_start()
                    elif event.kind == "speech_frames" and event.samples is not None:
                        await self._on_speech_frames(event.samples, event.sample_rate or self._config.sample_rate)
                    elif event.kind == "utterance" and event.utterance is not None:
                        await self._start_turn(event.utterance)
        except asyncio.CancelledError:
            raise
        finally:
            await self._shutdown()

    async def _warmup(self) -> None:
        self._ui.set_state("warming_up", "loading MLX models")
        try:
            await asyncio.gather(
                self._stt.warmup(),
                self._llm.warmup(),
                self._tts.warmup(),
            )
        except Exception as exc:
            self._ui.error(str(exc))
            raise SystemExit(1) from exc

    def _on_speech_start(self) -> None:
        """Called when speech starts. Initialize incremental STT state."""
        self._pending_turn_id = uuid.uuid4().hex[:12]
        self._pending_partial = ""
        self._cancel_pending_stt()
        self._cancel_speculative()
        self._ui.set_state("listening", "speech detected")

    async def _on_speech_frames(self, samples, sample_rate: int) -> None:
        """Called periodically during speech with accumulated audio."""
        if self._pending_turn_id is None:
            return

        self._cancel_pending_stt()

        async def do_transcribe() -> str:
            return await self._stt.transcribe_snapshot(samples, sample_rate)

        self._stt_task = asyncio.create_task(do_transcribe())
        try:
            partial = await self._stt_task
            if partial:
                old_partial = self._pending_partial
                self._pending_partial = partial
                self._ui.partial_transcript(partial)

                if partial.strip() and partial.strip()[-1] in ".?!":
                    self._detector.set_dynamic_end_ms(self._config.short_vad_end_ms)

                if self._config.enable_speculative_llm:
                    self._maybe_launch_speculative(partial, old_partial)
        except asyncio.CancelledError:
            pass
        finally:
            self._stt_task = None

    def _should_speculate(self, partial: str, stable_for_ms: float) -> bool:
        """Determine if we should launch speculative LLM on this partial."""
        if stable_for_ms < self._config.speculate_after_ms:
            return False
        stripped = partial.strip()
        if not stripped:
            return False
        if stripped[-1] in ".?!":
            return True
        if len(stripped.split()) >= 4 and stable_for_ms >= self._config.speculate_after_ms * 2:
            return True
        return False

    def _maybe_launch_speculative(self, partial: str, old_partial: str) -> None:
        """Check if we should launch or restart speculative LLM."""
        now = time.monotonic()

        if self._speculative is None:
            self._speculative = SpeculativeState(
                transcript=partial,
                stable_since=now,
            )
        elif partial != self._speculative.transcript:
            self._cancel_speculative()
            self._speculative = SpeculativeState(
                transcript=partial,
                stable_since=now,
            )
            return

        stable_for_ms = (now - self._speculative.stable_since) * 1000.0

        if self._speculative.task is None and self._should_speculate(partial, stable_for_ms):
            self._speculative.task = asyncio.create_task(
                self._run_speculative(partial)
            )

    async def _run_speculative(self, transcript: str) -> None:
        """Run speculative LLM generation (does not play audio, just buffers)."""
        if self._speculative is None:
            return

        spec = self._speculative
        spec.chunker = SentenceChunker(
            stall_ms=self._config.stall_chunk_ms,
            char_threshold=self._config.stall_chunk_chars,
            first_chunk_stall_ms=self._config.first_chunk_stall_ms,
            first_chunk_chars=self._config.first_chunk_chars,
        )

        messages = self._memory.build_messages(transcript)

        try:
            async for text in self._llm.stream_reply(messages, lambda: spec.cancelled):
                if spec.cancelled:
                    return
                spec.reply_parts.append(text)
                for sentence in spec.chunker.push(text):
                    await self._synthesize_speculative(spec, sentence)
                stalled = spec.chunker.flush_due_to_stall()
                if stalled is not None:
                    await self._synthesize_speculative(spec, stalled)

            final_tail = spec.chunker.finalize()
            if final_tail is not None:
                await self._synthesize_speculative(spec, final_tail)
            spec.generation_done = True
        except asyncio.CancelledError:
            pass

    async def _synthesize_speculative(self, spec: SpeculativeState, text: str) -> None:
        """Synthesize TTS for speculative output (buffer, don't play yet)."""
        if spec.cancelled or not text:
            return
        samples, sample_rate = await self._tts.synthesize(text)
        if spec.cancelled or len(samples) == 0:
            return
        spec.tts_items.append((text, samples, sample_rate))

    def _cancel_speculative(self) -> None:
        """Cancel any in-flight speculative LLM task."""
        if self._speculative is not None:
            self._speculative.cancelled = True
            if self._speculative.task is not None and not self._speculative.task.done():
                self._speculative.task.cancel()
            self._speculative = None

    def _cancel_pending_stt(self) -> None:
        """Cancel any in-flight STT snapshot task."""
        if self._stt_task is not None and not self._stt_task.done():
            self._stt_task.cancel()
            self._stt_task = None

    async def _start_turn(self, utterance: Utterance) -> None:
        if self._active_turn is not None:
            self._interrupt_active_turn(announce=False)

        self._cancel_pending_stt()

        turn_id = self._pending_turn_id or uuid.uuid4().hex[:12]
        initial_partial = self._pending_partial
        speculative = self._speculative
        self._pending_turn_id = None
        self._pending_partial = ""
        self._speculative = None

        self._ui.set_state("transcribing")
        context = TurnContext(turn_id=turn_id, utterance_id=utterance.utterance_id, transcript="")
        task = asyncio.create_task(
            self._handle_turn(context, utterance, initial_partial, speculative)
        )
        self._active_turn = ActiveTurn(context=context, task=task)

    async def _handle_turn(
        self,
        context: TurnContext,
        utterance: Utterance,
        initial_partial: str = "",
        speculative: SpeculativeState | None = None,
    ) -> None:
        try:
            if initial_partial:
                transcript = await self._stt.transcribe_snapshot(
                    utterance.samples, utterance.sample_rate
                )
            else:
                transcript = await self._stt.transcribe(
                    utterance,
                    lambda event: self._on_partial_transcript(context, event),
                    lambda: context.cancelled,
                )
            if context.cancelled:
                return
            transcript = transcript.strip()
            if not transcript:
                self._ui.set_state("listening", "empty transcript ignored")
                if speculative is not None:
                    speculative.cancelled = True
                    if speculative.task and not speculative.task.done():
                        speculative.task.cancel()
                return
            context.transcript = transcript
            self._ui.final_user(transcript)

            if speculative is not None and speculative.transcript.strip() == transcript:
                await self._use_speculative_output(context, speculative, transcript)
            else:
                if speculative is not None:
                    speculative.cancelled = True
                    if speculative.task and not speculative.task.done():
                        speculative.task.cancel()
                await self._generate_fresh_reply(context, transcript)

        except Exception as exc:
            if context.cancelled:
                return
            self._ui.error(str(exc))
            self._ui.set_state("error")
        finally:
            if utterance.wav_path.exists():
                utterance.wav_path.unlink()

    async def _use_speculative_output(
        self, context: TurnContext, spec: SpeculativeState, transcript: str
    ) -> None:
        """Use pre-computed speculative output instead of running LLM again."""
        self._ui.set_state("speaking", "using speculative output")

        for text, samples, sample_rate in spec.tts_items:
            if context.cancelled:
                return
            context.tts_chunks.append(text)
            context.interruptible = True
            context.pending_playback += 1
            await self._playback_queue.put(
                PlaybackItem(turn_id=context.turn_id, samples=samples, sample_rate=sample_rate)
            )

        if spec.task and not spec.task.done():
            try:
                await spec.task
            except asyncio.CancelledError:
                pass

        if not spec.generation_done and spec.chunker:
            final_tail = spec.chunker.finalize()
            if final_tail:
                await self._queue_tts(context, final_tail)

        reply = "".join(spec.reply_parts).strip()
        context.reply_text = reply
        context.generation_done = True
        if reply and not context.cancelled:
            self._memory.commit_turn(transcript, reply)
            self._ui.final_assistant(reply)
            if context.pending_playback > 0:
                self._ui.set_state("speaking")
            else:
                self._finish_active_turn(context.turn_id)
        else:
            self._finish_active_turn(context.turn_id)

    async def _generate_fresh_reply(self, context: TurnContext, transcript: str) -> None:
        """Generate LLM reply from scratch (no speculative match)."""
        self._ui.set_state("thinking")
        messages = self._memory.build_messages(transcript)

        chunker = SentenceChunker(
            stall_ms=self._config.stall_chunk_ms,
            char_threshold=self._config.stall_chunk_chars,
            first_chunk_stall_ms=self._config.first_chunk_stall_ms,
            first_chunk_chars=self._config.first_chunk_chars,
        )
        reply_parts: list[str] = []
        async for text in self._llm.stream_reply(messages, lambda: context.cancelled):
            if context.cancelled:
                return
            reply_parts.append(text)
            for sentence in chunker.push(text):
                await self._queue_tts(context, sentence)
            stalled = chunker.flush_due_to_stall()
            if stalled is not None:
                await self._queue_tts(context, stalled)

        final_tail = chunker.finalize()
        if final_tail is not None:
            await self._queue_tts(context, final_tail)

        reply = "".join(reply_parts).strip()
        context.reply_text = reply
        context.generation_done = True
        if reply and not context.cancelled:
            self._memory.commit_turn(transcript, reply)
            self._ui.final_assistant(reply)
            if context.pending_playback > 0:
                self._ui.set_state("speaking")
            else:
                self._finish_active_turn(context.turn_id)
        else:
            self._finish_active_turn(context.turn_id)

    async def _queue_tts(self, context: TurnContext, text: str) -> None:
        if context.cancelled or not text:
            return
        context.tts_chunks.append(text)
        samples, sample_rate = await self._tts.synthesize(text)
        if context.cancelled or len(samples) == 0:
            return
        context.interruptible = True
        context.pending_playback += 1
        if (
            self._active_turn is not None
            and self._active_turn.context.turn_id == context.turn_id
        ):
            self._ui.set_state("speaking")
        await self._playback_queue.put(
            PlaybackItem(turn_id=context.turn_id, samples=samples, sample_rate=sample_rate)
        )

    async def _playback_loop(self) -> None:
        while True:
            item = await self._playback_queue.get()
            if item is None:
                return
            if self._active_turn is None:
                continue
            if item.turn_id != self._active_turn.context.turn_id:
                continue
            await self._audio.play(item.samples, item.sample_rate)
            if self._active_turn is not None and item.turn_id == self._active_turn.context.turn_id:
                self._active_turn.context.pending_playback = max(
                    0,
                    self._active_turn.context.pending_playback - 1,
                )
                if (
                    self._active_turn.context.generation_done
                    and self._active_turn.context.pending_playback == 0
                ):
                    self._finish_active_turn(item.turn_id)

    def _interrupt_active_turn(self, announce: bool = True) -> None:
        if self._active_turn is None:
            return
        active_turn = self._active_turn
        active_turn.context.cancelled = True
        self._track_background_turn(active_turn.task)
        self._audio.stop_playback()
        self._drain_playback_queue()
        self._cancel_speculative()
        self._active_turn = None
        if announce:
            self._ui.set_state("interrupted")

    def _drain_playback_queue(self) -> None:
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _on_partial_transcript(self, context: TurnContext, event: TranscriptEvent) -> None:
        if context.cancelled:
            return
        self._ui.partial_transcript(event.text)

    async def _shutdown(self) -> None:
        if self._active_turn is not None:
            self._interrupt_active_turn(announce=False)
        self._cancel_pending_stt()
        self._cancel_speculative()
        if self._playback_task is not None:
            await self._playback_queue.put(None)
            await self._playback_task
        if self._background_turns:
            await asyncio.gather(*self._background_turns, return_exceptions=True)
        self._stt.close()
        self._llm.close()
        self._tts.close()
        await self._audio.stop()

    def _finish_active_turn(self, turn_id: str) -> None:
        if self._active_turn is None:
            self._ui.set_state("listening")
            return
        if self._active_turn.context.turn_id != turn_id:
            return
        self._active_turn = None
        self._ui.set_state("listening")

    def _track_background_turn(self, task: asyncio.Task[None]) -> None:
        if task.done():
            return
        self._background_turns.add(task)
        task.add_done_callback(self._finish_background_turn)

    def _finish_background_turn(self, task: asyncio.Task[None]) -> None:
        self._background_turns.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._ui.error(str(exc))

    def _should_interrupt_for_speech_start(self) -> bool:
        if self._active_turn is None:
            return False
        return self._active_turn.context.interruptible
