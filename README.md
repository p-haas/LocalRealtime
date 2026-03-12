# RealtimeSystem

A fully local, real-time voice assistant for **Apple Silicon Macs**, built on [MLX](https://github.com/ml-explore/mlx). No cloud APIs. No network round-trips. Everything runs on-device.

| Stage | Model |
|-------|-------|
| Speech-to-Text | [`mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) |
| Language Model | [`mlx-community/Ministral-3-3B-Instruct-2512`](https://huggingface.co/mlx-community/Ministral-3-3B-Instruct-2512) |
| Text-to-Speech | [`mlx-community/Kokoro-82M-bf16`](https://huggingface.co/mlx-community/Kokoro-82M-bf16) |

---

## Features

- **Continuous listening** — VAD state machine detects utterance boundaries at 16 kHz / 20 ms frames
- **Barge-in / interruption** — new speech cancels playback and immediately starts a fresh turn
- **Speculative generation** — LLM starts generating while the user is still speaking (when the partial transcript looks stable), cutting perceived latency
- **Incremental TTS playback** — reply is chunked into sentences and synthesized/queued in parallel with generation
- **Rolling conversation context** — configurable number of recent turns fed to the LLM
- **Terminal UI** — live state + transcript display, no GUI required
- **Fully configurable** — models, voices, devices, and latency tuning all via CLI flags

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| macOS on Apple Silicon | M1 / M2 / M3 / M4 |
| Python 3.11+ | |
| Microphone access | Grant to Terminal / iTerm2 in System Settings |
| ~8 GB free RAM | For all three models loaded simultaneously |
| ~5 GB disk | First run downloads model weights automatically |

---

## Installation

### Option A — pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

### Option B — uv (faster)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## Usage

### Start the assistant

```bash
python3 realtime.py
```

On first run, MLX downloads and caches all three model weights. Subsequent starts are faster.

### Override devices, voice, and tuning

```bash
python3 realtime.py \
  --input-device  "MacBook Pro Microphone" \
  --output-device "MacBook Pro Speakers" \
  --voice         af_heart \
  --latency-preset low \
  --context-turns 4 \
  --vad-end-ms    500
```

### List all options

```bash
python3 realtime.py --help
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--stt-model` | `Voxtral-Mini-4B-Realtime-2602-4bit` | HuggingFace model ID for speech-to-text |
| `--llm-model` | `Ministral-3-3B-Instruct-2512` | HuggingFace model ID for the language model |
| `--tts-model` | `Kokoro-82M-bf16` | HuggingFace model ID for text-to-speech |
| `--voice` | `af_heart` | Kokoro voice name |
| `--input-device` | system default | Microphone device name (passed to `sounddevice`) |
| `--output-device` | system default | Speaker device name (passed to `sounddevice`) |
| `--latency-preset` | `low` | `low` / `balanced` / `quality` |
| `--context-turns` | `4` | Number of recent user/assistant exchanges to keep |
| `--vad-end-ms` | `500` | Silence duration (ms) required to close an utterance |
| `--min-utterance-ms` | `300` | Minimum speech duration before a turn is accepted |
| `--max-utterance-seconds` | `10.0` | Hard cutoff for a single utterance |
| `--transcription-delay-ms` | `240` | Voxtral incremental transcription cadence |

---

## Architecture

```
realtime.py
└── RealtimeOrchestrator        (src/orchestrator.py)
    ├── AudioIO                 (src/audio/audio.py)      — mic capture + speaker playback
    ├── TurnDetector            (src/audio/vad.py)        — WebRTC VAD state machine
    ├── VoxtralSTT              (src/models/voxtral.py)   — streaming + snapshot transcription
    ├── MinistralLLM            (src/models/ministral.py) — streaming token generation
    ├── KokoroTTS               (src/models/kokoro.py)    — sentence-level synthesis
    ├── SentenceChunker         (src/pipeline/chunking.py)— LLM output → speakable segments
    ├── ConversationMemory      (src/pipeline/dialogue.py)— rolling context for the LLM
    └── TerminalUI              (src/ui/terminal_ui.py)   — status + transcript display
```

### Turn lifecycle

```
mic frames (20 ms)
  → VAD detects speech start  → start incremental STT snapshots
                               → (optionally) launch speculative LLM
  → VAD detects end-of-speech → final transcription
                               → if speculative transcript matches → reuse buffered TTS
                               → otherwise → fresh LLM stream → chunked TTS synthesis
                               → playback queue → speaker
  → new speech during playback → cancel active turn → return to listening
```

### Project layout

```
realtime.py          — entry point
src/
  orchestrator.py    — async turn coordination and interruption logic
  audio/
    audio.py         — sounddevice capture / playback
    vad.py           — WebRTC VAD state machine
  core/
    config.py        — AppConfig dataclass + CLI argument parser
    runtime_types.py — shared types (Utterance, TurnContext, PlaybackItem, …)
  models/
    voxtral.py       — VoxtralSTT wrapper
    ministral.py     — MinistralLLM wrapper
    kokoro.py        — KokoroTTS wrapper
  pipeline/
    chunking.py      — sentence chunker with stall-flush logic
    dialogue.py      — ConversationMemory (rolling context)
  ui/
    terminal_ui.py   — live terminal status and transcript rendering
```

---

## Operational notes

- **Headphones are strongly recommended.** Open-speaker use risks the microphone picking up the assistant's own speech and triggering spurious barge-ins.
- **Cold start** is slow the first time each model is loaded. Subsequent turns within the same session are much faster.
- The **speculative LLM** feature (`enable_speculative_llm`, on by default) launches generation while you are still speaking. It is cancelled and discarded if the final transcript diverges from the speculative one.

---

## Troubleshooting

**No microphone input**

- Go to **System Settings → Privacy & Security → Microphone** and enable access for your terminal app.
- Pass `--input-device` explicitly if the default input is not your microphone.

**No audio output**

- Pass `--output-device` explicitly, or check **System Settings → Sound → Output**.

**Import or runtime errors**

- Make sure the virtual environment is activated: `source .venv/bin/activate`
- Reinstall in editable mode: `pip install -e .`

**Slow first response**

- Model weights are downloading or the MLX runtime is JIT-compiling kernels. This only happens once per model version.

---

## Acknowledgements

- [mlx-audio](https://github.com/blaizzy/mlx-audio) — MLX-native audio models (Voxtral STT, Kokoro TTS)
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — MLX-native language model inference
- [webrtcvad](https://github.com/wiseman/py-webrtcvad) — WebRTC voice activity detection
- [sounddevice](https://python-sounddevice.readthedocs.io/) — cross-platform audio I/O

---

## License

MIT
