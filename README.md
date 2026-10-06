# peerbots-fast-agentic

This project extends the [nao-fast-agentic](https://github.com/jevonlipsey/nao-fast-agentic) functionality over to PEERBots. While that project connected the Aldebaran Nao robot to ChatGPT so it could listen, generate responses, and speak with physical gestures, this project brings that fast, agentic conversational architecture to the PEERbots platform—specifically tailored for **Lulo**, a physical therapy companion robot. This project builds on that foundation with two goals:

1. **Improve conversational speed and responsiveness.** The pipeline features sub-second response times with token streaming, early emotion and color parsing for the PEERbots face API, and high-performance neural text-to-speech with [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) (accelerated via Apple Silicon MPS GPU) streamed in-memory to a headless audio daemon.
2. **Simplify the setup process.** The original NAO project required a legacy Python 2.7 Conda environment and vendor C++ SDK patching. This project runs on a single, modern Python 3.10+ environment without any legacy SDK dependencies or virtual machines.

---

## How It Works

The pipeline coordinates speech capture, LLM reasoning, face animation, and audio playback in a modular, low-latency streaming loop:

| Component / Script | Role | Environment |
| ------------------ | ---------------------------------------------------------------------- | --------------------------------- |
| `whisper_stt.py` | Listens to the microphone with silence detection and speech-to-text | Python 3.10+ (CoreML on macOS, Whisper fallback) |
| `openai_response.py` | Streams LLM completions, extracts early face emotions/colors, executes MCP tools | Python 3.10+ |
| `peerbots_client.py` | Sends REST API calls to update the PEERbots face, emotion, and glowing color | Python 3.10+ |
| `kokoro_manager.py` | Local neural TTS engine running Kokoro on Apple Silicon MPS | Python 3.10+ / FastAPI |
| `droid_client.py` | In-memory sequential audio streamer to the headless audio speaker daemon | Python 3.10+ |

On macOS, speech-to-text uses a compiled Swift CoreML worker (FluidAudio) for fast local transcription, with Google Speech Recognition and local Whisper as automatic fallbacks.

---

## 1. Prerequisites

You will need the following installed on your machine before running the setup scripts.

### All Platforms

| Tool | What it does here | Install |
| ---- | ----------------- | ------- |
| Python 3.10+ | Runs the entire pipeline and AI libraries | [python.org](https://www.python.org/) or via `brew` / `pyenv` (Python 3.11 recommended) |
| [uv](https://docs.astral.sh/uv/) | Runs Python-based MCP tool servers | See [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) |
| [OpenAI API Key](https://platform.openai.com/api-keys) | Authenticates requests to the LLM | Create an account at [platform.openai.com](https://platform.openai.com/) |
| [PEERbots Account & API Key](https://peerbots.org/) | Authenticates requests to the PEERbots REST API | Create an account at [peerbots.org](https://peerbots.org/) |

### Mac Only

| Tool | What it does here | Install |
| ---- | ----------------- | ------- |
| Xcode Command Line Tools | Provides Swift runtime and toolchain for CoreML STT worker | Run `xcode-select --install` in your terminal |
| [Homebrew](https://brew.sh/) | Package manager used to install PortAudio | See [brew.sh](https://brew.sh/) |
| PortAudio | Audio library required by the `PyAudio` Python package for microphone access | Run `brew install portaudio` |

### Linux Only

| Tool | What it does here | Install |
| ---- | ----------------- | ------- |
| PortAudio | Audio library required by the `PyAudio` Python package for microphone access | Run `sudo apt-get install portaudio19-dev` |

### Windows

No additional platform-specific tools are needed beyond standard Python 3.10+ and PortAudio/PyAudio binaries.

---

## 2. Environment Setup

1. Clone and navigate to this repository:
   ```bash
   cd peerbots-fast-agentic
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` file from the provided example template:
   ```bash
   cp .env.example .env
   ```

4. Open `.env` and fill in your credentials:
   ```ini
   OPENAI_API_KEY="sk-..."
   OPENAI_MODEL="gpt-4.1-nano"
   PEERBOTS_API_KEY="your-peerbots-api-key"
   PEERBOTS_USERNAME="your-peerbots-username"
   ```

---

## 3. Running the Pipeline

Launch the entire pipeline with a single command:

```bash
python main.py
```

Once you see `[[LISTENING]]` in the terminal, Lulo is calibrated and ready to converse.

### CLI Options

| Flag | Purpose | Example |
| ---- | ------- | ------- |
| `--text` | Run in text input mode without using a microphone | `python main.py --text` |
| `--list-mics` | List all available microphone input devices and their indices | `python main.py --list-mics` |
| `--mic-index <N>` | Use a specific microphone device index | `python main.py --mic-index 2` |
| `--pause-threshold <S>` | Seconds of silence before speech is finalized (default `1.2`s) | `python main.py --pause-threshold 1.0` |
| `--no-droid` | Disable external Droid audio streaming and use PEERbots tablet audio directly | `python main.py --no-droid` |

---

## 4. Audio Pipeline & TTS Configuration

The pipeline supports two output modes for audio and speech:

1. **High-Performance Streaming (Kokoro + Droid Daemon):**
   - **TTS:** Synthesizes speech locally via [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) on Apple Silicon MPS GPU (~300ms per sentence).
   - **Voice:** Supports blended voices via `KOKORO_VOICE` (default: `af_heart:0.6+af_bella:0.4` for Lulo's warm, upbeat character).
   - **Playback:** Streams raw in-memory audio bytes directly over the local network / Tailscale to the headless Droid audio daemon, bypassing tablet speaker latency.
   - **Face Animation:** Updates PEERbots face emotion and glowing color silently so the tablet never duplicates speech audio.

2. **Portable Fallback (Direct PEERbots Tablet Audio):**
   - If Kokoro or the Droid audio daemon is unreachable, the system automatically falls back to sending speech directly to the PEERbots REST API, having the tablet speak the line.

---

## 5. Configuration & Customization

### Persona & Dialogue Rules
Lulo's personality, physical therapy companion persona, deer injury backstory, and pain escalation rules are defined in `config/system_prompt.md`.

### Adding MCP Tools
External tool capabilities are defined in `mcp_config.json` at the project root. To add a new tool server:

```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  }
}
```

The system automatically discovers and registers MCP tools during startup.

---

## Project Structure

```
peerbots-fast-agentic/
  main.py                  # orchestrator, coordinates STT, LLM streaming, Kokoro, Droid & Peerbots
  mcp_config.json          # MCP tool server configuration
  requirements.txt         # Python 3.10+ dependencies
  .env                     # API keys and configuration (not checked in)
  .env.example             # template environment file
  config/
    system_prompt.md       # Lulo PT companion persona, dialogue tree, and response rules
  scripts/
    whisper_stt.py         # multi-tier STT (CoreML Swift worker -> Google SR -> Whisper)
    openai_response.py     # streaming LLM cognitive layer with structured emotion/color parsing
    peerbots_client.py     # PEERbots REST API client (face emotions, colors, speech)
    kokoro_manager.py      # Kokoro-FastAPI lifecycle manager and in-memory neural TTS synthesizer
    droid_client.py        # headless audio daemon client and gapless sequential playback queue
    lib/
      file_utils.py        # safe file I/O helpers
      mcp_loader.py        # dynamic MCP server loader
  stt-coreml/              # Swift CoreML speech-to-text worker (macOS)
  state/                   # conversation history and runtime state files
  skills/                  # agent skills directory
```
