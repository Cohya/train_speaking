# train_speaking

A local, browser-based interview simulator and speaking coach. You paste a job description,
an animated interviewer asks you spoken questions, you answer out loud, and at the end you
get a report on how you spoke: pace, pauses, filler words, and (with an LLM configured)
grammar, vocabulary, clarity, tone, and answer structure, with quotes from your own answers.

Speech recognition (faster-whisper), speech synthesis (Kokoro), and voice activity detection
(Silero) run locally. The LLM is optional and can be local (Ollama) or a cloud provider.

## Requirements

- Windows, Linux, or macOS with Python 3.11 or 3.12 (Kokoro does not support 3.13).
- A Chromium-based browser or Firefox with a microphone. Headphones are recommended.
- For real-time speech: an NVIDIA GPU with recent drivers. Without CUDA, speech recognition
  falls back to the smaller `small.en` model on the CPU, which is slower.
- About 2.5 GB of disk space for model weights, downloaded into `./models` on first start.

## Setup

```powershell
python -m venv .venv
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"   # optional: keep pip's cache inside the project
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# CUDA build of torch first (the default PyPI wheel on Windows is CPU-only).
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\python.exe -m pip install -e ".[speech,dev]"
```

On Linux or macOS use `.venv/bin/python` and skip the CUDA index URL if it does not apply.

If your venv is based on Anaconda's Python on Windows, the app loads the system's Microsoft
C++ runtime before importing torch, because the copy Anaconda bundles is too old for recent
torch builds (symptom: `WinError 1114 ... c10.dll`). No action is needed.

## Run

On Windows, double-click **Start Interview Simulator.bat** in this folder. A console window
opens, then the browser opens at <http://127.0.0.1:8080>. Leave that window open while you
practice, and close it when you are finished.

From a terminal, the same command is:

```powershell
.\.venv\Scripts\python.exe -m train_speaking
```

The first start downloads the Whisper and Kokoro weights; later starts load them from
`./models`.

1. Paste or upload (.txt, .md, .pdf) a job description, and optionally add interviewer rules.
2. Choose the duration, style, focus, and evaluation criteria, and give microphone consent.
3. Press **Start**. Answer each question out loud. A pause (1.2 s by default, configurable)
   ends your answer, or press **I'm done answering**. You can also type an answer.
4. After the interview, review the report, replay recorded answers (if you consented to
   recording), save or delete the session, download Markdown or JSON, or practice a single
   question again and compare.

## Configuration

Settings come from environment variables with the `TS_` prefix or from a `.env` file in the
project root. See [`.env.example`](.env.example) for every option.

### Enabling an LLM

Without an LLM, the interviewer uses a built-in question bank and the report includes only
measured categories (pace, pauses, filler words, lexical diversity).

Local, with [Ollama](https://ollama.com) installed separately:

```text
ollama pull qwen2.5:14b
TS_LLM_PROVIDER=openai_compatible
TS_LLM_BASE_URL=http://localhost:11434/v1
TS_LLM_MODEL=qwen2.5:14b
```

Any OpenAI-compatible cloud endpoint works with `TS_LLM_BASE_URL`, `TS_LLM_MODEL`,
`TS_LLM_API_KEY`, and `TS_LLM_IS_CLOUD=true`. In that mode the setup page requires explicit
consent before transcript text is sent to the provider. Set `TS_COST_PER_1K_*` to your
provider's prices to see a running cost estimate. Each session has a token budget
(`TS_SESSION_TOKEN_BUDGET`); when it is exhausted the interviewer switches to the built-in
question bank.

## How the report is scored

Every category is labeled with how it was scored:

- **Measured**: computed in code from the word timestamps produced by speech recognition.
  This covers pace (words per minute, long pauses), filler words ("um" and "uh" per 100
  words), the pause part of fluency, and lexical diversity (MTLD, which needs 100 or more
  words). Measured scores are skipped when less than 20 seconds of speech or fewer than 30
  words are captured.
- **AI judgment**: scored by the LLM from the transcript (grammar, clarity, tone, structure,
  and parts of fluency and vocabulary). Each judgment cites quotes, and any quote that does
  not appear in your answers is removed and counted in the report.
- **Hybrid**: the average of the measured and AI parts.
- **Not assessed**: categories that cannot be scored honestly with the current setup.

Limitations:

- Filler counts are a lower bound, because speech recognizers often omit "um" and "uh".
  Words such as "like" and "you know" are listed separately and do not count as fillers.
- Pronunciation is **not assessed**. It is not inferred from the transcript or from
  recognizer confidence, which mostly reflect microphone quality and recognizer bias
  against some accents. Accent is never scored. `evaluation/pronunciation.py` defines a
  plug-in interface for a future audio-based assessor; none ships with the app.
- The camera preview is shown only so the session feels like a real interview. Video
  stays in the browser and is never analyzed: there is no facial-expression, emotion, or
  personality inference.

## Privacy

- Audio is processed in memory. Recordings are written only if you consent to recording,
  to `data/tmp/`, which is deleted when the app starts and stops.
- **Save session** copies the transcript, report, and recordings to `data/sessions/<id>/`.
  **Delete session** removes them.
- Transcripts are not logged unless `TS_LOG_TRANSCRIPTS=true`. API keys are never logged.
- The server listens on `127.0.0.1` only. Recorded-audio URLs need a per-run token.

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest                 # unit, integration, and UI tests (fake providers)
.\.venv\Scripts\python.exe -m ruff format src tests
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src tests
```

The real-model smoke test (Kokoro to Silero to Whisper on the GPU) is opt-in:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "gpu and real_providers"
```

Pytest writes its temporary files to `.cache/pytest-tmp`.

Layout (`src/train_speaking/`):

| Package | Responsibility |
| --- | --- |
| `session/` | State machine, orchestrator (turn-taking, timing, pause and resume), storage |
| `voice/` | Audio protocol, VAD, turn detection, sentence splitting, visemes, WebSocket endpoint |
| `providers/` | STT, TTS, and LLM adapters plus fakes for tests |
| `interviewer/` | Interview planning, question policies, LLM and scripted interviewers |
| `evaluation/` | Metrics, filler detection, rubric, LLM judge, evidence validation, scoring |
| `reporting/` | Report assembly and Markdown export |
| `ui/`, `static/` | NiceGUI pages, browser audio capture and playback, SVG avatar |

## Not implemented

- A 3D or photorealistic avatar, and hosted avatar services. The interviewer is a 2D SVG
  with amplitude-driven or viseme (`TS_AVATAR_MODE=viseme`) lip movement.
- Speech-to-speech realtime APIs. The pipeline is always STT, then LLM, then TTS.
- Pronunciation scoring (see the scoring section above).
- Languages other than English.
