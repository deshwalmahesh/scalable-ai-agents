# Voice Agent

This repo is a phone-call voice agent built around:

- LiveKit for call rooms, media transport, and agent dispatch
- Twilio for PSTN/SIP ingress
- Deepgram for STT
- Cartesia or Deepgram for TTS
- an OpenAI-compatible LLM endpoint for text generation
- Chroma plus embeddings for policy-document retrieval
- FastAPI for control-plane endpoints

It answers policy questions on a live call by:

1. receiving a phone call
2. turning caller audio into text
3. grounding policy questions against the knowledge base
4. generating a reply or using a direct grounded answer
5. turning the reply back into audio

## System Flow Design
<img width="5481" height="5612" alt="system_flow_diagram" src="https://github.com/user-attachments/assets/e8455307-6379-451d-baa9-2939ec62e0c3" />


## What This Code Does

There are 4 main parts.

1. API control plane
- file: [`src/voice_agent/api.py`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/src/voice_agent/api.py)
- exposes:
  - `/healthz`
  - `/readyz`
  - `/metrics`
  - `/kb/search`
  - `/telephony/sip/status`
  - `/telephony/sip/provision`

2. Voice worker
- file: [`src/voice_agent/worker.py`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/src/voice_agent/worker.py)
- joins LiveKit jobs
- runs the live voice pipeline
- records latency/failure metrics

3. Knowledge base and retrieval
- files:
  - [`src/voice_agent/kb.py`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/src/voice_agent/kb.py)
  - [`src/voice_agent/retrieval.py`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/src/voice_agent/retrieval.py)
  - [`src/voice_agent/seed.py`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/src/voice_agent/seed.py)
- ingests the source PDF into Chroma
- supports local or remote embeddings
- returns grounded excerpts for policy questions

4. Validation and smoke/load tools
- files:
  - [`src/voice_agent/livekit_smoke.py`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/src/voice_agent/livekit_smoke.py)
  - [`src/voice_agent/load_test.py`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/src/voice_agent/load_test.py)
  - [`scripts/start.sh`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/scripts/start.sh)

## Runtime Flow

The current happy path is:

`Caller -> Twilio SIP -> LiveKit room/dispatch -> worker -> STT -> retrieval if needed -> LLM or direct grounded answer -> TTS -> caller`

More detail:

1. Twilio forwards the call to LiveKit over SIP.
2. LiveKit creates a room and dispatches the room to the worker.
3. The worker receives live audio.
4. VAD/endpointing decides when the caller has finished the turn.
5. Deepgram STT converts speech to text.
6. The worker decides whether the turn needs policy grounding.
7. If it does, the worker queries Chroma through the retrieval boundary.
8. The worker either:
   - returns a direct grounded answer, or
   - sends grounded context to the LLM.
9. TTS converts the final reply to audio.
10. The audio is sent back into the live call.

Architecture and flow diagrams live in:
- [`docs/ARCHITECTURE.md`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/docs/ARCHITECTURE.md)

## Requirements

- Python `3.12`
- `uv`
- LiveKit project
- Twilio account for PSTN/SIP proof
- Deepgram API key
- OpenAI-compatible LLM API key/base URL
- optional Cartesia API key if using `TTS_PROVIDER=cartesia`

## Environment Setup

1. Install dependencies:

```bash
uv sync --extra dev
```

2. Copy the example env file:

```bash
cp .env.example .env
```

3. Fill in the required values in `.env`.

Minimum required values:
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `DEEPGRAM_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Common local defaults already handled in code:
- `VOICE_RUNTIME_MODE=cascade`
- `CHROMA_PROVIDER=local`
- `EMBEDDING_PROVIDER=local`
- `API_PORT=8080`
- `WORKER_PORT=8081`
- `WORKER_MAX_CONCURRENT_SESSIONS=1`

Important optional values:
- `TWILIO_PHONE_NUMBER`
- `TWILIO_SIP_DOMAIN`
- `CARTESIA_API_KEY`
- `REDIS_URL`
- `KB_SOURCE_PDF_URL`

## Seed The Knowledge Base

Before running the agent, seed Chroma from the PDF:

```bash
uv run voice-agent-seed-kb
```

What this does:
- downloads the configured policy PDF
- chunks it
- embeds it
- stores the chunks in the configured Chroma collection

## Run The System Locally

### Option 1: Run API and worker manually

Start the API:

```bash
uv run voice-agent-api
```

Start the worker in another terminal:

```bash
WORKER_PORT=0 uv run voice-agent-worker start
```

Notes:
- `WORKER_PORT=0` lets the worker choose a free HTTP port
- the API uses `API_PORT` from `.env`, default `8080`

### Option 2: Use the helper script

Run both API and worker:

```bash
sh scripts/start.sh both
```

Run only the API:

```bash
sh scripts/start.sh api
```

Run only the worker:

```bash
sh scripts/start.sh worker
```

## Run With Docker Compose

Build and start:

```bash
docker compose up --build
```

Current compose services:
- `redis`
- `api`
- `worker`

Notes:
- compose injects `REDIS_URL=redis://redis:6379/0`
- compose mounts `./data` into the container
- `.env` is still required

## Control Plane Endpoints

Health:

```bash
curl http://127.0.0.1:8080/healthz
```

Readiness:

```bash
curl http://127.0.0.1:8080/readyz
```

Metrics:

```bash
curl http://127.0.0.1:8080/metrics
```

KB search:

```bash
curl "http://127.0.0.1:8080/kb/search?q=refund&limit=3"
```

Telephony status:

```bash
curl http://127.0.0.1:8080/telephony/sip/status
```

Provision LiveKit SIP resources:

```bash
curl -X POST http://127.0.0.1:8080/telephony/sip/provision
```

## Validation

### Stable local test suite

Runs unit, integration, and regression tests:

```bash
sh scripts/start.sh test
```

### Live validation

Runs stable tests, starts API and worker, then runs live smoke and E2E tests:

```bash
RUN_SMOKE=1 RUN_E2E=1 sh scripts/start.sh validate
```

### Live multi-turn validation

Runs the quota-sensitive multi-turn live path as well:

```bash
RUN_SMOKE=1 RUN_E2E=1 RUN_MULTI_TURN_E2E=1 sh scripts/start.sh validate
```

## Useful Commands

Seed KB:

```bash
uv run voice-agent-seed-kb
```

Single-turn live smoke:

```bash
uv run voice-agent-livekit-smoke
```

Load test harness:

```bash
uv run voice-agent-load-test
```

## Current Limits

This repo is working, but some of the intended scale/latency goals are still not proven.

Still not proven:
- `<600 ms` average round-trip across `100` calls
- `100` concurrent calls
- final recorded demo video
- final polished quality-evidence package

Current status is tracked in:
- [`TO_DO.md`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/TO_DO.md)
