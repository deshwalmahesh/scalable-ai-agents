# Scalable AI Agents

**Production-grade, end-to-end AI systems built for scale.**

This repo demonstrates how to architect and implement production-ready AI agents that handle real-world engineering concerns: **concurrency**, **security**, **state management**, **authentication**, **error handling**, **observability**, **multi-tenancy**, **async processing**, and **horizontal scalability** — no shortcuts, no fancy over-engineering.

Each sub-project is a **standalone, deployable system** solving a specific problem domain with production-level rigor.

---

## Projects

### 💻 [General Coding Agent — On-Device Software Engineer](./general-coding-agent)
**Autonomous multi-agent pair programmer with cryptographic state safety, boundary sandboxing, and zero external infra dependencies.**

On-device coding agent that inspects, reasons, edits, and tests codebases locally. Built on LangGraph, it runs completely standalone on embedded SQLite, isolating planning from code execution while enforcing strict filesystem and shell safety boundaries.

**Engineering highlights:**
- **Zero External Infra**: Embedded SQLite (`AsyncSqliteSaver` + `AsyncSqliteStore`) — no Redis, no external vector database servers, no Docker required for basic execution.
- **Hierarchical Multi-Agent Architecture**: Supervisor routes between an isolated `planner` (read-only) and `coder` (execution).
- **Pre-Flight AST & OS Sandboxing**: Pre-flight syntax validation prevents command substitutions and background escapes; macOS Seatbelt / Linux Bubblewrap jail mutations strictly inside `workspace_root`.
- **Fuzzy Whitespace-Tolerant Patching**: `fs_edit` handles trailing whitespace differences and nested indentation shifts with zero code corruption, failing closed on ambiguity.
- **Cryptographic HIL Approvals**: Human-in-the-loop approval tickets are bound to canonical argument hashes (`canonical_args_hash`) to prevent tampering across graph pause/resume cycles.
- **Deterministic Secret Rejection**: Automatically detects and blocks API keys, AWS credentials, and tokens from entering persistent memory.
- **Bounded Non-Destructive Compaction**: Two-tier memory projection compacts older turns when token watermarks are reached, preserving full message and tool fidelity.
- **Comprehensive Validation**: 44 end-to-end integration and unit tests covering sandboxing, compaction, recovery, and tool execution.

**Stack:** Python 3.12 · LangGraph · LangChain · SQLite (`aiosqlite`) · Any LLM (Gemini / Claude / OpenAI / Local vLLM/Ollama)

[→ README + architecture diagram](./general-coding-agent/README.md)

---

### 📄 [DocuMind — Text RAG Agent](./text-rag-agent)
**Document-grounded conversational assistant with live grounding evaluation.**

Multi-user RAG system that answers questions strictly from uploaded documents (PDF/DOCX/DOC/TXT/MD), with inline citations, line-level source highlighting, and realtime LLM-as-judge grounding verdicts.

**Engineering highlights:**
- Multi-tenant JWT auth, per-user document + chat isolation
- Async ingestion pipeline (upload → queue → worker → vectorstore)
- Hybrid retrieval (dense + BM25 sparse, RRF fusion in Qdrant)
- SSE streaming with token-by-token citations
- Stateless, horizontally scalable (one Docker image: SPA + API)
- Per-user quota, concurrency caps, rate limiting
- Human-readable errors, structured logs, Prometheus metrics
- Offline + online evaluation harness (LLM-as-Judge)

**Stack:** FastAPI · Qdrant · Redis · Postgres · arq worker · React · Gemini (OpenAI-compatible)

[→ README + architecture diagram + screenshot](./text-rag-agent/README.md)

---

### 📞 [Voice Agent — Phone Call AI](./voice-agent)
**Live voice AI for phone calls with RAG-grounded policy Q&A.**

PSTN/SIP-ingress voice agent that answers caller questions by retrieving from a policy knowledge base, streaming sub-600ms round-trip responses over LiveKit.

**Engineering highlights:**
- Full telephony stack: Twilio SIP → LiveKit → STT (Deepgram) → LLM → TTS (Cartesia/Deepgram)
- Bounded concurrency, session-level state isolation
- Chroma vectorstore for policy document retrieval
- Graceful error handling, circuit breakers, metrics
- Docker Compose + single-command local dev
- Smoke + load + multi-turn E2E validation harness

**Stack:** LiveKit · Twilio · Deepgram · Chroma · FastAPI · Redis · OpenAI-compatible LLM

[→ README + system flow diagram](./voice-agent/README.md) | [System Flow Design](https://github.com/user-attachments/assets/e8455307-6379-451d-baa9-2939ec62e0c3)

---

## Philosophy

**"Scalable" means production-ready:**
- Async-first, stateless where possible
- Proper auth, multi-tenancy, quota enforcement
- Graceful degradation, load shedding, circuit breakers
- Centralized config, env-driven, Docker/K8s-ready
- Observable: structured logs, metrics, health checks
- Testable: unit + integration + smoke + E2E
- Deployable: Railway/Render/Fly/K8s with zero code changes

**No bloat.** Stdlib first, boring tech, minimal abstractions. Complex where justified (bounded concurrency, multi-stage pipelines, adversarial eval), simple everywhere else.

---

## Quick Start

Each project is self-contained — navigate to its directory and follow its README.

```bash
# General Coding Agent (On-Device)
cd general-coding-agent
cp .env.example .env # fill in GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY
uv sync
uv run python main.py

# Text RAG Agent
cd text-rag-agent
export GEMINI_API_KEY=your_key
make up              # → http://localhost:8000

# Voice Agent
cd voice-agent
cp .env.example .env # fill in LiveKit, Twilio, Deepgram, LLM keys
uv sync
uv run voice-agent-seed-kb
sh scripts/start.sh both
```

---

## License

MIT
