# Voice Agent — TODO

## Production Best Practices (Holy Grail)

This is the standing production-grade checklist for this repo. It is intentionally biased toward **system architecture, runtime behavior, concurrency control, and observability**, not vendor marketing or casual model swapping. A practice only belongs here if it is both:

1. repeatedly supported by strong sources, and
2. actually applicable to this codebase.

**Status legend**
- `FOLLOWING`: already materially present in the current system
- `NEXT`: high-impact next production upgrade for this repo
- `LATER`: valid practice, but not yet justified ahead of the current bottlenecks

| Area | Production best practice | Why it matters for this repo | Status | Where it lives / should land |
|---|---|---|---|---|
| Turn-taking / latency | Treat **mouth-to-ear latency** as the real UX metric, and break it down by stage (`EOU`, `LLM TTFT`, `TTS TTFB`) | Voice UX breaks on accumulated pipeline delay, not one slow function; our current biggest measured gaps are still `EOU` and `LLM TTFT` | `FOLLOWING` | `worker.py`, `observability.py`, live smoke / E2E |
| Turn-taking / latency | Treat **end-of-turn detection** as a first-class subsystem, not “just wait for silence” | A long pause, hesitation, spelling, or network delay can either cut the caller off or make the agent feel slow; this is one of the biggest remaining structural latency levers in the repo | `NEXT` | `worker.py`, `livekit_smoke.py`, `tests/e2e/test_voice_flow.py` |
| Live path design | Keep the **live-turn critical path thin**; avoid extra model/tool loops on every utterance | Real-time voice breaks when orchestration grows faster than the user’s patience; bounded retrieval and direct grounded injection are the right current shape | `FOLLOWING` | `worker.py`, `kb.py`, `retrieval.py` |
| Retrieval | Use **bounded top-K retrieval** with compact grounded context; do not scan or overstuff the prompt | This preserves accuracy without adding avoidable latency or token noise; current retrieval bugs already proved that too much context can hurt correctness | `FOLLOWING` | `kb.py`, `retrieval.py`, grounded prompt path in `worker.py` |
| Failure handling | Separate **caller-visible failure UX** from backend retry logic | A caller needs a fast reprompt, wait message, or clean fallback, not silent retry storms; backend recovery and caller experience are different problems | `FOLLOWING` | `worker.py`, failure policy around session/runtime events |
| Retries / timeouts | Put **timeouts on every remote call**, and retry at **one layer only** with bounded backoff/jitter where the user won’t hear dead air | Retry multiplication is how small outages turn into system-wide overload; voice calls punish long hidden retries especially hard | `NEXT` | provider clients, `worker.py`, control-plane calls, not in the live response loop |
| Concurrency | Scale with a **worker pool and explicit admission control**, not by blindly stuffing more sessions into one worker | Voice concurrency is session-based, long-lived, and stateful; the repo already proved one worker with one safe slot becomes the first bottleneck | `NEXT` | deployment/startup layer, `worker.py`, `scripts/start.sh`, real orchestrator later |
| Shared state | Use **Redis or equivalent atomic ephemeral state** for active sessions / leases instead of local files | File-backed counters are fine for local proof, but not for multi-worker or multi-host production concurrency | `FOLLOWING` | `observability.py`, `worker.py`, `api.py`, `docker-compose.yml` |
| Metrics | Use **Prometheus multiprocess mode or external telemetry** instead of local JSONL fan-in once multiple workers matter | Per-process Python metrics need a production aggregation shape; local JSONL merging is a dev convenience, not a scale story | `NEXT` | `observability.py`, `api.py` |
| Isolation | Keep failures **session-local whenever possible** and let jobs drain cleanly on shutdown | One bad call should not poison other calls; LiveKit’s job isolation is already the right primitive, and we should preserve that design | `FOLLOWING` | LiveKit worker/job model, `worker.py`, startup/shutdown path |
| Networking | Reuse connections, avoid DNS/TLS work in the hot path, and keep telephony/media + inference **geographically close** where possible | In voice systems, small network costs stack up every turn; connection setup and cross-region hops are real latency contributors | `LATER` | provider client construction, deployment topology |
| External systems | For slow external lookups, use **short wait messages**, prefetch predictable data early, and push very long work off the live loop | If a dependency takes seconds, the caller needs feedback; if a lookup is always needed, do it in parallel before it blocks the answer path | `LATER` | future tool/external-API boundary in `worker.py` / service layer |
| Evaluation | Validate with **real audio**, multi-turn conversations, objection flows, interruption, and stepped concurrency; do not trust easy tests or one canned prompt | Production voice breaks on turn-taking, long conversations, and concurrency edges, not just on “does this one answer sound right” | `FOLLOWING` | `livekit_smoke.py`, `tests/e2e/test_voice_flow.py`, `load_test.py` |
| Evaluation | Measure **session-based concurrency**, not just request throughput | Voice sessions are persistent and long-lived; concurrency failure is about active streams and admission, not simple RPS | `NEXT` | `load_test.py`, worker-pool deployment proof |
| Memory / multi-turn | Keep **session-local context engineering** explicit; do not jump to global memory before multi-turn follow-up handling is proven | Most real gains come from better follow-up grounding and session continuity before adding complex persistent memory systems | `NEXT` | `worker.py`, retrieval query assembly, future session-summary helper |

**Why these practices were chosen**
- They show up repeatedly in the strongest sources: AWS on timeouts/retries/jitter, Twilio on caller-facing failover and latency framing, Deepgram on end-of-turn measurement, LiveKit on worker pools and job isolation, Redis on atomic counters, Prometheus on multiprocess metrics, and Chroma on bounded query patterns.
- Secondary engineering write-ups from Cresta, Daily, and Speechmatics were only used where they matched the same themes:
  - session-based concurrency is different from web throughput
  - turn-taking is a first-class problem
  - realistic evaluation must include long, multi-turn, audio-based runs
  - the live response loop should stay thin and deterministic
- These are the practices that most directly fit this repo’s current bottlenecks:
  - `EOU` delay
  - worker-pool / admission limits
  - local file-backed shared state
  - multi-process metrics aggregation
  - realistic multi-turn / concurrency proof

**Open-source implementation choices for this repo**
- We are not adopting AWS or any other paid cloud dependency just because a best-practice document mentions a useful rule.
- Use strong paid-vendor docs only as engineering guidance when the rule is generic and portable:
  - example: timeouts on remote calls
  - example: retries at one layer only
  - example: jittered backoff off the live-turn path
- Prefer open-source infrastructure when the current system actually needs it:
  - `Redis OSS` for shared ephemeral session state and leases
  - `Prometheus` for metrics, with Python multiprocess mode once multiple workers matter
  - `Grafana` later if dashboards are worth the operational cost
  - `Docker Compose` as the first orchestration step for local production-like validation
- Do **not** spin up new infra just because it sounds production-grade.
  - Add it only when the current local implementation becomes the real blocker.
  - Right now that applies to:
    - file-backed active-session state
    - cross-process metrics aggregation
    - multi-worker local proof
- Telephony nuance:
  - truly public PSTN numbers are usually not free/open-source
  - but local/internal SIP testing can use open-source components such as:
    - `Asterisk`
    - `FreeSWITCH`
    - `Kamailio`
    - `OpenSIPS`
  - for this assignment, Twilio is still acceptable for PSTN proof, but it should not drive the rest of the architecture

**What this means in practice**
- `NEXT` infra additions should stay minimal:
  - add `Redis` only when replacing the file-backed active-session counter
  - add Prometheus multiprocess wiring only when multi-worker metrics need to be trustworthy
  - use Docker Compose to run those locally when that work starts
- `NOT NEEDED YET`:
  - Kubernetes
  - managed cloud caches
  - managed observability stacks
  - message brokers
  - service mesh
  - distributed workflow systems

**Sources**
- AWS Builders Library: [Timeouts, retries and backoff with jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)
- Twilio: [Core latency in AI voice agents](https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents)
- Twilio: [Programmable Voice failover best practices](https://www.twilio.com/docs/voice/twilio-voice-failover-best-practices)
- Deepgram: [Flux quickstart](https://developers.deepgram.com/docs/flux/quickstart)
- Deepgram: [Measuring STT latency](https://developers.deepgram.com/docs/measuring-streaming-latency)
- LiveKit: [Self-hosted deployments](https://docs.livekit.io/deploy/custom/deployments/)
- LiveKit: [Job lifecycle](https://docs.livekit.io/agents/server/job/)
- Redis: [INCR](https://redis.io/docs/latest/commands/incr/)
- Prometheus Python client: [Multiprocess mode](https://prometheus.github.io/client_python/multiprocess/)
- Chroma: [Query and Get](https://docs.trychroma.com/docs/querying-collections/query-and-get)
- Cresta: [Engineering for real-time voice agent latency](https://cresta.com/blog/engineering-for-real-time-voice-agent-latency)
- Daily / Pipecat: [Benchmarking LLMs for voice agent use cases](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/)
- Speechmatics: [How we built real-time concurrency for Voice AI at scale](https://www.speechmatics.com/company/articles-and-news/how-we-built-real-time-concurrency-for-voice-ai-at-scale)

## Status: What's Done vs What's Needed

### DONE
- [x] LiveKit Agents SDK integration (cascade mode: Deepgram STT → LLM → Deepgram TTS)
- [x] Gemini Live mode (experimental alternative)
- [x] Knowledge base with hybrid search (Chroma + Jina/local embeddings)
- [x] PDF ingestion pipeline (download → parse → chunk → embed → store)
- [x] Barge-in / interruption support (VAD-based turn detection)
- [x] Observability: per-turn latency breakdown (EOU + LLM TTFT + TTS TTFB)
- [x] Prometheus metrics endpoint
- [x] FastAPI control plane (health, readiness, metrics, KB search)
- [x] Smoke tests (Deepgram, LiveKit E2E)
- [x] Load test harness (concurrent sessions)
- [x] Retry logic, SSL handling
- [x] Unit / integration / regression / e2e test structure
- [x] Local vs remote embedding path switch, tested with the same Jina model
- [x] Local vs remote Chroma switch, tested with the same rebuilt corpus
- [x] Retrieval boundary extracted behind `RetrievalService`
- [x] `on_user_turn_completed(...)` grounding injection replaces the old tool-call shape
- [x] Split-final-transcript reconstruction for grounded turns
- [x] Lightweight retrieval gate for obvious greetings / connection checks
- [x] Active-session accounting shared between worker and API metrics
- [x] Worker capacity/load reporting tied to real active jobs
- [x] Cartesia TTS path wired and validated end to end
- [x] Cascade default validated on `gemini-2.5-flash`
- [x] Twilio SIP trunk path validated end to end with Twilio Origination test call
- [x] LiveKit inbound SIP trunk + dispatch rule provisioned from app code
- [x] Worker pickup and answer of Twilio inbound test call proven via logs
- [x] Prompt trimmed to the grounded-context-only shape (no tool-calling instructions)
- [x] Explicit upstream failure metrics for retrieval / session start / session runtime errors
- [x] Optional explicit LLM failover path (config-gated, not silent)
- [x] STT timeout classification for missing final transcript
- [x] Load-test failure reporting now surfaces real failure examples instead of swallowing causes
- [x] One-script validation flow via `scripts/start.sh test|validate`
- [x] Validation runs now use isolated temp runtime state instead of shared stale session/metrics files
- [x] Deprecated `aiohttp` session subclass patch removed from worker
- [x] API startup no longer eagerly warms the local embedding model
- [x] Worker prewarm/startup materially reduced by removing unnecessary API-side model warmup
- [x] Live validate shutdown no longer emits the previous leaked-semaphore warning after API-side eager model warmup removal
- [x] False-positive TTS no-audio watchdog removed; runtime now relies on actual provider/session TTS errors instead of noisy timeout guesses
- [x] Dedicated live multi-turn audio smoke path added in the existing LiveKit harness
- [x] Worker startup now resets stale local active-session counter state so logs/metrics do not inherit dead sessions from a previous run
- [x] Load harness now uses per-query evaluation expectations instead of grading every query with the refund-policy keyword set
- [x] Local two-worker startup was proven after fixing the file-backed active-session-store temp-file race
- [x] Redis-backed active-session state added for multi-worker/shared-session counting, with file fallback retained for no-Redis local runs
- [x] Docker Compose now includes local Redis for production-like shared session-state validation
- [x] STT timeout path is now more conservative for longer partial utterances and resets escalation after a successful final transcript
- [x] Explicit startup greeting removed from the live startup path so concurrent sessions are not blocked by ceremonial first-turn TTS
- [x] OpenAI-compatible live-turn LLM calls now honor `HTTP_TIMEOUT_SECONDS` and default to `AGENT_MAX_RETRIES=0` so provider overload fails fast instead of silently holding a call slot
- [x] Local direct-answer path for reinstatement / objection facts is now regression-tested so the remaining objection blocker is narrowed to the live answer-generation boundary, not basic retrieval
- [x] `llm_node(...)` now recovers grounded direct answers when turn-scoped guidance is missing, so the live reply path no longer depends entirely on `on_user_turn_completed(...)` having populated the chat context first
- [x] Live single-turn reinstatement / objection smoke now passes with the grounded `two years` fact and no forbidden `three` / `five` drift
- [x] Worker prewarm now calls direct local-model warmup in local-embedding mode; the code path is validated locally, and live latency impact still needs re-checking
- [x] Live smoke results now include actual RTP-quality evidence from the media track (`jitter_ms`, `packet_loss_fraction`, `packets_received`, `packets_lost`, `estimated_mos`) instead of leaving MOS / jitter / loss as control-plane-only metrics
- [x] Unsupported policy questions with zero grounded matches now use a deterministic short safe answer (`I can't answer that from this policy document...`) instead of relying on free-form LLM paraphrase
- [x] Live multi-turn smoke now returns a session-grade artifact (`turns` + `quality_summary`) so one run can show per-turn correctness, latency, and media-quality evidence together

### NOT DONE (Assignment Requirements)
- [ ] Sub-600ms average round-trip latency (THE critical requirement)
- [ ] Demonstrate 100 concurrent calls
- [ ] Handle at least one objection scenario gracefully with explicit proof
- [ ] Recorded demo video (concurrent calls, latency run, barge-in)
- [ ] Submission-grade MOS / jitter / packet loss evidence across the final proof set

### Current Proven State

**Assignment gap summary**

| Requirement | Status | What proves it | What is still missing / why |
|---|---|---|---|
| Inbound PSTN via SIP trunk | DONE | Twilio test call completed with `200 OK`; LiveKit room created; worker transcribed and answered | A real caller PSTN experience is optional extra proof, not the main blocker anymore |
| Real-time media routing into LiveKit | DONE | Twilio trunk -> LiveKit SIP URI -> LiveKit inbound trunk -> dispatch rule -> worker path is proven | None for inbound proof |
| Domain KB questions | PARTIAL | successful grounded refund and reinstatement single-turn live smoke runs, deterministic unsupported-query refusal, actual-call quality capture in live smoke, and a harder 4-turn live conversation artifact with supported + unsupported turns in one session | the latest live 4-turn run still failed on the reinstatement turn (`five years` drift), so multi-turn correctness is not stable enough to call finished |
| Barge-in / interruption | PARTIAL | supported in cascade path | still needs deliberate demo evidence |
| Objection handling | PARTIAL | live single-turn objection smoke returns the grounded `two years` fact with no forbidden drift, and the multi-turn artifact now exercises objection + unsupported handling in one session | objection handling is still not submission-complete because the latest 4-turn live run regressed to `five years` on the reinstatement turn |
| Latency instrumentation | PARTIAL | EOU, LLM TTFT, TTS TTFB, JSONL logs, Prometheus metrics | the required latency target is still far from met |
| MOS / jitter / packet loss | PARTIAL | actual single-turn live smoke now returns measured `jitter_ms`, `packet_loss_fraction`, `packets_received`, `packets_lost`, and `estimated_mos`; latest proven single-turn refund smoke showed `jitter_ms=6.0`, `packet_loss_fraction=0.0`, `estimated_mos=4.28` | still needs a polished final evidence set across the submission runs, not just one successful smoke |
| Failure recovery mechanism | PARTIAL | explicit retrieval degradation, optional LLM failover, session/runtime failure metrics, STT timeout, and provider-reported TTS/session failures | still needs a live forced-failure proof and clearer operator-facing recovery story |
| 100 concurrent calls | NOT PROVEN | load harness exists; truthful stepped runs now include `2` workers and `4` workers, with the best current live point at `3/4` success for `concurrency=4` and `avg_ms=9389.61` | topology is no longer the only blocker; the next job is to stabilize multi-turn/turn-boundary behavior and then continue stepped live load honestly |
| Final artifacts | PARTIAL | `README.md`, `Dockerfile`, `docker-compose.yml`, `scripts/start.sh`, `docs/ARCHITECTURE.md` now exist | demo video and polished final metric evidence are still missing |

**Deliverables now present in repo**
- setup instructions: `README.md`
- container/startup artifacts:
  - `Dockerfile`
  - `docker-compose.yml`
  - `scripts/start.sh`
- architecture/documentation artifact:
  - `docs/ARCHITECTURE.md`
    - system architecture diagram
    - call-flow diagram
    - scaling plan
    - LiveKit vs Pipecat comparison
    - 1-pager on 1000-call failure modes and current bottlenecks
- reproducible validation entrypoints:
  - `sh scripts/start.sh test`
  - `RUN_SMOKE=1 RUN_E2E=1 sh scripts/start.sh validate`

**Why this matters**
- The repo now contains the assignment's non-demo setup and architecture deliverables.
- The remaining missing items are now mostly proof artifacts, not missing documentation scaffolding.

**Twilio / SIP proof (real evidence)**
- Twilio trunk exists: `TK23f542fae32cd3ab17fe7c8ca2a8b89f`
- Twilio number attached: `+16627075141`
- Twilio Origination URI: `sip:2m046c1ockr.sip.livekit.cloud;transport=tcp`
- LiveKit inbound trunk exists: `ST_JwaSkyagLEgq`
- LiveKit dispatch rule exists: `SDR_JYyiHJBTsPQN`
- Twilio verification call completed with `200 OK`
- LiveKit room `RM_n4Ci3KgWHazJ` / `call-_hello_uX5g5oPqwPwz` had `2` participants
- Worker logs showed transcript of the Twilio verification prompt and an assistant reply

**Why this matters**
- PSTN ingress is now proven.
- The assignment is no longer blocked by telephony plumbing.
- The remaining hard blockers are latency, resilience, concurrency proof, and final deliverables.

**Current resilience / failure-handling state**
- bad config still fails fast on startup
- retrieval failure degrades safely instead of hallucinating policy facts
- optional LLM fallback exists, but only when explicitly configured
- live-turn OpenAI-compatible LLM calls now use `HTTP_TIMEOUT_SECONDS` directly and default to `0` retries through `AGENT_MAX_RETRIES`
- session start failures are counted explicitly
- session runtime errors are classified by upstream component
- STT stalls now emit `stt_timeout`
- load-test failures now surface real example causes

**Why this matters**
- The system no longer hides several important failure modes.
- The assignment's resilience requirement is now partially satisfied in code.
- What is still missing is a live forced-failure proof and a clearer operational runbook.

**Current turn-completion behavior**
- the agent currently decides a caller turn is complete through the LiveKit voice pipeline, not through a custom hand-rolled silence detector
- current path:
  - `turn_detection="vad"` in `worker.py`
  - `min_endpointing_delay` / `max_endpointing_delay` in `SupportAgent`
  - Deepgram STT interim/final transcript events
- why this matters:
  - a long pause can finalize too early
  - a delayed final transcript can hold the response too long
  - network/provider jitter can make the turn boundary feel wrong even when STT itself is accurate
- where to validate it:
  - `worker.py`
  - `livekit_smoke.py`
  - `tests/e2e/test_voice_flow.py`

**Current validation state**
- stable validation path:
  - `sh scripts/start.sh test`
  - runs unit + integration + regression
- live validation path:
  - `RUN_SMOKE=1 RUN_E2E=1 sh scripts/start.sh validate`
  - runs stable suite, starts API + worker, runs live smoke and live E2E, then shuts everything down
- dedicated quota-sensitive multi-turn live path:
  - `RUN_SMOKE=1 RUN_E2E=1 RUN_MULTI_TURN_E2E=1 sh scripts/start.sh validate`
  - keeps the deeper caller-like multi-turn audio proof explicit instead of silently breaking baseline validation
- current strongest multi-turn artifact:
  - one 4-turn live session with:
    - refund turn: success
    - grace-period turn: success
    - unsupported CEO question: success with deterministic refusal
    - reinstatement turn: failed with forbidden `five years` drift
  - latest session quality summary:
    - `turn_count=4`
    - `successful_turns=3`
    - `avg_response_latency_ms=8487.43`
    - `worst_jitter_ms=14.0`
    - `worst_packet_loss_fraction=0.0014`
    - `lowest_estimated_mos=4.11`
- validation now uses isolated temp files for:
  - `ACTIVE_SESSIONS_PATH`
  - `TURN_METRICS_PATH`

**Why this mattered**
- shared runtime state was causing false cross-run contamination in live validation
- the validation path now reflects the current run instead of stale previous sessions

**Current default non-Twilio path**
```
Caller audio
  -> LiveKit
  -> Deepgram STT
  -> SupportAgent.on_user_turn_completed(...)
       -> RetrievalService
       -> grounded developer context injected
  -> Gemini text model via OpenAI-compatible endpoint
  -> Cartesia TTS
  -> caller audio
```

**Current default runtime choices**
- `VOICE_RUNTIME_MODE=cascade`
- `EMBEDDING_PROVIDER=local`
- `CHROMA_PROVIDER=local`
- `TTS_PROVIDER=cartesia`
- `GEMINI_LLM_MODEL_NAME=gemini-2.5-flash`

**Latest proven successful default smoke**
- `response_latency_ms`: `9019.31`
- `round_trip_success`: `true`
- worker turn snapshot:
  - `end_of_utterance_delay_ms`: `551.88`
  - `llm_ttft_ms`: `2678.09`
  - `tts_ttfb_ms`: `216.23`
  - `turn_total_ms`: `3446.20`

**Why this matters**
- The system is real and works end to end, including Twilio inbound verification.
- Retrieval is no longer the dominant bottleneck in the best local/local path.
- The main remaining latency is now the **voice/LLM path**, especially `LLM TTFT` and turn-finalization delay.
- API startup is now lighter because it no longer loads the local embedding model just to boot the control plane.

---

## Architecture Direction (What the stronger design actually is)

The current repo is a **working prototype**, not the final scalable shape. It works end to end, but one worker process is still responsible for too much:

`LiveKit session orchestration + STT + retrieval + local embedding model + vector DB access + LLM + TTS`

That is acceptable for proving functionality. It is **not** the architecture to trust for large corpora, high worker counts, or strict latency targets.

### The stronger target architecture

```text
Caller
  -> LiveKit SFU
  -> Thin Voice Worker
       -> STT / turn handling
       -> Retrieval Service
            -> shared embedding path
            -> shared vector DB
            -> top-K + rerank only
       -> LLM provider
       -> TTS provider
```

### Why this is the right direction

1. **Thin workers scale better than heavy workers**
   - A worker should primarily own room/session lifecycle, interruption behavior, and provider orchestration.
   - It should not carry large retrieval state or heavy local models if the system is expected to scale horizontally.
   - This matches LiveKit's production guidance around worker pools, deployment management, load reporting, and scaling.

2. **Retrieval should be shared, not duplicated inside every worker**
   - For a tiny corpus, local retrieval inside the worker is fine.
   - For large corpora, loading embedding models or retrieval state in every worker wastes memory and makes capacity planning harder.
   - The worker should call a fast retrieval boundary and receive grounded excerpts back.

3. **Vector search must stay top-K only**
   - The production pattern is: query embedding -> ANN/vector search -> top-K results -> rerank candidates only.
   - Not full scans, not full collection loads, not per-worker copies of a large corpus.

4. **Capacity must be explicit**
   - Workers must report when they are full and stop accepting new sessions.
   - Session limits, load thresholds, and active-session accounting are architecture features, not optional tuning.

### What we measured that led to this conclusion

- Retrieval has been brought down to roughly `~186-240 ms` in the best local/local path.
- The remaining major latency is still in:
  - end-of-turn / endpointing
  - LLM TTFT
  - overall response path
- That means the next big wins are no longer "change DB only" or "change embeddings only."
- The next big wins are architectural:
  - thinner workers
  - better turn handling
  - faster model path
  - explicit scaling boundaries

### Sources that support this direction

- LiveKit agent deployment / scaling: [https://docs.livekit.io/deploy/agents/](https://docs.livekit.io/deploy/agents/)
- Deepgram Flux for conversational STT / turn handling: [https://developers.deepgram.com/docs/flux/quickstart](https://developers.deepgram.com/docs/flux/quickstart)
- Chroma query/get pattern: [https://docs.trychroma.com/docs/querying-collections/query-and-get](https://docs.trychroma.com/docs/querying-collections/query-and-get)
- Milvus reranking / top-K candidate merge pattern: [https://milvus.io/docs/reranking.md](https://milvus.io/docs/reranking.md)

---

## PRIORITY 1: Latency — Hitting <600ms Round-Trip

The total measured latency = `end_of_utterance_delay_ms` + `llm_ttft_ms` + `tts_ttfb_ms`.

**Actual measured numbers from our environment**

**Current best default cascade path**
- retrieval in the best local/local path: **~186-240ms**
- latest successful default smoke: **~9019ms**
- latest successful default worker turn snapshot:
  - `EOU ~552ms`
  - `LLM TTFT ~2678ms`
  - `TTS TTFB ~216ms`

**Gemini Live comparison path**
- grounded run: **~10917ms**
- no-tool run: **~4659ms**

These are the real baselines. The 600ms target requires significant changes, not just tuning.

### What `100` concurrent actually means

It does **not** automatically mean `100 workers`.

What matters is:

`concurrent call slots = worker replicas * safe sessions per worker`

Right now, based on the live stepped runs we actually performed, the safe density is still close to:
- `1 session / worker`

So the current path toward `100` is much closer to:
- many workers with low proven density

not:
- a few fat workers with high unproven per-worker density

**Why this matters**
- voice concurrency is session-based and long-lived, not web-request style throughput
- the current hot path is still long and provider-sensitive
- forcing higher per-worker density before the runtime is stable would produce fake scale claims

### Current-System Priority Order (Most impact for this repo)

This is the order that makes sense **for the current system**, not for a hypothetical future platform.

#### P1. Make grounded direct answers concise enough for voice without losing exact facts
**Where**
- `kb.py`
- `worker.py`
- `livekit_smoke.py`
- `tests/e2e/test_voice_flow.py`

**What to do**
- keep the new exact-fact path, but make the spoken answer short enough to sound like a real phone response
- preserve the exact number / time window while trimming OCR noise and secondary clauses that do not answer the caller's question directly
- keep the fix generic:
  - no question-specific prompts
  - no special casing a single policy query

**How**
- tighten snippet extraction / clause trimming in `kb.py` so direct answers keep the exact supported clause but drop irrelevant tail text
- validate against at least:
  - refund single-turn live smoke
  - reinstatement single-turn live smoke
- keep the answer factual and exact, but reduce spoken length and latency

**Why this is highest priority now**
- the assignment requires objection handling, and the factual part is now working
- the next missing piece is graceful spoken delivery, not raw correctness
- this also helps latency because the current recovered direct answers are too long for a real voice UX

**Reasoning / source**
- this is based on our own current evidence:
  - the live objection path now uses a recovered direct grounded answer
  - the spoken response is factually correct, but still too long and OCR-heavy
  - that is now unnecessary latency and poor caller UX sitting directly in the live path
- voice-agent best practice is to optimize time to first useful audio, not just raw correctness

**Immediate acceptance criteria**
- live refund smoke still passes
- live reinstatement smoke still passes
- reinstatement answer becomes materially shorter while keeping:
  - `reinstate`
  - `two`
  - `years`
  - and without `three`, `five`, `3`, or `5`
- if a trim attempt drops the exact fact, revert it

**Latest hard proof**
- April 4, 2026 live reinstatement smoke now passes after:
  - recovering a grounded direct answer inside `llm_node(...)`
  - and stopping turn-boundary over-splitting from cutting the question in half
- current live reinstatement response is still too verbose and OCR-heavy for production voice UX
- current live refund smoke still passes
- so the next improvement is trimming the direct answer, not chasing correctness drift again

#### P2. End-of-turn / endpointing proof, then optimization
**Where**
- `worker.py`
- `livekit_smoke.py`
- `tests/e2e/test_voice_flow.py`

**What to do**
- first prove the current turn boundary with caller-like audio across multiple turns in one room
- then reduce turn-finalization latency without breaking interruption behavior
- validate whether the current VAD + endpointing settings are too conservative

**How**
- use the existing LiveKit smoke harness to synthesize caller audio, publish it turn by turn, and transcribe the agent's spoken replies
- keep it in one room/session so it exercises real multi-turn behavior, not three isolated calls
- gate the multi-turn live test behind explicit env so it remains truthful without breaking baseline validation when provider quota is exhausted:
  - `RUN_MULTI_TURN_E2E=1`
- after that proof is stable, compare:
  - current VAD path
  - tighter endpointing settings
  - only one winning configuration stays

**Why this remains near the top**
- current measured EOU delay is still a major latency component
- this is a structural latency problem, not a “prompt tweak” problem
- tuning endpointing blindly is risky; we need the live objection path stable first, then a caller-like multi-turn proof before changing the turn boundary

**Reasoning / source**
- Deepgram and other voice-agent guidance consistently treat end-of-turn latency as one of the main UX bottlenecks
- our own measurements already show EOU is one of the biggest remaining chunks
- real voice systems are judged on turn-taking feel, not just STT word accuracy

**What is already done here**
- the explicit startup greeting has been removed from the concurrent startup path in `worker.py`
- a live 4-worker run after that change no longer showed the earlier Cartesia greeting-path failure
- the next pressure point after greeting removal shifted back to the real turn and provider path instead of ceremonial startup TTS
- the repo already has a real multi-turn live harness:
  - `run_multi_turn_smoke(...)` in `livekit_smoke.py`
  - `test_live_voice_agent_multi_turn_audio_flow(...)` in `tests/e2e/test_voice_flow.py`
- turn-scoped grounding is now replaced per turn instead of accumulating stale developer guidance in the same session
- grounded excerpts are now narrowed to the highest-scoring answer sentence instead of passing an entire noisy policy chunk into the live turn
- what is still missing is not code existence, but repeatable live proof in the current environment

**Immediate acceptance criteria**
- one live room/session
- at least `3` caller-like audio turns in sequence
- each turn must produce a correct grounded spoken answer
- no false `stt_timeout` reprompt during a successful turn
- the proof should be repeatable enough to trust before using it as the basis for bigger concurrency runs
- if a rerun fails, the first failing hop must be identified as one of:
  - transcript drift
  - retrieval miss
  - grounded-answer drift
  - provider quota / provider failure

#### P2A. Live-turn LLM path: fail fast on overload, then revalidate stepped load
**Where**
- `worker.py`
- OpenAI-compatible provider path used by the default cascade runtime
- stepped live load proof

**What to do**
- keep live-turn LLM calls on a short bounded timeout
- avoid hidden retry multiplication on the live response path
- rerun stepped load and see whether the next visible bottleneck is still provider overload or something more local

**How**
- wire `HTTP_TIMEOUT_SECONDS` and `AGENT_MAX_RETRIES` directly into the OpenAI-compatible client
- keep the default at `AGENT_MAX_RETRIES=0`
- keep `FallbackAdapter(max_retry_per_llm=0)` so failover is explicit but retry storms do not grow inside the live path
- validate with focused tests first, then one stepped live load point

**Why this mattered**
- retries at multiple layers are exactly how small provider stalls turn into user-visible dead air and slot starvation
- under concurrent sessions, an overloaded LLM provider should fail fast instead of tying up scarce session capacity
- this directly supports the assignment’s resilience and concurrency requirements without adding infrastructure

**Reasoning / source**
- AWS Builders Library guidance on timeouts and one-layer retries maps directly here
- the repo’s own live runs already showed provider overload can be a concurrency ceiling once worker topology is no longer the first bottleneck

**What is already done here**
- OpenAI-compatible LLM creation in `worker.py` now passes:
  - `timeout=httpx.Timeout(settings.http_timeout_seconds)`
  - `max_retries=settings.agent_max_retries`
- default `AGENT_MAX_RETRIES` is now `0`
- focused tests prove the client is created with zero retries and the configured timeout
- a live 4-worker stepped load point after this change produced:
  - `concurrency=4`
  - `successes=3/4`
  - `avg_ms=9389.61`
  - `p95_ms=9589.49`
- this is still far from the assignment target, but it is materially better than the earlier `0/4` collapse seen after greeting-path cleanup alone

#### P3. Honest stepped concurrency proof after the objection path is stable enough
**Where**
- `load_test.py`
- `scripts/start.sh`
- local multi-worker topology with Redis-backed active-session state

**What to do**
- rerun stepped live load with the now-correct objection path and current worker topology
- stop at the first real failure level and classify it honestly

**How**
- start from the last truthful proven point and move carefully:
  - `4`
  - `6`
  - `10`
- record:
  - success rate
  - first failure mode
  - p95 latency
  - whether failures are provider-side, endpointing-side, or worker-slot-side

**Why this is next**
- the assignment still requires `100` concurrent calls
- we already removed a real correctness blocker from the live objection path
- the next useful truth is not more architecture theory; it is the next real concurrency failure point

#### P4. STT timeout path: live revalidation after the local fix
**Where**
- `worker.py`
- live concurrent audio path
- stepped load proof

**What to do**
- confirm that the more conservative, transcript-length-aware STT timeout actually reduces false reprompts under concurrent speech
- verify that successful final transcripts reset timeout escalation cleanly across turns

**How**
- use the adaptive timeout already implemented in `worker.py`
- rerun live concurrent audio after provider quota allows it
- keep only one question in mind:
  - did false `stt_timeout` become materially rarer under the same kind of concurrent utterances?

**Why this is here**
- the local code path was already tightened
- it still needs live proof because concurrent partial speech was one of the first misleading failure modes we saw
- the next goal is to reduce avoidable runtime instability before rerunning larger stepped load

**What was implemented**
- flat `3.0s` STT timeout was replaced with a more conservative timeout that grows with partial-utterance length
- successful final transcripts now reset STT-timeout escalation so one recovered turn does not poison later turns

#### P3. Honest concurrency proof
**Where**
- `load_test.py`
- `worker.py`
- runtime configuration
- deployment/startup layer (`scripts/start.sh` locally, real orchestrator in production)

**What to do**
1. fix the local topology so the test is not artificially capped at one worker
2. then run stepped concurrency:
   - `2 -> 5 -> 10 -> 25 -> 50 -> 100`
3. record:
   - successes
   - failures
   - p95 latency
   - setup failures
   - worker capacity behavior
   - which failure mode shows up first

**How**
- keep the current harness and its query-specific grading
- run more than one worker replica locally/in staging; do not try to “solve” concurrency by blindly increasing `WORKER_MAX_CONCURRENT_SESSIONS`
- keep `WORKER_MAX_CONCURRENT_SESSIONS=1` as the safe default until a higher per-worker session count is explicitly proven
- treat overload/no-answer as real failures
- write down where it breaks first
- for production-grade shared state:
  - Redis-backed active-session counting is now implemented as the first production-grade substitution
  - next improvement is adding expiry/lease semantics once we move beyond the current single-counter model
  - replace JSONL cross-process latency sharing with Prometheus multiprocess mode or an external telemetry backend

**Why this matters now**
- the assignment explicitly asks for `100` concurrent calls
- concurrency proof is more important than speculative architecture churn
- this is the only honest way to know whether the current worker shape holds at all
- the current repo already proved that a single local worker is a topology bottleneck before any deeper scaling discussion even starts
- the latest multi-worker runs also proved that adding workers helps, but runtime/provider behavior becomes the next bottleneck

**Reasoning / source**
- LiveKit production guidance is built around explicit worker load, admission, and scaling
- a system is not “scalable” because we think it is; it is scalable when measured under stepped load
- LiveKit docs explicitly describe balanced job distribution across agent servers plus autoscaling off the same `load_fnc`/`load_threshold`
- Redis documents `INCR` as the standard atomic counter pattern and explicitly calls out `INCR` + `EXPIRE` for short-lived counters
- Prometheus Python docs explicitly require multiprocess mode when metrics are shared across Python worker processes

**What is already done here**
- `load_test.py` now supports stepped runs via `run_stepped_load(...)`
- the load summary now reports the actual `concurrency` level alongside `sessions`, `successes`, `failures`, `avg_ms`, and `p95_ms`
- failure examples remain surfaced instead of being swallowed
- worker startup now clears the stale local active-session file so local concurrency validation starts from a clean state
- the same active-session path now supports Redis-backed shared counting when `REDIS_URL` is configured
- the load harness now uses the correct expected-keyword set for each evaluation query
- integration coverage now proves:
  - explicit `concurrency` / `turns` overrides are respected
  - one summary is returned per requested load level
  - `run_single_session(...)` now forwards query-specific expectations into the live smoke path
- local multi-worker startup now works again after fixing the `ActiveSessionsStore.write(...)` temp-file race
- local two-worker testing proved:
  - both workers register cleanly
  - LiveKit dispatch can hand two concurrent rooms to different workers
  - the app no longer depends on a local file counter when Redis is available

**What is still missing**
- stable multi-turn live proof in one room/session before treating bigger concurrency numbers as trustworthy
- higher-concurrency reruns after the current first-failure modes are reduced:
  - false-positive `stt_timeout` under heavier concurrent speech
  - provider overload / rate-limit behavior after the live-turn retry path was made fail-fast
  - remaining Cartesia behavior under concurrent generation
- then real stepped live runs such as:
  - `4 -> 6 -> 10 -> 25 -> 50 -> 100`
- the resulting evidence in the final assignment/demo artifacts

**First truthful stepped proof so far**
- live stepped run at `1 -> 2` with one turn each produced:
  - `concurrency=1`: `1/1` success, `avg_ms=19137.77`
  - `concurrency=2`: `0/2` success, `2` failures, `failure_examples=TimeoutError`
- what this means:
  - the current local single-worker shape is not concurrency-safe beyond one active session
  - the next job is not to pretend it scales, but to move to a real worker pool and rerun stepped load with honest grading

**Next truthful stepped proofs**
- live stepped run with **two workers** and one turn each produced:
  - `concurrency=2`: `2/2` success, `avg_ms=9918.31`
  - `concurrency=4`: `2/4` success, `2` failures, `failure_examples=TimeoutError`
- live stepped run with **four workers** and one turn each produced:
  - `concurrency=4`: `3/4` success, `1` failure, `avg_ms=11389.54`
  - `concurrency=6`: `2/6` success, `4` failures, `failure_examples=TimeoutError | RuntimeError: No agent audio was received after publishing the client question`
- live stepped rerun with **four workers** after greeting removal and live-turn LLM fail-fast wiring produced:
  - `concurrency=4`: `3/4` success, `1` failure, `avg_ms=9389.61`, `p95_ms=9589.49`
- what this means:
  - worker-pool topology was the first real blocker, and increasing worker slots did improve concurrency
  - greeting-path fragility is no longer the first obvious startup blocker
  - after topology improved, the next bottlenecks became provider/runtime behavior under concurrent generation, not retrieval
  - the current system is still nowhere near the assignment target, but the failure point is now clearer and more honest
  - the next precise job is to make the multi-turn live conversation loop trustworthy before pushing harder on higher concurrency levels

**Production-grade substitutions for the current local implementation**
- **Worker pool instead of one oversized worker**
  - where:
    - deployment/orchestration layer
    - `scripts/start.sh` for local proof
  - how:
    - run multiple worker replicas
    - keep one active session per worker until higher density is explicitly proven
    - autoscale off the same signal used by `load_fnc`
  - why:
    - LiveKit already distributes jobs across agent servers and documents autoscaling against `load_fnc` / `load_threshold`
    - our own testing already showed topology is the first concurrency bottleneck
- **Redis for ephemeral shared session state**
  - where:
    - `observability.py`
    - `worker.py`
    - `api.py`
    - `docker-compose.yml`
  - how:
    - implemented:
      - optional `REDIS_URL`
      - shared counter operations backed by Redis `GET` / `SET` / `INCR`
      - non-negative decrement path for active-session cleanup
      - file fallback kept for minimal local runs
    - next:
      - add expiry/lease semantics if the counter model becomes insufficient under crash recovery testing
  - why:
    - the current file-backed counter is a local-dev convenience, not a production shared-state mechanism
    - Redis is a standard fit for short-lived counters, leases, and coordination across multiple worker processes/hosts
- **Prometheus multiprocess or external telemetry instead of JSONL fan-in**
  - where:
    - `observability.py`
    - `api.py`
  - how:
    - use Prometheus Python multiprocess mode when exporting across multiple worker processes
    - or move p95/avg aggregation into a real metrics backend
  - why:
    - the current JSONL merge is simple and fine for local dev, but it is not the right production aggregation shape for multi-worker exports
- **Docker Compose as the first local production-like infra layer**
  - where:
    - repo root infra artifacts
    - `docker-compose.yml`
    - `scripts/start.sh`
  - how:
    - keep the app container wiring simple
    - add Redis and Prometheus only when those substitutions are actually implemented
    - prove local multi-worker behavior with the same Compose topology before considering anything heavier
  - why:
    - we need local, reproducible concurrency proof before any serious deployment claim
    - Compose is the smallest useful orchestration step; anything heavier right now would be bloat

**Current first failure modes under stepped live load**
- **False-positive STT timeout under concurrent speech**
  - where:
    - `worker.py`
  - what happened:
    - `stt_timeout` fired on partial utterances that later completed successfully
  - why it matters:
    - this can trigger unnecessary reprompts and degrade caller UX under load
  - what was changed:
    - timeout is now adaptive to partial-utterance length instead of a flat `3.0s`
    - STT timeout escalation is reset after a real final transcript so one recovered turn does not poison the next turn
  - what is still needed:
    - revalidate under live concurrent audio after provider quota allows it
- **Concurrent Cartesia greeting failures**
  - where:
    - live worker runtime / TTS provider path
  - what happened:
    - concurrent startup produced `no audio frames were pushed for text: Hello, this is the policy support assistant...`
  - why it matters:
    - the synthesized greeting itself is currently part of the concurrency cost and one of the first unstable points
  - next:
    - reduce greeting-path fragility before treating higher stepped-load results as trustworthy
- **Gemini free-tier quota exhaustion**
  - where:
    - live LLM provider path
  - what happened:
    - real `429 RESOURCE_EXHAUSTED` during stepped live load
  - why it matters:
    - this is now part of the observed concurrency ceiling in the current environment
    - it does not mean the architecture is solved; it means the environment is also a limiter
  - next:
    - keep this documented as an environment constraint while fixing the local runtime issues we actually control first

#### Completed foundation: Caller-visible failure UX
**Where**
- `worker.py`
- session/runtime error handling

**What was implemented**
- STT timeout reprompts once, then escalates to a clean spoken fallback
- retrieval failure already degrades safely instead of inventing policy facts
- LLM/session failures now trigger a caller-visible fallback message rather than silent backend-only logging
- fake `tts_no_audio` timeout logic was removed because it produced false alarms on successful calls

**Why this stays out of the top open priorities**
- the current path already has the minimal failure-policy behavior it needs
- the bigger remaining gaps are now live stability and honest concurrency proof

#### P4. Objection scenario proof
**Where**
- tests / smoke scenarios / demo plan
- maybe prompt/grounding behavior if needed

**What to do**
- add one explicit objection scenario and prove it:
  - example: cancellation after the free-look period
  - example: grace-period objection

**How**
- define one concrete user journey
- verify the agent:
  - answers the supported part directly
  - states what is unclear or unsupported
  - does not hallucinate

**Why**
- this is explicitly required by the assignment
- it is smaller and more actionable than broad “make agent smarter” work

**Reasoning / source**
- direct assignment requirement
- aligns with the current grounded-answer architecture

#### P5. Multi-turn follow-up query rewriting
**Where**
- `worker.py`
- helper near `SupportAgent._build_grounding_query(...)`

**What to do**
- rewrite short follow-up questions into retrieval-friendly queries using recent turns

**How**
- only use recent local session context
- keep it deterministic and small
- do not add global memory or persistent user profiles

**Why**
- current agent is multi-turn only in a basic session-history sense
- ambiguous follow-ups are the biggest realism gap in the current agent behavior

**Reasoning / source**
- Daily / Pipecat engineering guidance repeatedly emphasizes context engineering for long and multi-turn voice conversations
- this gives real conversational lift without heavy memory infrastructure

#### P6. Deeper architecture only after the above
**Where**
- retrieval boundary / separate service
- alternate provider routing / multi-region tricks

**What to do**
- only revisit if the current measured bottlenecks still justify it after P1-P5

**Why**
- these ideas are real, but they are not the next 80/20 move for this repo
- adding them too early increases complexity faster than it improves the current assignment outcome

### 1C. Eliminate Double LLM Round-Trip on KB Calls (DONE for current cascade path)
**File:** `worker.py:80-149` — the `SupportAgent` class

**What was done**
- Removed the old tool-shaped grounded answer path from the critical turn flow.
- Grounding now happens in `on_user_turn_completed(...)` before the LLM runs.
- Added a lightweight gate so greetings / connection checks do not always hit retrieval.
- Added split-turn reconstruction so final STT fragments such as `premium back?` do not overwrite the real question.

**Why it was necessary**
- The previous architecture paid unnecessary conversational overhead for grounded turns.
- The split-turn bug caused factual regressions even when retrieval quality was fine.
- Fixing the query assembly was required for correctness, not just latency.

**What is still missing**
- Surface `on_user_turn_completed` delay explicitly in metrics.
- Prevent duplicate retrieval work on partial/final transcript churn more aggressively if LiveKit still emits redundant finalization patterns.
- Improve multi-turn follow-up grounding for elliptical follow-up questions that omit the original subject.

### 1B. LLM Model + Provider Choice (HIGH IMPACT — cascade mode)
**File:** `worker.py:152-159`, `settings.py`

**Problem:** LLM TTFT is a major latency contributor. Model/provider choice matters enormously.

**Options:**

| Provider/Model | TTFT (streaming) | Pros | Cons |
|---|---|---|---|
| **Groq (llama-3.1-8b)** | 50-150ms | Blazing fast, free tier | Smaller model, may miss nuance |
| **Groq (llama-3.3-70b)** | 100-250ms | Fast + smart | Rate limits on free tier |
| **OpenAI GPT-4o-mini** | 150-350ms | Good balance | Costs more than Groq |
| **Gemini 2.0 Flash** | 200-400ms | Multimodal | Via bridge = extra overhead |
| **Cerebras (llama-3.3-70b)** | 50-100ms | Fastest inference | Limited availability |

**What we already validated**
- `gemini-2.5-flash-lite`: faster but factually wrong on the policy question. Rejected.
- `gemini-2.5-flash`: faster than the old default and factually correct on the same smoke question. Kept as the default.

**Current keep-state**
```
OPENAI_MODEL=gemini-2.5-flash
```

**Why this was the right change**
- It improved the default successful smoke from `~9509ms` to `~9019ms`.
- It preserved factual correctness on the tested refund-policy question.

**What still needs to be done**
- Benchmark at least one non-Gemini fast text provider only if we can validate correctness at the same time.
- Do not switch providers based on TTFT marketing alone.

**Free-tier reality:**
- Groq free: 1,000 req/day org-wide, ~30K TPM. Fine for a demo of 1-5 calls. Cannot sustain 100 concurrent.
- Cerebras free: 1M tokens/day, no LiveKit plugin, but speaks OpenAI-compatible API (`OPENAI_BASE_URL=https://api.cerebras.ai/v1`). Worth trying — reported 50-100ms TTFT.
- Deepgram: $200 free credit covers both STT and TTS. Most generous free allowance we have.

**Note:** TTFT numbers above are vendor-reported/community-benchmarked, not measured in our environment. Must validate after switching.

### 1A. Tune STT Endpointing (PARTIALLY DONE for cascade; more optimization still possible)
**File:** `worker.py:237`

**Applies only if cascade is the primary assignment path.** Not relevant for Gemini Live mode.

**Problem:** `deepgram.STT(model="nova-3", language="multi")` — multi-language detection adds overhead for language identification on every utterance.

**What we already changed**
```python
stt=deepgram.STT(
    model="nova-3",
    language="en-US",
    smart_format=False,
    interim_results=True,
)
```

**Why this mattered**
- Multi-language auto-detection was not justified for the current English-only policy demo.
- We also needed partial transcripts enabled so we could see and fix the split-turn behavior.

**What still needs to be done**
- If we keep cascade as primary, test a more conversational STT / turn-taking path such as Flux-style turn handling.

### 1D. Fix Retrieval Latency (DONE for the current prototype path; scale work remains)
**File:** `kb.py:205-213`, `retrieval.py`

**What was done**
- Replaced the fake local placeholder embedding path with the real local Jina model.
- Added `local|remote` switches for both embeddings and Chroma.
- Rebuilt the same corpus in both local and remote Chroma and compared them with the same data.
- Kept only the production-correct retrieval shape:
  - query embedding
  - top-K vector query
  - rerank candidates only

**What we measured**
- remote/remote style path was materially slower
- local/local path brought retrieval down to roughly `~186-240ms` in the best runs
- local Chroma vector query was effectively negligible compared with the text model path

**Why this mattered**
- It proved retrieval was fixable without fake caching shortcuts.
- It also proved retrieval is no longer the main latency villain in the current prototype path.

**What still needs to change for real scale**
- Stop loading the embedder inside every worker process.
- Move retrieval ownership behind a shared boundary if we care about large corpora / many workers.

**What was additionally fixed during validation hardening**
- grounding breadth was widened from top-1 to top-3 for live turns
- the grounding instruction now tells the model to answer directly when excerpts support the question, and to refuse only when they do not support the answer at all

**Why this mattered**
- the live E2E refund flow failed on a realistic phrasing (`cancel and return the policy`) even though the KB did have relevant excerpts
- narrow grounding was too brittle for real conversation phrasing

### 1E. Trim the System Prompt (DONE)
**File:** `prompts.py`

**What was done**
- Removed tool-calling instructions after the grounded-turn injection path replaced the old tool flow.
- Kept the prompt focused on:
  - short phone-friendly answers
  - grounded facts only
  - explicit refusal when grounding is absent or weak

**Why this mattered**
- The prompt now matches the actual architecture.
- Keeping stale tool instructions would have been misleading and unnecessary.

### 1F. TTS Optimization (LOWER PRIORITY than retrieval/LLM, but still measurable)
**File:** `worker.py:239`

**Measured TTFB (independent benchmarks):** Deepgram Aura-2 is **90-200ms isolated** (median ~90-150ms, p95 ~200ms — sources: Podcastle benchmark, Jambonz TTS leaderboard, CloudX 30+ stack tests). In full pipelines it can reach 300-450ms due to orchestration overhead. That's 15-33% of the 600ms budget — not dominant vs retrieval/LLM, but not nothing.

**Faster alternatives exist but aren't practical on free tier:**
- Cartesia Sonic Turbo (~40ms TTFB) — `livekit-plugins-cartesia` exists for 1.4.x, but free tier is only 10K characters (~minutes of speech). Unusable for a 100-call demo.
- ElevenLabs Turbo v2.5 (~135ms) — free tier is 10 min/month. Also unusable.

**What we actually validated**
- Cartesia is measurably faster than Deepgram on the TTS stage in this repo.
- The TTS-stage gain is real, but it does not solve the end-to-end latency target by itself.

**Current keep-state**
- Keep `TTS_PROVIDER=cartesia` for the current default path because it improved the stage that we could actually measure here.

### 1G. Audio-to-Audio Model (Gemini Live) — Comparison only, not primary
**File:** `worker.py:224-234` (already implemented as `gemini_live` mode)

**What it does:** Collapses STT + LLM + TTS into one model. Audio in → audio out.

**Measured reality in our environment:**
- No-tool run: **~4659ms** (not the "200-400ms" claimed in vendor marketing)
- Grounded (with KB tool): **~10917ms**

**Why NOT primary:**
1. Assignment explicitly asks to instrument STT → LLM → TTS with per-component latency (EOU, TTFT, TTFB). Audio-to-audio is a black box — can't show breakdowns.
2. KB integration in Gemini Live realtime mode is limited.
3. Our actual measured latency does not support the claim that it's faster.

**Use it as:** A documented comparison in the write-up. Show both modes, present honest numbers, explain the tradeoffs.

---

## PRIORITY 2: Infrastructure & Scaling

### 2A. Split retrieval out of the voice worker (HIGHEST architecture priority for scale)
**Why this must happen**
- The current worker still owns too much:
  - local embedding model load
  - vector DB query path
  - reranking
- That shape does not scale well to large worker counts because memory and retrieval complexity are duplicated across workers.
- A thin worker architecture is easier to autoscale, easier to debug, and more honest about capacity.

**Target shape**
```text
Voice worker -> Retrieval service -> Shared vector DB
```

**What the retrieval service should do**
- accept raw user query text
- compute query embedding
- run top-K vector search
- rerank candidates only
- return compact grounded excerpts plus source metadata

**What the worker should do**
- send the query
- inject returned context into the turn
- call the LLM once
- speak the answer

**What to watch out for**
- do not add queue/broker latency here
- keep the retrieval service in the same region as the worker and vector DB
- keep response payloads compact

**Why this is better**
- avoids loading heavy retrieval logic in every worker
- creates one clear place to optimize retrieval latency
- supports bigger corpora without rewriting the whole voice runtime

**Current state**
- We already extracted a code-level retrieval seam via `RetrievalService`.
- The worker no longer reaches directly into KB internals.
- This is enough for now to avoid overengineering.
- A separate retrieval service process is still **not** implemented, because it would add architecture surface before the current assignment blockers are proven.

### 2B. Docker / containerization (DONE)
**What was added**
- `Dockerfile`
- `docker-compose.yml`
- `scripts/start.sh`
- `README.md` setup instructions

**Why this mattered**
- The assignment explicitly requires setup instructions and scripts for spinning up infra.
- These files are the minimum useful deliverable set for reproducible local/demo startup.

### 2C. Scaling Architecture for 100 Concurrent Calls
**Current state: NOT proven at scale.** The design is intended for horizontal scaling, but we have not validated it. Missing pieces:
- No validated 100-session run
- Free-tier provider rate limits (Groq: 1K RPD, Deepgram: $200 credit) cannot sustain 100 concurrent calls

**Intended scaling design (document honestly as intended, not current):**
- LiveKit SFU handles media routing and session routing
- Voice workers stay thin and capacity-bound
- Retrieval is shared instead of embedded in every worker
- Vector search remains top-K only with rerank-on-candidates
- Provider limits are treated as first-class infra constraints

**What to add to make it real:**
- Multi-worker validation with the current bounded worker admission and shared active-session metrics
- Show how to run multiple worker instances (docker-compose scale / orchestrator)
- Document that 100-call proof requires paid provider tiers and same-region deployment

**Reasoning**
- Capacity must be explicit or the worker will lie about scale.
- A system that is "fast for one call" but has no admission/backpressure is not production-ready.

### 2D. Worker process design
**What needs to happen**
- Keep parent worker startup light
- Load heavy job-specific state only where needed
- Avoid eager heavyweight initialization in the parent process

**What we already fixed**
- The parent worker no longer eagerly loads the local KB/embedder before any job exists.
- Load-based availability is now tied to worker session capacity.
- The API/control plane no longer eagerly warms the local embedding model just to construct `KnowledgeBase`.
- The deprecated `aiohttp` subclass patch was removed in favor of a simpler constructor patch.

**What is still missing**
- explicit per-session budgets and failure accounting
- multi-worker validation, not just one worker

**Why this mattered**
- eager local model warmup in the API was unnecessary startup work
- it also made validation noisier and less representative of the actual control-plane role

### 2E. Failure Recovery (PARTIALLY DONE; still needs live proof)
**What is already implemented**
- explicit retrieval degradation:
  - if grounding fails, the agent is instructed not to invent policy facts
- optional LLM provider failover:
  - gated by `FALLBACK_OPENAI_BASE_URL`, `FALLBACK_OPENAI_API_KEY`, `FALLBACK_OPENAI_MODEL`
  - no silent swap; availability changes are logged and metered
- explicit session and upstream failure accounting:
  - `session_start`
  - session runtime errors
  - `stt_timeout`
  - explicit provider-reported TTS/session failures
- load-test failure reporting now preserves example causes

**Where**
- `worker.py`
- `settings.py`
- `observability.py`
- `load_test.py`

**Why this matters**
- The assignment requires more than blind retries.
- These changes make failures visible, classifiable, and safer to operate.
- They also avoid the dangerous behavior of silently masking grounded-answer failures.

**What is still missing**
- a live forced-failure validation of the fallback path
- a short operator-facing recovery note/runbook for the final deliverable

### 2F. Validation Reliability (DONE for current repo workflow)
**What was done**
- `scripts/start.sh` now has a real stable validation mode and a real live validation mode
- live validation uses temp runtime state instead of the shared `data/active_sessions.json` and `data/turn_metrics.jsonl`

**Why this mattered**
- repeated live validation runs were contaminating each other via shared state
- a validation script that reuses stale runtime files is not trustworthy

---

## PRIORITY 3: Documentation & Deliverables

### 3A. Architecture Diagrams (DONE)
**Where**
- `docs/ARCHITECTURE.md`

**What is included**
1. high-level system architecture
2. call-flow diagram
3. scaling plan

**Why this mattered**
- These are explicit assignment deliverables.
- Keeping them in one file avoids unnecessary file sprawl while still making the design reviewable.

### 3B. LiveKit vs Pipecat Comparison (DONE)
**Where**
- `docs/ARCHITECTURE.md`

**Why this mattered**
- The assignment explicitly asks for the comparison.
- The write-up now exists in-repo and is aligned with the current architecture choice.

### 3C. 1-Pager: What Breaks at 1000 Calls (DONE)
**Where**
- `docs/ARCHITECTURE.md`

**Why this mattered**
- The assignment explicitly asks for this written analysis.
- It is now documented with the current measured bottlenecks and the intended scaling direction.

---

## PRIORITY 4: Testing & Demo

### 4A. Prove <600ms with Real Measurements
- Implement 1C + 1D + 1B on top of the stronger architecture (thin worker + shared retrieval boundary)
- Run load test, capture turn_metrics.jsonl
- Show per-component breakdown
- **Be honest about results.** If we can't hit 600ms, document where the time goes and what would fix it.

### 4B. Demo Video Requirements
- Multiple concurrent calls (use load_test.py, show 5-10+ simultaneous)
- One latency measurement run (show actual numbers, good or bad)
- Barge-in demo (interrupt agent mid-sentence)

### 4C. Actual Network Quality Metrics
Currently MOS is estimated from jitter/packet_loss formula. For the demo, capture real RTP stats from LiveKit room stats API.

### 4D. Objection Scenario Proof
- add one explicit objection scenario to the demo/test set
  - example: cancellation after the free-look period
  - example: premium grace-period objection
- show that the agent answers the supported part directly and clearly states the unsupported part when the document does not fully answer it

**Why**
- the assignment explicitly asks for at least one objection scenario handled gracefully
- the current repo answers policy questions, but does not yet prove objection handling as a deliberate behavior

---

## Enhancements (Not current assignment blockers, but important)

### E1. Multi-Turn Follow-Up Retrieval
**Current state**
- the session is multi-turn in a basic sense because LiveKit keeps the conversation history for the room session
- grounded excerpts injected in `on_user_turn_completed(...)` are turn-local, not durable memory
- retrieval currently grounds based on the latest assembled user turn only

**What this means in practice**
- multi-turn conversation works when the user repeats enough context
- short follow-ups such as `what about after 12 days?` can become ambiguous because retrieval may not see the missing prior context

**What to add**
- lightweight follow-up query rewriting before retrieval
- use recent user/assistant turns to expand ambiguous follow-up questions into a retrieval-friendly query

**Where**
- `worker.py`
- potentially a small helper near `SupportAgent._build_grounding_query(...)`

**Why**
- this is the biggest gap between a working prototype and a more natural real agent conversation
- it improves real multi-turn robustness without requiring a fancy memory system

### E2. Session Summary / Local Memory
**Current state**
- no explicit session summary memory exists
- no cross-session or persistent user memory exists

**What to add later**
- compact session summary memory for long conversations
- keep it session-scoped, not customer-global

**Why**
- long conversations eventually push too much raw history into the model context
- session summary is useful before any global memory system is justified

### E3. Global / Persistent Memory
**Current state**
- not implemented
- not required for the assignment

**Why it is not current priority**
- the assignment does not require customer profiles or long-term personalization
- global memory adds privacy, correctness, and lifecycle complexity
- it should not be added before latency, concurrency, and demo proof are finished

---

## Execution Order

1. **Update the final documentation to match the real implemented resilience state**
   - failure handling is no longer "missing"
   - this matters because the code and TODO must agree before we claim anything externally
2. **Concurrency proof**
   - validate 5 -> 10 -> 25 -> 50 -> 100
   - record failures, latency, and capacity behavior honestly
3. **Live forced-failure proof**
   - validate one intentional primary-LLM failure with fallback configured
   - validate one retrieval failure path
   - this matters because resilience should be demonstrated, not only implemented
4. **Objection scenario proof**
   - add one explicit objection conversation and validate the behavior
   - this matters because the assignment explicitly asks for it and the repo does not yet prove it
5. **Demo capture**
   - concurrent calls
   - one measured latency run
   - barge-in demo
6. **Only then revisit deeper architecture changes if still needed**
   - separate retrieval service
   - alternate STT path
   - alternate provider stack

---

## Key SDK Findings (reference for implementation)

**LiveKit Agent pipeline nodes** (overridable in `Agent` subclass):
- `stt_node(audio, model_settings)` — transcribe audio frames
- `on_user_turn_completed(turn_ctx, new_message)` — **hook between STT and LLM** — `turn_ctx` is temporary/mutable (`agent_activity.py:1568`), safe to inject without polluting history
- `llm_node(chat_ctx, tools, model_settings)` — LLM inference
- `tts_node(text, model_settings)` — synthesize audio
- `transcription_node(text, model_settings)` — post-process transcription

**ChatContext API:**
- `turn_ctx.add_message(role="developer"|"system"|"user"|"assistant", content="...")` (`chat_context.py:368`)
- Available roles: `developer`, `system`, `user`, `assistant`

**Agent constructor params relevant to latency:**
- `min_endpointing_delay` — minimum silence before finalizing turn
- `max_endpointing_delay` — maximum wait before forcing turn end
- `min_consecutive_speech_delay` — debounce for barge-in
- `allow_interruptions` — barge-in support

**Metrics:** SDK tracks `on_user_turn_completed` delay at `agent_activity.py:1614` — surface this to measure KB injection overhead.

**LLM fallback:** `livekit.agents.llm.fallback_adapter` (`fallback_adapter.py:34`) — for provider failover, not latency.

---

## Diagram Note

- Keep one detailed nested Mermaid graph in [`docs/ARCHITECTURE.md`](/Users/deshwalmahesh/Desktop/Personal/voice_agent/docs/ARCHITECTURE.md) that shows the exact end-to-end query/data flow:
  - caller / PSTN / Twilio / LiveKit
  - room audio input
  - VAD / endpointing
  - STT
  - user turn assembly
  - grounding gate
  - retrieval / embeddings / Chroma
  - snippet selection
  - turn-scoped guidance / direct grounded answer
  - `llm_node(...)`
  - TTS
  - room audio output back to caller
  - metrics / active-session store / control plane
- This should stay current with the actual runtime path, because it is submission-grade documentation, not decoration.
