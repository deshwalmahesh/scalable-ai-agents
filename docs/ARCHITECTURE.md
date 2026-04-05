# Architecture And Trade-Offs

## 1. Unified End-To-End Flow

```mermaid
flowchart TB
    Caller["Caller speaks into a phone call or a synthetic smoke-test call"]

    subgraph Telephony["Call entry and audio transport"]
        Twilio["Twilio receives the incoming phone call and forwards it over SIP"]
        LiveKit["LiveKit creates the call room, moves audio, and assigns the call to a worker"]
    end

    subgraph Worker["The voice agent worker handling this live call"]
        RoomIn["Receive the caller's live audio from the room"]
        VAD["Decide when the caller has stopped talking enough for the agent to reply"]
        STT["Convert the spoken audio into text"]
        Turn["Assemble the current user turn from the transcript"]
        TurnHook["Decide whether this turn needs policy grounding"]
        Query["Build the search question using the current turn and recent user context"]
        Guard["Is this a policy / product question or just small talk?"]
        SmallTalk["For small talk, tell the agent to answer briefly without policy lookup"]
        LLMNode["Prepare the final reply path for this turn"]
        LLMCall["Ask the LLM to generate a reply only when a direct grounded answer is not enough"]
        TTS["Turn the final reply text into spoken audio"]
        AudioOut["Send the spoken reply back into the live call"]
        Metrics["Record latency, failures, and call-session activity"]
    end

    subgraph Retrieval["Knowledge lookup for policy questions"]
        Hybrid["Search the policy knowledge base for the best matching content"]
        Chroma["Read candidate policy chunks from the vector database"]
        Embed["Create semantic embeddings locally or through the configured API"]
        Rank["Combine lexical and vector scores to rank the best matches"]
        Snippet["Keep only the most relevant policy sentence or excerpt"]
        Guidance["Attach turn-only grounding instructions so the model stays on-policy"]
        Direct["Store a direct grounded answer when the best excerpt can answer immediately"]
    end

    subgraph Control["Operational control and shared state"]
        API["FastAPI control plane for health checks, metrics, and telephony status"]
        Prom["Prometheus-compatible metrics endpoint"]
        Active["Shared active-session counter stored in a local file or Redis"]
        Telephony["LiveKit telephony provisioning and status management"]
    end

    Caller --> Twilio --> LiveKit --> RoomIn
    RoomIn --> VAD --> STT --> Turn --> TurnHook --> Query --> Guard
    Guard -->|non-policy| SmallTalk --> LLMNode
    Guard -->|policy turn| Hybrid
    Hybrid --> Embed
    Hybrid --> Chroma
    Hybrid --> Rank --> Snippet
    Snippet --> Guidance --> LLMNode
    Snippet --> Direct --> LLMNode
    LLMNode -->|direct answer present| TTS
    LLMNode -->|otherwise| LLMCall --> TTS
    TTS --> AudioOut --> LiveKit --> Caller

    Turn --> Metrics
    TurnHook --> Metrics
    LLMNode --> Metrics
    TTS --> Metrics
    Worker --> Metrics
    Active --> Metrics
    API --> Prom
    API --> Active
    API -.-> Telephony -.-> LiveKit
```

## Why this is the real current flow

- It shows the real runtime split between:
  - SIP / media ingress
  - media transport
  - turn detection
  - grounding
  - text generation
  - speech synthesis
  - control-plane visibility and provisioning
- It makes the current bottlenecks explicit:
  - endpointing / turn boundary
  - live LLM path
  - first grounded-turn overhead
- It also shows the real fallback/recovery shape now in code:
  - small-talk bypass
  - retrieval-guided turn-scoped developer message
  - `llm_node(...)` recovery path when that guidance is missing

## Why this shape exists

- LiveKit owns SIP/media ingress and worker dispatch.
- The worker owns turn handling, grounding, reply generation, and speech synthesis.
- Retrieval stays behind a boundary so it can remain local now and move out of process later if needed.
- The FastAPI control plane is not in the live reply path; it owns health, metrics, and telephony provisioning/status.

## LiveKit vs Pipecat

### LiveKit

Pros:
- Built-in real-time media routing
- Strong SIP / telephony support
- Native worker/session model
- Good fit for PSTN and multi-party media

Cons:
- Heavier setup than a simple Python pipeline
- Real-time debugging can be harder
- More operational surface area

Best when:
- telephony is first-class
- you need SFU/media routing
- you want built-in room/session orchestration

### Pipecat

Pros:
- Lightweight Python-first pipeline model
- Easier to prototype custom AI/media steps
- Good for non-telephony experiments

Cons:
- Does not replace a media server
- You still need separate WebRTC/SIP infrastructure
- More integration work for PSTN-grade routing

Best when:
- you want a custom Python pipeline
- telephony/media routing is handled elsewhere
- you are optimizing experimentation speed over integrated infra

## Current measured bottleneck summary

- Retrieval is no longer the dominant bottleneck in the best local/local path.
- The main remaining latency is in:
  - end-of-turn / endpointing delay
  - LLM TTFT
  - full spoken-response path

Latest proven successful default smoke:
- response latency: about `9019 ms`
- worker turn snapshot:
  - EOU: about `552 ms`
  - LLM TTFT: about `2678 ms`
  - TTS TTFB: about `216 ms`

## 1-Pager: What breaks at 1000 calls?

### What breaks first

1. LLM provider limits and latency
- the current text path is already the largest measured delay
- free/demo tiers will not sustain real high concurrency

2. Worker memory footprint
- local embeddings inside worker processes do not scale cleanly to very high worker counts

3. Local file-backed metrics/state
- JSONL turn logs and file-backed active-session state are fine for this prototype, not for large distributed scale

4. Retrieval ownership inside workers
- acceptable for this repo
- not the right final shape for large corpora and many workers

5. Honest admission control
- one overloaded worker must stop accepting work
- otherwise failures turn into slow no-answer behavior

### How to fix it

1. Move retrieval out of the worker into a shared service
2. Keep workers thin and capacity-bound
3. Replace local file-backed metrics/state with a proper shared telemetry backend
4. Use paid/provider-appropriate LLM and telephony tiers
5. Scale workers horizontally and validate at stepped concurrency, not by assumption

### Where is the bottleneck today?

- the voice / LLM path, not the vector DB path
- specifically:
  - turn-finalization delay
  - LLM TTFT

## Honest conclusion

- The current system is a working prototype with real telephony ingress and grounded answers.
- It is not yet meeting the current latency target.
- The right next improvements are operational proof and honest scaling evidence, not more speculative architecture churn.
