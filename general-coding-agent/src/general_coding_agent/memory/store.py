"""Long-term memory store (cross-thread), the IDEA doc's "structured long-term memory" tier.

Separate from the checkpointer: the checkpointer holds one conversation (thread); the Store
holds durable facts that follow the USER (preferences) or the PROJECT (repo facts) across every
thread. Local backend = SQLite; the same interface swaps to Postgres for the web/multi-user build.

Namespaces follow the IDEA resource hierarchy (tenant -> user / project):
    user-vise    -> ("memories", "users", user_id)
    project-vise -> ("memories", "projects", project_id)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Any
import hashlib
import asyncio
import json
import re
from datetime import datetime, timezone
from itertools import zip_longest
from langgraph.store.base import BaseStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_config
from general_coding_agent.config import settings
from general_coding_agent.state import ToolObservation


class MemoryCapacityExceeded(ValueError):
    """Configured retained-memory quota is exhausted; no partial mutation is allowed."""


class RunBudgetExceeded(ValueError):
    """Shared model/tool budget exhausted before another operation can begin."""


async def reserve_run_usage(store: BaseStore, context: dict, event_id: str, *,
                            models: int = 0, tools: int = 0, active_seconds: float = 0.0) -> dict:
    """Reserve a logical operation; identical resume/replay does not charge twice."""
    tenant = context.get("tenant_id", "default_tenant")
    run_id = context.get("run_id", "default_run")
    namespace = ("run_usage", namespace_label(tenant), namespace_label(run_id))
    event_key = hashlib.sha256(event_id.encode()).hexdigest()

    async def _get(k: str):
        if hasattr(store, "aget"):
            return await store.aget(namespace, k)
        return store.get(namespace, k)

    async def _put(k: str, v: dict):
        if hasattr(store, "aput"):
            await store.aput(namespace, k, v)
        else:
            store.put(namespace, k, v)

    # Idempotent replay check: identical event key does not double-count
    if await _get(event_key):
        previous = await _get("usage")
        return previous.value if previous else {}

    previous = await _get("usage")
    usage = dict(previous.value) if previous and previous.value else {"models": 0, "tools": 0, "active_seconds": 0.0}
    usage["models"] = usage.get("models", 0) + models
    usage["tools"] = usage.get("tools", 0) + tools
    usage["active_seconds"] = usage.get("active_seconds", 0.0) + active_seconds

    if usage["models"] > settings.model_call_limit or usage["tools"] > settings.tool_call_limit:
        raise RunBudgetExceeded("Shared model/tool call limit reached.")

    await _put("usage", usage)
    await _put(event_key, {"reserved": True})
    return usage



def tool_operation_ns(context: dict, history_scope: str) -> tuple[str, ...]:
    tenant_id = context.get("tenant_id", "default_tenant")
    run_id = context.get("run_id", "default_run")
    return (
        "tool_operations",
        namespace_label(tenant_id),
        namespace_label(run_id),
        namespace_label(history_scope),
    )


async def _store_get(store: BaseStore, namespace: tuple[str, ...], key: str) -> dict | None:
    if hasattr(store, "aget"):
        item = await store.aget(namespace, key)
    else:
        item = store.get(namespace, key)
    return item.value if item else None


async def _store_put(store: BaseStore, namespace: tuple[str, ...], key: str, value: dict) -> None:
    if hasattr(store, "aput"):
        await store.aput(namespace, key, value)
    else:
        store.put(namespace, key, value)


async def _store_search(store: BaseStore, namespace: tuple[str, ...], limit: int = 50) -> list:
    if hasattr(store, "asearch"):
        return await store.asearch(namespace, limit=limit)
    return store.search(namespace, limit=limit)


async def claim_tool_intent(
    store: BaseStore,
    context: dict,
    history_scope: str,
    tool_call_id: str,
    tool_name: str,
    args_hash: str,
    preconditions: dict | None = None,
    expected_post_hash: str | None = None,
) -> tuple[Literal["new", "replay", "conflict", "reconcile", "blocked"], dict]:
    """Atomically claim intent for a tool operation; checking then blindly upserting is not a claim."""
    namespace = tool_operation_ns(context, history_scope)
    existing = await _store_get(store, namespace, tool_call_id)
    if existing:
        if existing.get("args_hash") != args_hash:
            return "conflict", existing
        st = existing.get("status")
        if st in ("succeeded", "failed"):
            return "replay", existing
        if st in ("started", "waiting_for_approval"):
            return "reconcile", existing
        return "blocked", existing

    record = {
        "operation_id": f"{history_scope}:{tool_call_id}",
        "run_id": context.get("run_id", ""),
        "history_scope": history_scope,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "args_hash": args_hash,
        "status": "started",
        "ok": False,
        "exit_code": None,
        "artifact_id": None,
        "bytes": 0,
        "truncated": False,
        "error_type": None,
        "result_content": "",
        "preconditions": preconditions or {},
        "expected_post_hash": expected_post_hash,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "evidence_refs": [f"{history_scope}:{tool_call_id}"],
    }
    await _store_put(store, namespace, tool_call_id, record)
    return "new", record


async def persist_tool_result(
    store: BaseStore,
    context: dict,
    history_scope: str,
    tool_call_id: str,
    status: Literal["succeeded", "failed", "outcome_unknown"],
    observation: ToolObservation,
) -> dict:
    """Update operation ledger with final status, normalized observation, and evidence refs."""
    namespace = tool_operation_ns(context, history_scope)
    existing = await _store_get(store, namespace, tool_call_id) or {}
    existing["status"] = status
    existing["ok"] = observation.ok
    existing["exit_code"] = observation.exit_code
    existing["artifact_id"] = observation.artifact_id
    existing["bytes"] = observation.bytes
    existing["truncated"] = observation.truncated
    existing["error_type"] = observation.error_type
    existing["result_content"] = observation.content
    existing["completed_at"] = datetime.now(timezone.utc).isoformat()
    refs = [f"{history_scope}:{tool_call_id}"]
    if observation.artifact_id:
        refs.append(observation.artifact_id)
    existing["evidence_refs"] = refs

    await _store_put(store, namespace, tool_call_id, existing)
    return existing



async def get_tool_operation(
    store: BaseStore,
    context: dict,
    history_scope: str,
    tool_call_id: str,
) -> dict | None:
    namespace = tool_operation_ns(context, history_scope)
    item = await store.aget(namespace, tool_call_id)
    return item.value if item else None


def _expired(value: dict) -> bool:
    expiry = value.get("expires_at")
    if not expiry:
        return False
    try:
        instant = datetime.fromisoformat(expiry)
        return instant.tzinfo is None or instant <= datetime.now(timezone.utc)
    except (TypeError, ValueError):
        # Malformed persisted expiry is not permission to recall the fact forever.
        return True


def namespace_label(value: str) -> str:
    """SQLite Store uses periods as separators; identities must remain single labels."""
    if not isinstance(value, str) or not value:
        raise ValueError("Namespace identity must be a nonempty string")
    return "~" + value.encode("utf-8").hex() if "." in value or value.startswith("~") else value


async def put_immutable(store: BaseStore, namespace: tuple[str, ...], key: str, value: dict) -> None:
    """Insert once, accepting identical replay but never replacing original evidence."""
    current = await _store_get(store, namespace, key)
    if current is None:
        await _store_put(store, namespace, key, value)
    elif current != value:
        raise ValueError("Immutable evidence conflicts with the existing record")



def user_ns(user_id: str | None = None) -> tuple[str, ...]:
    return ("memories", "users", namespace_label(user_id or settings.user_id))


def project_ns(project_id: str | None = None) -> tuple[str, ...]:
    return ("memories", "projects", namespace_label(project_id or settings.project_id))


def ns_for(scope: str) -> tuple[str, ...]:
    """Derive tool scope from trusted invocation config, never model arguments."""
    if scope not in ("user", "project"):
        raise ValueError("Memory scope must be user or project")
    try:
        context = get_config().get("configurable", {}).get("run_context")
    except RuntimeError:
        context = None
    if context is None:
        return project_ns() if scope == "project" else user_ns()
    # Keep the existing local CLI's records readable without copying or deleting data.
    # Other identities never fall back to this legacy namespace.
    if (context["tenant_id"] == "default_tenant" and context["user_id"] == settings.user_id
            and context["workspace_id"] == str(settings.workspace_root)):
        return project_ns() if scope == "project" else user_ns()
    owner = context["user_id"] if scope == "user" else context["workspace_id"]
    return ("memories", namespace_label(context["tenant_id"]), scope, namespace_label(owner))


async def recall_memories(store: BaseStore, query: str = "", scope: str | None = None,
                          *, match_required: bool = True) -> str:
    """Rank scoped facts and render one bounded response for prompts and explicit recall."""
    scopes = [scope] if scope else ["project", "user"]
    terms = set(re.findall(r"\w+", query.casefold()))
    ranked = []
    capped = False
    for current_scope in scopes:
        namespace = ns_for(current_scope)
        candidates = []
        # ponytail: bounded lexical scan; use indexed retrieval when recall/latency
        # evaluations show the configured scan ceiling is insufficient.
        for offset in range(0, settings.memory_scan_records, 100):
            count = min(100, settings.memory_scan_records - offset)
            page = await store.asearch(namespace, limit=count, offset=offset)
            for item in page:
                if item.namespace != namespace:
                    continue
                if item.value.get("retired") or _expired(item.value):
                    continue
                content = item.value.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    continue
                words = set(re.findall(r"\w+", content.casefold()))
                score = len(terms & words) * 2
                for t in terms:
                    if len(t) >= 4:
                        for w in words:
                            if len(w) >= 4 and (t.startswith(w) or w.startswith(t)):
                                score += 1
                if match_required and terms and not score:
                    continue
                candidates.append((score, item.updated_at, item.key, item.value.get("revision", 0), content))
            if len(page) < count:
                break
        else:
            capped = True
        candidates.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        ranked.append([(current_scope, key, revision, content) for _, _, key, revision, content in candidates])

    note = "[Memory scan capped; records beyond the configured scan limit were not searched.]" if capped else ""
    note = note[:settings.memory_context_chars]
    remaining = settings.memory_context_chars - len(note) - bool(note)
    lines = []
    seen = set()
    # Alternate scopes so a large user profile cannot exclude all project context.
    for group in zip_longest(*ranked):
        for item in group:
            if item is None:
                continue
            current_scope, key, revision, content = item
            identity = (current_scope, content)
            if identity in seen:
                continue
            line = f"- [{current_scope} id={key} revision={revision}] {content}"
            if len(line) + 1 > remaining:
                continue
            lines.append(line)
            seen.add(identity)
            remaining -= len(line) + 1
            if len(lines) == settings.memory_recall_records:
                return "\n".join([*lines, note] if note else lines)
    return "\n".join([*lines, note] if note else lines)


async def _replace_memory(store: BaseStore, namespace: tuple[str, ...], key: str,
                          previous: dict | None, value: dict) -> bool:
    """Compare-and-swap the fact and its audit revisions in one storage transaction."""
    revisions = ("memory_revisions", *namespace[1:], namespace_label(key))
    records = [(revisions, str(value["revision"]), value)]
    if previous is not None:
        records.insert(0, (revisions, str(previous.get("revision", 0)), previous))
    event_namespace = ("memory_events", *namespace[1:])
    event = value.get("source_event")
    if event:
        records.append((event_namespace, event, {"memory_id": key, "revision": value["revision"]}))
    if event and await _store_get(store, event_namespace, event):
        return False
    curr = await _store_get(store, namespace, key)
    if curr != previous:
        return False
    existing_facts = await _store_search(store, namespace, limit=settings.memory_max_records)
    if previous is None and len(existing_facts) >= settings.memory_max_records:
        raise MemoryCapacityExceeded("Retained fact quota reached.")
    existing_revs = await _store_search(store, revisions[:-1], limit=settings.memory_max_revisions)
    additional = 1 + int(previous is not None and "revision" not in previous)
    if len(existing_revs) + additional > settings.memory_max_revisions:
        raise MemoryCapacityExceeded("Retained revision quota reached.")
    for ns, revision, record in records:
        await _store_put(store, ns, revision, record)
    await _store_put(store, namespace, key, value)
    return True



SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("OpenAI/LLM API Key", re.compile(r"\b(sk-[a-zA-Z0-9_\-]{20,})\b")),
    ("AWS Access Key ID", re.compile(r"\b(AKIA[0-9A-Z]{16})\b")),
    ("GitHub Token", re.compile(r"\b(gh[pousr]_[a-zA-Z0-9]{36,})\b")),
    ("Private Key", re.compile(r"-----BEGIN (?:[A-Z0-9_\-]+ )?PRIVATE KEY-----")),
    ("Slack Token", re.compile(r"\b(xox[baprs]-[0-9a-zA-Z]{10,})\b")),
    ("Secret/Password Assignment", re.compile(r"(?i)\b(?:password|passwd|api_key|apikey|secret_key|private_key|auth_token)\s*[:=]\s*['\"][^'\"]{4,}['\"]")),
]


def _detect_secrets(text: str) -> str | None:
    """Detect known credential patterns and return masked label, or None if clean."""
    for label, pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            raw = match.group(0)
            masked = raw[:4] + "..." + raw[-2:] if len(raw) > 6 else "***"
            return f"{label} ({masked})"
    return None


async def apply_memory(store: BaseStore, action: str, content: str = "", scope: str = "user",
                       memory_id: str | None = None, expected_revision: int | None = None,
                       expires_at: str | None = None, *, source: dict | None = None) -> str:
    """One mutation path for explicit tools and runtime memory proposals."""
    if action not in ("create", "update", "delete"):
        return "ERROR: invalid memory action."
    content = content.strip()
    if action != "delete" and (not content or len(content) > settings.memory_content_chars):
        return f"ERROR: content must be nonblank and at most {settings.memory_content_chars} characters."
    if action in ("create", "update"):
        detected_secret = _detect_secrets(content)
        if detected_secret:
            return f"ERROR: Content rejected. Sensitive credential pattern detected: {detected_secret}. Never store secrets in durable memory."
    if expires_at is not None:
        try:
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
                raise ValueError("Expiry must be a future instant with timezone")
            expires_at = expiry.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            return "ERROR: expires_at must be a future ISO-8601 timestamp with timezone."
    namespace = ns_for(scope)
    source = source or {}
    event = source.get("event_id")
    event_namespace = ("memory_events", *namespace[1:])
    if event and await store.aget(event_namespace, event):
        return "Already applied this source event."
    if action == "create":
        # Preserve existing random-ID memories; identical creates must reuse their ID.
        matches = await store.asearch(namespace, filter={"content": content}, limit=1)
        if matches and matches[0].namespace == namespace:
            if matches[0].value.get("retired") or _expired(matches[0].value):
                return "CONFLICT: this memory was retired; do not recreate it automatically."
            return f"Already remembered {matches[0].key} ({scope}, revision={matches[0].value.get('revision', 0)}): {content}"
        key = hashlib.sha256(content.encode("utf-8")).hexdigest()
        previous = await store.aget(namespace, key)
        if previous and (previous.value.get("content") != content or previous.value.get("retired")):
            return "CONFLICT: this memory was corrected; search current memories before recreating its old value."
    else:
        if not memory_id:
            return f"ERROR: memory_id is required to {action}."
        key = memory_id
        previous = await store.aget(namespace, key)
        if previous is None:
            return f"ERROR: no {scope} memory with id={key}."
        if type(expected_revision) is not int or expected_revision != previous.value.get("revision", 0):
            return "CONFLICT: supply the current expected_revision from search_memory before changing this fact."
        if previous.value.get("retired"):
            return "CONFLICT: this memory was retired."
        if source.get("received_at") and source["received_at"] < previous.value.get("source_received_at", ""):
            return "CONFLICT: this source predates the current correction."
        if (action == "update" and previous.value.get("content") == content
                and (expires_at is None or expires_at == previous.value.get("expires_at"))):
            return f"Already remembered {key} ({scope}): {content}"
    old_value = previous.value if previous else None
    try:
        context = get_config().get("configurable", {}).get("run_context")
    except RuntimeError:
        context = None
    revision = (old_value.get("revision", 0) if old_value else 0) + 1
    value = {"content": old_value["content"] if action == "delete" else content, "scope": scope,
             "revision": revision, "retired": action == "delete",
             "expires_at": expires_at if expires_at is not None else (old_value or {}).get("expires_at"),
             "recorded_at": datetime.now(timezone.utc).isoformat(), "source_run": context,
             "source_event": event, "source_message": source.get("message_id"),
             "source_received_at": source.get("received_at") or datetime.now(timezone.utc).isoformat()}
    try:
        applied = await _replace_memory(store, namespace, key, old_value, value)
    except MemoryCapacityExceeded as exc:
        return f"ERROR: {exc} Review retention or increase the configured quota; nothing was changed."
    if not applied:
        if event and await store.aget(event_namespace, event):
            return "Already applied this source event."
        return "CONFLICT: memory changed concurrently; search current memories before retrying."
    if action == "delete":
        return f"Retired memory {key} ({scope}, revision={revision}); retained audit history."
    return f"{action}d memory {key} ({scope}, revision={revision}): {content}"


@asynccontextmanager
async def get_memory_store(db_path: Path | str | None = None) -> AsyncIterator[BaseStore]:
    """Yield the local store; a hosted adapter must also implement transactional memory updates."""
    path = str(db_path or settings.store_db_path)
    async with AsyncSqliteStore.from_conn_string(path) as store:
        if path == ":memory:":
            await store.setup()
        else:
            # Native setup locks only one connection: concurrent processes can replay
            # the same migration. Serialize initialization, not normal store operations.
            # ponytail: local Unix file lock; hosted adapters own their migration deployment.
            import fcntl
            with open(str(Path(path).resolve()) + ".setup.lock", "a") as lock:
                async with asyncio.timeout(30):
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            await asyncio.sleep(0.05)
                    await store.setup()
        yield store
