"""Durable checkpointer factory for general_coding_agent.

Uses AsyncSqliteSaver for thread-level state persistence, allowing pause/resume,
crash recovery, and session time travel.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import json
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore
from general_coding_agent.config import settings
from general_coding_agent.state import AgentState, CompactedSegment, RunContext
from general_coding_agent.memory.store import namespace_label, put_immutable


def _summary_namespace(context: dict, history_scope: str | None = None) -> tuple[str, ...]:
    identity = RunContext(**context)
    namespace = ("compactions", *(namespace_label(part) for part in
                                 (identity.tenant_id, identity.user_id, identity.workspace_id, identity.thread_id)))
    return (*namespace, "workers", namespace_label(history_scope)) if history_scope else namespace


def message_reference(message: AnyMessage) -> str:
    """Bind coverage to content as well as message ID; IDs alone permit replacement."""
    payload = json.dumps(message.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def save_compaction(store: BaseStore, context: dict, segment: CompactedSegment,
                          history_scope: str | None = None) -> None:
    """Publish an immutable summary before its pointer can enter a checkpoint."""
    value = segment.model_dump(exclude={"summary_id"})
    key = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    namespace = _summary_namespace(context, history_scope)
    # Content addressing makes retries idempotent and concurrent different summaries distinct.
    await put_immutable(store, namespace, key, value)
    segment.summary_id = key


async def read_history_page(
    state: AgentState, store: BaseStore | None, kind: str = "messages", offset: int = 0,
    char_offset: int = 0, limit: int = 4000, summary_id: str | None = None, worker_id: str | None = None,
) -> str:
    """Read one bounded piece; original native message fields remain reconstructible."""
    if kind not in ("messages", "summary") or offset < 0 or char_offset < 0 or limit <= 0:
        return "ERROR: invalid history kind or cursor; offsets must be nonnegative and limit positive."
    worker = None
    if worker_id:
        if store is None or not state.get("run_context"):
            return "ERROR: scoped history store is unavailable."
        if worker_id == "main":
            worker = await store.aget(("conversation_history", *_summary_namespace(state["run_context"])[1:]), "head")
        else:
            worker = await store.aget(("worker_runs", *_summary_namespace(state["run_context"])[1:]), worker_id)
        if worker is None:
            return "ERROR: worker history not found in this conversation."
    if kind == "messages" and worker:
        total = worker.value["message_count"]
        if offset >= total:
            return json.dumps({"kind": kind, "total_messages": total, "next": None, "text": ""})
        namespace = (("conversation_history", *_summary_namespace(state["run_context"])[1:]) if worker_id == "main"
                     else ("worker_history", *_summary_namespace(state["run_context"], worker_id)[1:]))
        item = await store.aget(namespace, str(offset))
        if item is None:
            return "ERROR: archived worker message is missing."
        value = item.value["message"]
        metadata = {"kind": kind, "offset": offset, "total_messages": total, "source": item.value["source"]}
    elif kind == "messages":
        messages = state.get("messages", [])
        total = len(messages)
        if offset >= len(messages):
            return json.dumps({"kind": kind, "total_messages": len(messages), "next": None, "text": ""})
        value = messages[offset].model_dump(mode="json")
        metadata = {"kind": kind, "offset": offset, "total_messages": len(messages),
                    "source": message_reference(messages[offset])}
    else:
        current = worker.value if worker else (state.get("segment") or {})
        key = summary_id or current.get("summary_id")
        if not key or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            return "ERROR: provide a valid summary_id from the current context or its parent summary."
        if store is None or not state.get("run_context"):
            return "ERROR: scoped history store is unavailable."
        history_scope = None if worker_id == "main" else (worker_id or state.get("history_scope"))
        item = await store.aget(_summary_namespace(state["run_context"], history_scope), key)
        if item is None:
            return "ERROR: summary not found in this conversation."
        value = item.value
        metadata = {"kind": kind, "summary_id": key, "parent_summary_id": value.get("parent_summary_id")}
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if char_offset > len(text):
        return "ERROR: char_offset exceeds this record."
    end = min(len(text), char_offset + min(limit, settings.artifact_read_chars))
    if end < len(text):
        next_page = {"kind": kind, "offset": offset, "char_offset": end}
        if kind == "summary":
            next_page["summary_id"] = metadata["summary_id"]
    elif kind == "messages" and offset + 1 < total:
        next_page = {"kind": kind, "offset": offset + 1, "char_offset": 0}
    else:
        next_page = None
    if worker_id:
        metadata["worker_id"] = worker_id
        if next_page:
            next_page["worker_id"] = worker_id
    return json.dumps({**metadata, "char_offset": char_offset, "text": text[char_offset:end],
                       "next": next_page}, ensure_ascii=False)


async def save_worker_result(store: BaseStore, context: dict, worker_id: str, name: str,
                              state: AgentState, response: str, status: str = "returned") -> dict:
    """Persist the detailed transcript before making a bounded handoff visible to the parent."""
    namespace = ("worker_history", *_summary_namespace(context, worker_id)[1:])
    messages = state.get("messages", [])
    for offset, message in enumerate(messages):
        await put_immutable(store, namespace, str(offset), {
            "message": message.model_dump(mode="json"), "source": message_reference(message)})
    result = {"worker_id": worker_id, "worker_name": name, "status": status,
              "message_count": len(messages), "final_message_offset": len(messages) - 1,
              "summary_id": (state.get("segment") or {}).get("summary_id"),
              "response": response[:settings.artifact_read_chars],
              "response_truncated": len(response) > settings.artifact_read_chars}
    await put_immutable(store, ("worker_runs", *_summary_namespace(context)[1:]), worker_id, result)
    return result


async def archive_main_history(store: BaseStore, state: AgentState) -> int:
    """Archive only new parent messages; publish the readable count after all writes succeed."""
    namespace = ("conversation_history", *_summary_namespace(state["run_context"])[1:])
    messages = state["messages"]
    base = state.get("history_offset", 0)
    for offset in range(max(0, state.get("main_history_upto", 0) - base), len(messages)):
        await put_immutable(store, namespace, str(base + offset), {
            "message": messages[offset].model_dump(mode="json"), "source": message_reference(messages[offset])})
    await store.aput(namespace, "head", {"message_count": base + len(messages),
                                       "summary_id": (state.get("segment") or {}).get("summary_id")})
    return base + len(messages)


@asynccontextmanager
async def get_checkpointer(
    db_path: Path | str | None = None,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Yield a connected AsyncSqliteSaver checkpointer instance.
    
    In local development, uses SQLite. In production, swapping to PostgresSaver
    requires replacing this context manager without changing graph logic.
    """
    path = str(db_path or settings.memory_db_path)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        yield saver
