"""Durable tool execution boundary and operation ledger.

Intercepts tool execution at the ToolNode boundary:
1. Validates single tool call at a time.
2. Enforces static tool capability/effect policies.
3. Checks ledger receipts: replays succeeded results and detects argument conflicts.
4. Atomically reserves run budget and claims intent in the Store.
5. Reconciles interrupted file operations and blocks on uncertain host/external effects.
6. Normalizes observations, preserves exit codes, and prevents errors from becoming successful receipts.
"""

import asyncio
import hashlib
import json
import re
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_store
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from general_coding_agent.memory.store import claim_tool_intent, persist_tool_result, reserve_run_usage, tool_operation_ns
from general_coding_agent.state import ToolEffect, ToolObservation
from general_coding_agent.tools.fs import _fuzzy_find_and_replace, _hash, _read_text, _resolve_safe_path
from general_coding_agent.tools.registry import get_tool_policy


class OperationBlocked(Exception):
    """Raised when an operation is blocked due to outcome_unknown or conflicting state."""


class RecursionRefusal(ValueError):
    """Raised when a non-main worker attempts subagent task delegation."""
    pass


def canonical_args_hash(args: Any) -> str:
    """Deterministic SHA-256 hash of tool call arguments."""
    normalized = args if isinstance(args, dict) else {}
    canonical_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_file_preconditions(tool_name: str, args: dict) -> tuple[dict, str | None]:
    """Capture file existence/hash before write and compute expected post-write hash."""
    if tool_name not in ("fs_write", "fs_edit"):
        return {}, None

    path_str = args.get("path")
    if not path_str:
        return {}, None

    try:
        target, err = _resolve_safe_path(path_str, op="check")
        if err or not target:
            return {"path": path_str, "error": err}, None
    except Exception as exc:
        return {"path": path_str, "error": str(exc)}, None

    file_exists = target.exists() and not target.is_dir()
    pre_hash, current_text = None, None
    if file_exists:
        try:
            current_text = _read_text(target)
            pre_hash = _hash(current_text)
        except Exception:
            pass

    preconditions = {
        "path": path_str,
        "file_exists": file_exists,
        "pre_hash": pre_hash,
        "expected_hash": args.get("expected_hash"),
    }

    expected_post_hash = None
    if tool_name == "fs_write":
        expected_post_hash = _hash(args.get("content", ""))
    elif tool_name == "fs_edit" and current_text is not None:
        target_text = args.get("target_text", "")
        replacement = args.get("replacement", "")
        if target_text:
            new_text, _ = _fuzzy_find_and_replace(current_text, target_text, replacement)
            if new_text is not None:
                expected_post_hash = _hash(new_text)

    return preconditions, expected_post_hash


def classify_tool_result(tool_name: str, result: Any, operation_id: str) -> ToolObservation:
    """Classify result into ToolObservation, preventing errors from becoming successful receipts."""
    if isinstance(result, ToolMessage):
        content = str(result.content or "")
        status = getattr(result, "status", "success")
        ok = status != "error"
        error_type = "tool_error" if not ok else None
        exit_code = None

        if tool_name == "shell_exec":
            match = re.search(r"^exit_code=(-?\d+)", content, re.MULTILINE)
            if match:
                exit_code = int(match.group(1))
                if exit_code != 0:
                    ok = False
                    error_type = "nonzero_exit"
            elif content.startswith("ERROR:") or content.startswith("COMMAND REJECTED"):
                ok = False
                error_type = "command_rejected" if "REJECTED" in content else "tool_error"

        if content.startswith(("ERROR:", "PERMISSION DENIED:", "CONFLICT:", "COMMAND REJECTED BY USER:")):
            ok = False
            error_type = error_type or "tool_error"

        artifact_match = re.search(r"offloaded to artifact ([a-f0-9]+)", content)
        artifact_id = artifact_match.group(1) if artifact_match else None

        return ToolObservation(
            ok=ok,
            operation_id=operation_id,
            content=content,
            exit_code=exit_code,
            artifact_id=artifact_id,
            bytes=len(content.encode("utf-8")),
            truncated="[line truncated]" in content or "..." in content,
            error_type=error_type,
        )

    if isinstance(result, Command):
        return ToolObservation(ok=True, operation_id=operation_id, content="Command update", bytes=len(str(result).encode("utf-8")))

    content = str(result or "")
    ok = not content.startswith(("ERROR:", "CONFLICT:", "PERMISSION DENIED:", "COMMAND REJECTED"))
    return ToolObservation(
        ok=ok,
        operation_id=operation_id,
        content=content,
        bytes=len(content.encode("utf-8")),
        error_type=None if ok else "tool_error",
    )


def recreate_tool_message(record: dict, tool_call_id: str, tool_name: str) -> ToolMessage:
    """Reconstruct ToolMessage from saved ledger record for replayed calls."""
    content = record.get("result_content", "")
    op_id = record.get("operation_id", tool_call_id)
    if f"[receipt: {op_id}]" not in content and f"[receipt={op_id}]" not in content:
        content = f"{content}\n[receipt: {op_id}]" if content else f"[receipt: {op_id}]"

    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=tool_name,
        status="success" if record.get("ok", False) else "error",
        additional_kwargs={"operation_id": op_id, "replayed": True},
    )


async def reconcile_started_operation(
    store: Any,
    context: dict,
    history_scope: str,
    tool_call_id: str,
    record: dict,
    tool_name: str,
    raw_args: dict,
) -> tuple[Literal["replay", "execute", "blocked"], dict]:
    """Reconcile an operation in 'started' state after crash or interruption."""
    operation_id = record.get("operation_id", f"{history_scope}:{tool_call_id}")

    if tool_name in ("fs_write", "fs_edit"):
        path_str = raw_args.get("path")
        if not path_str:
            return "blocked", record

        try:
            target, err = _resolve_safe_path(path_str, op="check")
            if err or not target:
                return "blocked", record
        except Exception:
            return "blocked", record

        expected_post_hash = record.get("expected_post_hash")
        pre = record.get("preconditions", {})
        pre_hash = pre.get("pre_hash")
        file_existed = pre.get("file_exists", False)

        current_exists = target.exists() and not target.is_dir()
        current_hash = None
        if current_exists:
            try:
                current_hash = _hash(_read_text(target))
            except Exception:
                pass

        if expected_post_hash and current_hash == expected_post_hash:
            if tool_name == "fs_write":
                created = not file_existed
                nbytes = len(raw_args.get("content", "").encode("utf-8"))
                obs_content = f"{'created' if created else 'overwrote'} {path_str} (hash={expected_post_hash}, bytes={nbytes})\n[receipt: {operation_id}]"
            else:
                obs_content = f"edited {path_str} (hash={expected_post_hash})\n[reconciled edit]\n[receipt: {operation_id}]"

            obs = ToolObservation(ok=True, operation_id=operation_id, content=obs_content, bytes=len(obs_content.encode("utf-8")))
            updated = await persist_tool_result(store, context, history_scope, tool_call_id, "succeeded", obs)
            return "replay", updated

        if (not file_existed and not current_exists) or (file_existed and current_hash == pre_hash):
            return "execute", record

        obs = ToolObservation(
            ok=False,
            operation_id=operation_id,
            content=f"CONFLICT: file state changed unexpectedly during started operation on {path_str}.",
            error_type="conflicting_state",
        )
        updated = await persist_tool_result(store, context, history_scope, tool_call_id, "outcome_unknown", obs)
        return "blocked", updated

    if record.get("status") == "waiting_for_approval":
        return "execute", record

    policy = get_tool_policy(tool_name)
    if policy and policy.effect in (ToolEffect.HOST_EXEC, ToolEffect.EXTERNAL):
        obs = ToolObservation(
            ok=False,
            operation_id=operation_id,
            content=f"Operation {operation_id} interrupted during host/external execution; outcome unknown.",
            error_type="outcome_unknown",
        )
        updated = await persist_tool_result(store, context, history_scope, tool_call_id, "outcome_unknown", obs)
        return "blocked", updated

    return "execute", record


async def durable_tool_wrapper(request: ToolCallRequest, execute: Callable) -> ToolMessage | Command:
    """ToolNode awrap_tool_call implementation enforcing durable ledger semantics."""
    state = request.state
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    tool_call_id = tool_call["id"]
    raw_args = tool_call.get("args") or {}

    history_scope = state.get("history_scope", "main")

    # 0. Strict Subagent Depth-1 Recursion Lock
    if history_scope != "main" and tool_name in (
        "delegate", "spawn_subagent", "create_worker", "call_subagent", "dispatch_worker"
    ):
        return ToolMessage(
            content="RecursionRefusal: Subagents cannot delegate tasks. Complete this task directly.",
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    # 1. Validate one tool call at a time
    messages = state.get("messages", [])
    last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if last and len(getattr(last, "tool_calls", [])) > 1:
        return ToolMessage(
            content="ERROR: call one tool at a time so approvals and state changes execute in order.",
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    # 2. Load policy
    policy = get_tool_policy(tool_name)
    if policy is None:
        return ToolMessage(
            content=f"ERROR: No tool execution policy defined for '{tool_name}'.",
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    is_state_control = policy.effect == ToolEffect.STATE_CONTROL
    history_scope = state.get("history_scope", "main")
    run_context = state.get("run_context") or {}
    operation_id = f"{history_scope}:{tool_call_id}"

    store = getattr(getattr(request, "runtime", None), "store", None)
    if store is None:
        try:
            store = get_store()
        except Exception:
            store = None

    args_hash = canonical_args_hash(raw_args)

    durable_active = (not is_state_control) and (store is not None) and bool(run_context)
    record = {}

    # 3. Inspect receipt (state-control bypasses external-effect replay)
    if durable_active:
        preconditions, expected_post_hash = get_file_preconditions(tool_name, raw_args)
        disp, record = await claim_tool_intent(
            store, run_context, history_scope, tool_call_id, tool_name, args_hash,
            preconditions=preconditions, expected_post_hash=expected_post_hash,
        )

        if disp == "conflict":
            if record.get("status") == "waiting_for_approval" or record.get("ticket"):
                raise OperationBlocked(
                    f"Operation {operation_id} ({tool_name}) approval ticket mismatch! "
                    f"Recorded hash={record.get('args_hash')}, current={args_hash}. Tampered resume detected."
                )
            return ToolMessage(
                content=f"CONFLICT: Tool call '{tool_call_id}' already executed with different arguments.",
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
            )
        if disp == "replay":
            return recreate_tool_message(record, tool_call_id, tool_name)
        if disp == "blocked":
            raise OperationBlocked(f"Operation {operation_id} ({tool_name}) is in blocked/outcome_unknown state.")
        if disp == "reconcile":
            action, rec = await reconcile_started_operation(
                store, run_context, history_scope, tool_call_id, record, tool_name, raw_args,
            )
            if action == "replay":
                return recreate_tool_message(rec, tool_call_id, tool_name)
            if action == "blocked":
                raise OperationBlocked(f"Operation {operation_id} ({tool_name}) failed reconciliation and is blocked.")
            if action == "execute":
                record = rec
                if record.get("ticket"):
                    current_ticket = canonical_args_hash({"tool": tool_name, "args": raw_args})
                    if current_ticket != record["ticket"]:
                        raise OperationBlocked(
                            f"Operation {operation_id} ({tool_name}) approval ticket mismatch! "
                            f"Recorded ticket={record['ticket']}, current={current_ticket}. Tampered resume detected."
                        )

        event_key = f"tool:{history_scope}:{last.id if last else 'unknown'}:{tool_call_id}"
        await reserve_run_usage(store, run_context, event_key, tools=1)

    # 4. Execute with precise error handling (re-raise GraphBubbleUp, propagate cancellation)
    try:
        result = await execute(request)
    except GraphBubbleUp:
        if durable_active:
            ticket = canonical_args_hash({"tool": tool_name, "args": raw_args})
            paused_rec = {**record, "status": "waiting_for_approval", "ticket": ticket}
            await store.aput(tool_operation_ns(run_context, history_scope), tool_call_id, paused_rec)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        if durable_active:
            obs = ToolObservation(ok=False, operation_id=operation_id, content="Execution cancelled.", error_type="cancelled")
            await persist_tool_result(store, run_context, history_scope, tool_call_id, "outcome_unknown", obs)
        raise
    except Exception as exc:
        if durable_active:
            if policy.effect in (ToolEffect.HOST_EXEC, ToolEffect.EXTERNAL):
                obs = ToolObservation(ok=False, operation_id=operation_id, content=f"Error during host/external execution: {exc}", error_type="outcome_unknown")
                await persist_tool_result(store, run_context, history_scope, tool_call_id, "outcome_unknown", obs)
                raise OperationBlocked(f"Operation {operation_id} ({tool_name}) failed with possible external effect: {exc}") from exc

            obs = ToolObservation(ok=False, operation_id=operation_id, content=f"ERROR: {exc}", error_type=type(exc).__name__)
            await persist_tool_result(store, run_context, history_scope, tool_call_id, "failed", obs)

        return ToolMessage(content=f"ERROR: {exc}", name=tool_name, tool_call_id=tool_call_id, status="error")

    # 5. Classify and persist result
    obs = classify_tool_result(tool_name, result, operation_id)
    if durable_active:
        status_to_persist = "succeeded" if obs.ok else "failed"
        await persist_tool_result(store, run_context, history_scope, tool_call_id, status_to_persist, obs)


    # 7. Return original ToolMessage/Command shape with receipt ID included
    if isinstance(result, ToolMessage):
        receipt_marker = f"[receipt: {operation_id}]"
        new_content = result.content if receipt_marker in result.content or f"[receipt={operation_id}]" in result.content else f"{result.content}\n{receipt_marker}"
        extra = dict(result.additional_kwargs or {})
        extra["operation_id"] = operation_id
        return ToolMessage(
            content=new_content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            additional_kwargs=extra,
        )

    return result
