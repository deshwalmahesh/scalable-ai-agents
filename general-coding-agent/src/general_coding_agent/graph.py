"""The agent graph.

    START -> capture_memories -> supervisor -> (agent | END)
                              |
                           agent = create_react_agent(model, tools, store)
                              with a pre_model_hook that compacts context non-destructively.

- Supervisor: triages, activates skills, and LOADS long-term memories (user + project) into state.
- Agent: the ReAct worker. Tools include the planner (write_todos), the fs/CLI mix, MCP, and the
  memory writer (remember). create_react_agent owns the model<->tool loop and step bounding.
- Compaction (IDEA doc): triggered by TOKEN budget. Builds a versioned, structured CompactedSegment
  (facts / open tasks / file changes / unresolved questions) from older messages and feeds the model
  a compacted VIEW via `llm_input_messages`. Supervisor prefixes leave the active checkpoint
  only after immutable originals are archived; workers have separate native checkpoints.
"""

import asyncio
import json
import logging
import uuid
import hashlib
import time
import re
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_store, get_config
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError, GraphBubbleUp
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator, model_validator

from general_coding_agent.config import settings
from general_coding_agent.llm import get_chat_model
from general_coding_agent.memory.store import recall_memories, apply_memory, ns_for, namespace_label, reserve_run_usage, RunBudgetExceeded
from general_coding_agent.memory.checkpointer import message_reference, save_compaction, save_worker_result, archive_main_history
from general_coding_agent.prompts import AGENT_PROMPT, COMPACTION_PROMPT, MEMORY_CAPTURE_PROMPT, get_supervisor_prompt
from general_coding_agent.skills.manager import skill_manager
from general_coding_agent.state import AgentState, CompactedSegment, MemoryCapture, RunContext, RunOutcome, RunStatusLiteral
from general_coding_agent.tools.artifacts import set_current_prompt_tokens
from general_coding_agent.tools.execution import durable_tool_wrapper, OperationBlocked
from general_coding_agent.tools.registry import AGENT_SPECS, filter_tools_for_session

log = logging.getLogger(__name__)


class ContextBudgetExceeded(ValueError):
    """Essential input cannot fit the configured model context allowance."""


class Triage(BaseModel):
    """Supervisor's dynamic routing and delegation decision."""

    route: str = Field(
        description="Destination worker name from Available Workers, or 'end' when complete."
    )
    skills: list[str] = Field(default_factory=list, description="Skill names from SKILL CATALOGUE to activate")
    worker_instruction: str | None = Field(default=None, description="Clear, focused instruction for the delegated agent")
    direct_response: str | None = Field(default=None, description="The final answer/summary when route == 'end'")

    @field_validator("route")
    @classmethod
    def validate_route(cls, v: str) -> str:
        clean = v.strip().lower().strip("'\"")
        valid = set(AGENT_SPECS.keys()) | {"end"}
        if clean not in valid:
            raise ValueError(f"Invalid route '{v}'. Must be one of {valid}")
        return clean

    @model_validator(mode="after")
    def validate_decision(self) -> "Triage":
        field = "direct_response" if self.route == "end" else "worker_instruction"
        value = getattr(self, field)
        if not value or not value.strip():
            raise ValueError(f"{field} must be nonblank for route '{self.route}'")
        setattr(self, field, value.strip())
        return self


def content_text(content: Any) -> str:
    """Flatten message content to plain text (Gemini returns a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def approx_tokens(messages: list[AnyMessage]) -> int:
    """Local estimate, including framing, content blocks, tool calls and call IDs.

    ponytail: this is not a provider tokenizer guarantee; calibrated/native accounting
    is required before accepting arbitrary media and multilingual context at capacity.
    """
    count = count_tokens_approximately(messages)
    # The library assumes list content already contains tool-use blocks (Anthropic).
    # Gemini can keep tool calls separately; count those too rather than omit their args.
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, list) and message.tool_calls:
            count += (len(json.dumps(message.tool_calls)) + 3) // 4
    return count


def _check_context(messages: list[AnyMessage], schema_tokens: int = 0) -> list[AnyMessage]:
    """Common admission guard for workers, supervisor and summarizer; preserve native messages."""
    prompt_tokens = approx_tokens(messages) + schema_tokens
    estimated = (
        prompt_tokens
        + settings.model_output_reserve_tokens
        + settings.context_safety_tokens
    )
    if estimated > settings.model_context_tokens:
        raise ContextBudgetExceeded(
            f"Estimated context {estimated} exceeds configured allowance {settings.model_context_tokens}; "
            "reduce context or configure the actual model's supported allowance.")
    set_current_prompt_tokens(prompt_tokens)
    return messages


def _supervisor_transcript(messages: list[AnyMessage], max_turns: int = 8) -> str:
    """Keep only dialogue turns (human inputs, human choices, and worker completions).

    Drops all internal tool-execution clutter.
    """
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"[user] {content_text(m.content).strip()}")
        elif isinstance(m, ToolMessage) and getattr(m, "name", None) == "ask_user":
            lines.append(f"[human choice] {content_text(m.content).strip()}")
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.content:
            lines.append(f"[{m.name or 'assistant'}] {content_text(m.content).strip()}")
    return "\n\n".join(lines[-max_turns:])


async def _load_memories(query: str = "") -> str:
    """Fetch this user's + this project's long-term memories, rendered for the prompt."""
    try:
        store = get_store()
    except RuntimeError:
        store = None
    if store is None:
        return ""
    return await recall_memories(store, query, match_required=False)


async def _reserve_model_call(state: AgentState, event: str) -> None:
    if state.get("run_context"):
        await reserve_run_usage(get_store(), state["run_context"], "model:" + event, models=1)


async def capture_memories_node(state: AgentState) -> dict:

    """Runtime-triggered capture; task execution never depends on an optional memory tool."""
    store = get_store()
    context = state["run_context"]
    namespace = ("memory_capture", *ns_for("user")[1:], namespace_label(context["workspace_id"]),
                 namespace_label(context["thread_id"]))
    base = state.get("history_offset", 0)
    cursor = max(0, state.get("memory_capture_upto", 0) - base)
    review = []
    for index, message in enumerate(state["messages"][cursor:], start=cursor):
        if not (isinstance(message, HumanMessage) and message.name != "supervisor"
                or isinstance(message, ToolMessage) and message.name == "ask_user"):
            continue
        source_id = message_reference(message)
        saved = await store.aget(namespace, source_id)
        if saved and saved.value.get("status") in ("completed", "needs_review"):
            continue
        try:
            text = content_text(message.content)
            if saved and saved.value.get("proposals") is not None:
                capture = MemoryCapture(proposals=saved.value["proposals"])
            else:
                recalled = await recall_memories(store, text)
                await _reserve_model_call(state, "capture:" + source_id)
                model = get_chat_model(temperature=0.0).with_structured_output(MemoryCapture)
                capture: MemoryCapture = await model.ainvoke(_check_context([
                    SystemMessage(content=MEMORY_CAPTURE_PROMPT),
                    HumanMessage(content=json.dumps({"user_message": text, "clarification_question": message.additional_kwargs.get("question"),
                                                     "current_memories": recalled}, ensure_ascii=False)),
                ], len(json.dumps(MemoryCapture.model_json_schema())) // 4))
            if any(p.quote not in text for p in capture.proposals):
                raise ValueError("Memory proposal is not an exact source quote")
            # Persist proposals before mutations: restart reuses the same extraction.
            record = {"status": "pending", "proposals": capture.model_dump()["proposals"], "results": []}
            await store.aput(namespace, source_id, record)
            received_at = message.additional_kwargs.get("received_at")
            for proposal in capture.proposals:
                if settings.memory_capture_mode == "review":
                    result = "REVIEW: automatic application is disabled; proposal retained."
                elif not received_at:
                    result = "REVIEW: historical source has no recorded receipt time."
                else:
                    event = hashlib.sha256(json.dumps([source_id, proposal.scope, proposal.quote]).encode()).hexdigest()
                    fact = proposal.quote
                    if message.additional_kwargs.get("question"):
                        fact = f"Question: {message.additional_kwargs['question']}\nUser answer: {proposal.quote}"
                    result = await apply_memory(
                        store, "update" if proposal.memory_id else "create", fact, proposal.scope,
                        proposal.memory_id, proposal.expected_revision,
                        source={"event_id": event, "message_id": source_id, "received_at": received_at})
                record["results"].append(result)
            record["status"] = "needs_review" if any(r.startswith(("ERROR", "CONFLICT", "REVIEW"))
                                                     for r in record["results"]) else "completed"
            await store.aput(namespace, source_id, record)
            if record["status"] == "needs_review":
                review.append(source_id)
        except (ContextBudgetExceeded, RunBudgetExceeded):
            raise
        except Exception as exc:
            log.warning("Memory capture failed for run %s (%s)", context["run_id"], type(exc).__name__)
            return {"memory_capture_upto": base + index, "memory_capture_status": {
                "status": "failed", "source_message": source_id, "reason": type(exc).__name__}}
    return {"memory_capture_upto": base + len(state["messages"]), "memory_capture_status": {
        "status": "needs_review" if review else "completed", "source_messages": review}}


_CODE_EXTENSIONS = (
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".java", ".rb", ".php", ".sh"
)

_TEST_CMD_PATTERNS = [
    re.compile(r"\bpytest\b", re.IGNORECASE),
    re.compile(r"\bunittest\b", re.IGNORECASE),
    re.compile(r"\bpython\d* -m unittest\b", re.IGNORECASE),
    re.compile(r"\bpython\d* -m pytest\b", re.IGNORECASE),
    re.compile(r"\b(?:python\d*|bash|sh|node)\s+\S*(?:test_|_test\.)\S*", re.IGNORECASE),
    re.compile(r"\b(?:cargo|npm|pnpm|yarn|go)\s+test\b", re.IGNORECASE),
    re.compile(r"\bvitest\b", re.IGNORECASE),
    re.compile(r"\bjest\b", re.IGNORECASE),
]


def _is_test_file(path: str) -> bool:
    p = path.lower().replace("\\", "/")
    parts = p.split("/")
    filename = parts[-1]
    return (
        filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.endswith(".test.js")
        or filename.endswith(".test.ts")
        or filename.endswith(".spec.js")
        or filename.endswith(".spec.ts")
        or "tests" in parts
        or "test" in parts
    )


def check_completion_verification(
    messages: list[AnyMessage],
    trajectory: list[dict[str, Any] | AnyMessage] | None = None,
    user_goal: str = "",
) -> tuple[bool, str | None]:
    """Inspect trajectory to ensure code changes have automated verification evidence.

    Returns (True, None) if completion is verified or no code was modified.
    Returns (False, reason) if code was modified without test execution, if tests failed,
    or if code was modified after the last test run.
    """
    all_items = list(trajectory or []) + list(messages)
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    for item in all_items:
        tcs = item.get("tool_calls", []) if isinstance(item, dict) else (getattr(item, "tool_calls", None) or [])
        for tc in tcs:
            cid = tc.get("id")
            if cid:
                tool_calls_by_id[cid] = tc

    code_files_modified: set[str] = set()
    last_code_mod_idx = -1
    last_test_run_idx = -1
    latest_test_exit_code = 0
    latest_test_output = ""

    for idx, item in enumerate(all_items):
        name = item.get("name", "") if isinstance(item, dict) else (getattr(item, "name", None) or "")
        content_str = str(item.get("content", "")) if isinstance(item, dict) else str(getattr(item, "content", ""))
        tool_call_id = item.get("tool_call_id", "") if isinstance(item, dict) else (getattr(item, "tool_call_id", None) or "")

        if name in ("fs_write", "fs_edit"):
            if not content_str.startswith("ERROR") and not content_str.startswith("CONFLICT"):
                tc = tool_calls_by_id.get(tool_call_id)
                path = tc.get("args", {}).get("path", "") if tc and isinstance(tc.get("args"), dict) else ""
                if not path:
                    parts = content_str.split()
                    if len(parts) >= 2:
                        path = parts[1]
                if any(path.endswith(ext) for ext in _CODE_EXTENSIONS):
                    code_files_modified.add(path)
                    last_code_mod_idx = idx

        elif name == "shell_exec":
            tc = tool_calls_by_id.get(tool_call_id)
            command = tc.get("args", {}).get("command", "") if tc and isinstance(tc.get("args"), dict) else ""
            is_test = False
            if command and any(pat.search(command) for pat in _TEST_CMD_PATTERNS):
                is_test = True
            elif any(term in content_str for term in ("pytest", "Ran 1 test", "Ran 2 tests", "Ran 3 tests", "collected", "test session starts")):
                is_test = True

            if is_test:
                exit_code = 0
                code_match = re.search(r"exit_code=(-?\d+)", content_str)
                if code_match:
                    exit_code = int(code_match.group(1))
                elif content_str.startswith("ERROR"):
                    exit_code = 1
                last_test_run_idx = idx
                latest_test_exit_code = exit_code
                latest_test_output = content_str

    test_files_modified = [f for f in code_files_modified if _is_test_file(f)]
    goal_requests_testing = bool(re.search(
        r"\b(test|tests|tested|testing|verify|verification|unittest|pytest)\b",
        user_goal,
        re.IGNORECASE,
    ))
    verification_enforced = bool(test_files_modified or goal_requests_testing or last_test_run_idx != -1)

    if code_files_modified and verification_enforced:
        if last_test_run_idx == -1:
            files_str = ", ".join(sorted(code_files_modified))
            return False, (
                f"SYSTEM GATE: Code modifications were made ({files_str}), but no automated test execution "
                "was observed in the run trajectory. Execute tests (e.g., pytest or unittest) to verify the code "
                "before completing the task."
            )
        # Pytest returns exit_code 5 when 0 items were collected (e.g. no tests in workspace)
        is_empty_suite = latest_test_exit_code == 5 and ("collected 0 items" in latest_test_output or "no tests ran" in latest_test_output)
        if latest_test_exit_code != 0 and not is_empty_suite:
            return False, (
                f"SYSTEM GATE: Automated tests failed (exit_code={latest_test_exit_code}). Inspect the failure and fix "
                "the issues before completing the task."
            )
        if last_code_mod_idx > last_test_run_idx:
            return False, (
                "SYSTEM GATE: Code was modified after the last test run. Re-run automated tests to verify the final "
                "state of your changes before completing the task."
            )

    return True, None


async def supervisor_node(state: AgentState) -> Command[Literal["planner", "coder", "__end__"]]:
    """Orchestrates Planner and Coder agents: triages user requests and evaluates worker progress."""
    memories = await _load_memories("\n".join([state.get("task_objective", ""), *state.get("task_requirements", [])]))
    msgs = state["messages"]
    iteration = state.get("supervisor_iteration", 0)
    max_iters = settings.max_supervisor_iterations

    # Check if this is a re-entry (worker executed and returned an AIMessage)
    is_reentry = bool(msgs and isinstance(msgs[-1], AIMessage) and iteration > 0)

    # Safety guard: stop runaway loops with explicit budget exhaustion outcome
    if is_reentry and iteration >= max_iters:
        last_ai = msgs[-1].content if msgs else "Task completed or reached iteration limit."
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=f"{content_text(last_ai)}\n\n[Supervisor: Maximum iterations reached. Stopped.]")],
                "supervisor_iteration": 0,
                "run_outcome": RunOutcome(
                    status="budget_exhausted",
                    reason=f"Maximum supervisor iterations ({max_iters}) reached without task completion.",
                ).model_dump(),
            },
        )

    # Determine authoritative user goal: prefer state.task_objective if set, fallback to user HumanMessage
    user_goal = state.get("task_objective")
    user_msg = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if not user_goal:
        user_goal = content_text(user_msg.content) if user_msg else ""
    if state.get("task_requirements"):
        user_goal += "\nADDITIONAL REQUIREMENTS / CORRECTIONS:\n" + "\n".join(
            f"- {req}" for req in state["task_requirements"]
        )
    summary_updates: dict[str, Any] = {}
    visible_messages = msgs
    if approx_tokens(msgs) > settings.compaction_token_threshold or state.get("segment"):
        older, _ = _split_for_compaction(msgs, settings.compaction_token_threshold // 2)
        if older:
            seg = await _compact_history(get_chat_model(temperature=0.0), state, older)
            summary_updates["segment"] = seg.model_dump()
            visible_messages = msgs[seg.covers_upto - state.get("history_offset", 0):]
    prior_msgs = [m for m in visible_messages if m is not user_msg]

    todos_text = ""
    if state.get("todos"):
        todos_text = "\nCURRENT PLAN (TODOS):\n" + "\n".join(
            f"- [{t.get('status', 'pending')}] {t.get('content', '')}" for t in state["todos"]
        )

    segment_text = ""
    current_segment = summary_updates.get("segment") or state.get("segment")
    if current_segment:
        seg = CompactedSegment(**current_segment)
        segment_text = f"COMPACTED CONTEXT (earlier turns):\n{seg.render()}\n\n"
    if state.get("run_context"):
        summary_updates["main_history_upto"] = await archive_main_history(get_store(), {**state, **summary_updates})
        if current_segment:
            base = state.get("history_offset", 0)
            captured = max(base, state.get("memory_capture_upto", 0))
            for message in msgs[captured - base:]:
                if isinstance(message, HumanMessage):
                    break
                captured += 1
            summary_updates["memory_capture_upto"] = captured
            cut = min(current_segment["covers_upto"], captured) - base
            if cut > 0:
                # Only evict the active checkpoint view AFTER immutable originals are saved.
                summary_updates["messages"] = [RemoveMessage(id=m.id) for m in msgs[:cut]]
                summary_updates["history_offset"] = base + cut

    available_list = ", ".join(f"'{k}'" for k in AGENT_SPECS.keys())
    if is_reentry:
        worker_last = content_text(msgs[-1].content)
        prompt_content = (
            f"ORIGINAL USER GOAL:\n{user_goal}\n\n"
            f"{todos_text}\n\n"
            f"KNOWN MEMORIES (untrusted facts):\n{memories or 'none'}\n\n"
            f"WORKER'S RECENT ACTIONS / OUTPUT:\n{worker_last}\n\n"
            f"BACKGROUND:\n{segment_text}{_supervisor_transcript(prior_msgs, max_turns=6)}\n\n"
            "EVALUATE: Has the worker completed the user's goal? "
            "If yes: route='end' and provide direct_response summarizing the outcome. "
            f"If no: route to one of ({available_list}) and provide clear worker_instruction on what to do next."
        )
    else:
        prompt_content = (
            f"SKILL CATALOGUE:\n{skill_manager.get_catalogue_prompt()}\n\n"
            f"KNOWN MEMORIES:\n{memories or 'none'}\n\n"
            f"BACKGROUND (earlier turns, context only):\n{segment_text}{_supervisor_transcript(prior_msgs, max_turns=8)}\n\n"
            f"CURRENT USER REQUEST (route and pick skills for THIS):\n{user_goal}"
        )

    await _reserve_model_call(state, f"supervisor:{iteration}")
    model = get_chat_model(temperature=0.0).with_structured_output(Triage)
    decision: Triage = await model.ainvoke(_check_context([
        SystemMessage(content=get_supervisor_prompt()),
        HumanMessage(content=prompt_content),
    ], len(json.dumps(Triage.model_json_schema())) // 4))

    # Routing is a model proposal; known unfinished work cannot become success.
    # Continue through the existing bounded loop instead of requiring another user turn.
    if decision.route == "end" and any(t.get("status") != "completed" for t in state.get("todos", [])):
        decision = Triage(route="coder", skills=state.get("active_skills", []),
                          worker_instruction="Continue the unfinished plan against the original objective. "
                          "Verify each result before marking it complete; report any concrete blocker.")

    # Evidence-backed completion verification gate (P1-05):
    # If code files were created or modified during the run, require automated test verification.
    if decision.route == "end":
        verified, reason = check_completion_verification(state.get("messages", []), state.get("trajectory", []), user_goal=user_goal)
        if not verified and reason:
            decision = Triage(
                route="coder",
                skills=state.get("active_skills", []),
                worker_instruction=reason,
            )

    if decision.route == "end":
        updates: dict[str, Any] = {
            **summary_updates,
            "supervisor_iteration": 0,
            "memories": memories,
            "run_outcome": RunOutcome(
                status="completed",
                reason=decision.direct_response or "Task completed.",
            ).model_dump(),
        }
        if decision.direct_response:
            updates["messages"] = [*summary_updates.get("messages", []),
                                   AIMessage(content=decision.direct_response, name="supervisor")]
        return Command(goto=END, update=updates)

    texts = []
    for name in decision.skills:
        body = skill_manager.load_skill_instructions(name)
        if body:
            texts.append(f"## Skill: {name}\n{body}")

    agent_updates: dict[str, Any] = {
        **summary_updates,
        "worker_id": uuid.uuid4().hex,
        "supervisor_iteration": iteration + 1,
        "active_skills": decision.skills,
        "skill_text": "\n\n".join(texts),
        "memories": memories,
        "run_outcome": RunOutcome(
            status="running",
            reason=f"Delegated to {decision.route}",
        ).model_dump(),
    }
    if decision.worker_instruction:
        # Keep worker_instruction as internal state; do not append as HumanMessage.
        agent_updates["worker_instruction"] = decision.worker_instruction

    return Command(goto=decision.route, update=agent_updates)


def _is_empty_tool_output(content: str) -> bool:
    c = content.strip()
    return c in (
        "exit_code=0\n--- stdout ---\n\n--- stderr ---",
        "exit_code=0\n--- stdout ---\n--- stderr ---",
        "exit_code=0\n--- stdout ---\n\n--- stderr ---\n",
        "Command completed with no output.",
        "{}",
        "[]",
        "",
    )


def _is_greeting(content: str) -> bool:
    c = content.strip().lower()
    return c in {
        "hi", "hello", "hey", "greetings", "hello!", "hi there", "hey there",
        "good morning", "good afternoon", "good evening", "howdy",
    }


def _snip_trivial_turns(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Deterministic Stage 2 compaction: prunes uninformative messages before LLM summarization.

    - Collapses empty tool execution outputs (exit code 0 with no stdout/stderr)
    - Collapses consecutive duplicate read observations (identical fs_read calls)
    - Prunes opening greeting exchanges when followed by substantive task turns
    """
    if not messages:
        return []

    # 1. Prune leading greeting exchanges if followed by substantive turns
    start_idx = 0
    if len(messages) >= 2:
        m0, m1 = messages[0], messages[1]
        if isinstance(m0, HumanMessage) and _is_greeting(content_text(m0.content)):
            if isinstance(m1, AIMessage) and not getattr(m1, "tool_calls", None):
                if len(messages) > 2:
                    start_idx = 2

    pruned: list[AnyMessage] = []
    for msg in messages[start_idx:]:
        # Collapse empty tool execution outputs
        if isinstance(msg, ToolMessage) and _is_empty_tool_output(str(msg.content)):
            pruned.append(ToolMessage(
                content="exit_code=0 (no output)",
                name=getattr(msg, "name", "tool"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
            ))
            continue

        # Collapse duplicate read observations (and the redundant tool invocation if isolated)
        if (
            isinstance(msg, ToolMessage)
            and getattr(msg, "name", None) in ("fs_read", "read_artifact")
        ):
            last_tool = next((m for m in reversed(pruned) if isinstance(m, ToolMessage)), None)
            if (
                last_tool is not None
                and getattr(last_tool, "name", None) == getattr(msg, "name", None)
                and str(last_tool.content) == str(msg.content)
            ):
                # If preceding message is an AIMessage with only this tool call, prune it too
                if pruned and isinstance(pruned[-1], AIMessage):
                    tcs = getattr(pruned[-1], "tool_calls", None) or []
                    if len(tcs) == 1 and tcs[0].get("id") == getattr(msg, "tool_call_id", None):
                        pruned.pop()
                continue

        pruned.append(msg)

    return pruned


def _split_for_compaction(messages: list[AnyMessage], keep_tokens: int) -> tuple[list, list]:
    """Compact complete exchanges, including exchanges within a single user turn.

    A parallel tool-call batch stays together until every response has arrived.
    Keep at least the last exchange, even if that exchange alone exceeds the target.
    """
    pending: set[str] = set()
    boundaries = []
    for i, message in enumerate(messages):
        if not pending and not isinstance(message, ToolMessage):
            boundaries.append(i)
        if isinstance(message, AIMessage):
            pending.update(call["id"] for call in message.tool_calls)
        elif isinstance(message, ToolMessage):
            pending.discard(message.tool_call_id)
    cut = 0
    suffix_tokens = [0] * (len(messages) + 1)
    for index in range(len(messages) - 1, -1, -1):
        suffix_tokens[index] = suffix_tokens[index + 1] + approx_tokens([messages[index]])
    for boundary in reversed(boundaries):
        if cut and suffix_tokens[boundary] > keep_tokens:
            break
        cut = boundary
        if suffix_tokens[boundary] >= keep_tokens:
            break
    return messages[:cut], messages[cut:]


def _format_message_for_compaction(m: AnyMessage) -> str:
    """Format a message for the compaction transcript, preserving tool calls and outputs."""
    if isinstance(m, HumanMessage):
        return f"[human] {content_text(m.content)}"
    if isinstance(m, AIMessage):
        parts = []
        txt = content_text(m.content).strip()
        if txt:
            parts.append(txt)
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            parts.append(f"call {name}({args})")
        return f"[ai] {' | '.join(parts)}" if parts else "[ai]"
    if isinstance(m, ToolMessage):
        name = getattr(m, "name", "tool")
        return f"[tool:{name}] {content_text(m.content)}"
    return f"[{m.type}] {content_text(m.content)}"


def _clip_overflow_tail(
    older: list[AnyMessage],
    state: AgentState,
    prev: CompactedSegment | None,
    base: int,
    exc: Exception,
    max_retained: int = 8,
) -> CompactedSegment:
    """Stage 5b deterministic tail retention fallback when LLM summarization fails.

    Extracts high-signal metadata (files modified, active todos, task objective)
    and retains recent turn context without calling an LLM.
    """
    retained = older[-max_retained:] if len(older) > max_retained else older
    detected_files = []
    for m in older:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                path = tc.get("args", {}).get("path")
                if path and path not in detected_files:
                    detected_files.append(path)

    active_tasks = [t["content"] for t in state.get("todos", []) if t.get("status") != "completed"]
    preserved = [f"Objective: {state.get('task_objective')}"] if state.get("task_objective") else []

    seg = CompactedSegment(
        version=(prev.version + 1) if prev else 1,
        covers_upto=base + len(older),
        summary_id="",
        parent_summary_id=(prev.summary_id or None) if prev else None,
        source_from=prev.source_from if prev and prev.source_from else message_reference(older[0]),
        source_to=message_reference(older[-1]),
        summary=f"[Compaction LLM unavailable ({type(exc).__name__}: {exc}) — deterministic preservation of {len(retained)} turns]",
        preserved_facts=preserved,
        file_changes=detected_files[-10:],
        open_tasks=active_tasks,
    )
    return seg


async def _compact_history(summarizer, state: AgentState, older: list[AnyMessage]) -> CompactedSegment:
    """One incremental, bounded summary pipeline for supervisor and worker context."""
    prev = CompactedSegment(**state["segment"]) if state.get("segment") else None
    base = state.get("history_offset", 0)
    if prev and base + len(older) <= prev.covers_upto:
        return prev
    structured = summarizer.with_structured_output(CompactedSegment)
    try:
        archive = get_store()
    except RuntimeError:
        archive = None
    if prev and not prev.summary_id and archive is not None and state.get("run_context"):
        await save_compaction(archive, state["run_context"], prev, state.get("history_scope"))
    cursor = max(0, prev.covers_upto - base) if prev else 0
    raw_slice = older[cursor:]
    snipped_slice = _snip_trivial_turns(raw_slice)
    summary_schema_tokens = len(json.dumps(CompactedSegment.model_json_schema())) // 4

    if not snipped_slice:
        if prev:
            prev.covers_upto = base + len(older)
            return prev
        return CompactedSegment(
            summary="All prior turns were trivial confirmations.",
            covers_upto=base + len(older),
            version=1,
            summary_id="",
            source_from=message_reference(older[0]),
            source_to=message_reference(older[-1]),
        )

    prior_txt = f"EXISTING COMPACTED SEGMENT (reconcile with new evidence):\n{prev.render()}\n\n" if prev else ""
    prefix = f"{prior_txt}OLDER MESSAGES TO COMPACT:\n"
    batch = []
    end = 0
    overhead = approx_tokens([SystemMessage(content=COMPACTION_PROMPT), HumanMessage(content=prefix)])
    remaining_chars = 4 * (settings.model_context_tokens - settings.model_output_reserve_tokens
                           - settings.context_safety_tokens - summary_schema_tokens - overhead - 2)
    while end < len(snipped_slice):
        line = _format_message_for_compaction(snipped_slice[end])
        if len(line) + 1 > remaining_chars:
            break
        batch.append(line)
        remaining_chars -= len(line) + 1
        end += 1
    if not batch:
        raise ContextBudgetExceeded("One source message exceeds the summarizer input allowance; read its evidence in chunks.")
    await _reserve_model_call(state, f"summary:{state.get('history_scope', 'main')}:{base + cursor}:{base + len(older)}")
    try:
        seg: CompactedSegment = await structured.ainvoke(_check_context([
            SystemMessage(content=COMPACTION_PROMPT),
            HumanMessage(content=prefix + "\n".join(batch)),
        ], summary_schema_tokens))
        # Coverage, lineage and identity come from runtime, never model output.
        seg.version = (prev.version + 1) if prev else 1
        seg.covers_upto = base + len(older)
        seg.summary_id = ""
        seg.parent_summary_id = (prev.summary_id or None) if prev else None
        seg.source_from = prev.source_from if prev and prev.source_from else message_reference(older[0])
        seg.source_to = message_reference(older[-1])
        if approx_tokens([SystemMessage(content=seg.render())]) > settings.summary_max_tokens:
            raise ContextBudgetExceeded("Compacted segment exceeds the configured summary allowance.")
    except Exception as exc:
        log.warning("Compaction LLM failed (%s: %s); falling back to Stage 5b tail-clipping", type(exc).__name__, exc)
        seg = _clip_overflow_tail(older, state, prev, base, exc)

    if archive is not None and state.get("run_context"):
        await save_compaction(archive, state["run_context"], seg, state.get("history_scope"))
    return seg


def _make_compaction_hook(summarizer, prompt_template: str = AGENT_PROMPT, schema_tokens: int = 0):

    async def hook(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        await _reserve_model_call(state, f"worker:{state.get('history_scope', 'main')}:{message_reference(messages[-1])}")
        base = prompt_template
        if state.get("task_objective"):
            base += "\n\n## Authoritative objective\n" + state["task_objective"]
        if state.get("task_requirements"):
            base += "\n\n## Accepted requirements\n" + "\n".join(state["task_requirements"])
        if state.get("todos"):
            base += "\n\n## Current plan\n" + "\n".join(
                f"- [{t['status']}] {t['content']}" for t in state["todos"])
        # Fetch on every reasoning step so a tool's correction replaces stale prompt facts.
        memories = await _load_memories("\n".join([state.get("task_objective", ""),
                                                  *state.get("task_requirements", []),
                                                  state.get("worker_instruction", "")]))
        if memories:
            base += "\n\n## Long-term memory (untrusted user & project facts)\n" + memories
        if state.get("skill_text"):
            base += "\n\n" + state["skill_text"]
        if state.get("worker_instruction"):
            base += f"\n\n## Current Supervisor Task\n{state['worker_instruction']}"
        system = SystemMessage(content=base)

        # Provider message ordering invariant:
        # Chat models (such as Gemini and Anthropic without prefilling) strictly require
        # the conversation history to end in a user message or tool response.
        # If the history ends with an AIMessage from a previous agent/turn, supply the
        # worker instruction as the prompt turn for this reasoning step without mutating state["messages"].
        working_messages = list(messages)
        if working_messages and isinstance(working_messages[-1], AIMessage):
            instruction = state.get("worker_instruction") or "Proceed with the next step according to the plan."
            working_messages.append(HumanMessage(content=instruction))

        if approx_tokens([system, *working_messages]) <= settings.compaction_token_threshold and not state.get("segment"):
            return {"llm_input_messages": _check_context([system, *working_messages], schema_tokens)}

        older, recent = _split_for_compaction(working_messages, settings.compaction_token_threshold // 2)
        if not older:
            return {"llm_input_messages": _check_context([system, *working_messages], schema_tokens)}

        seg = await _compact_history(summarizer, state, older)
        updates = {"segment": seg.model_dump()}

        # Provider-facing view only: never insert internal instructions into raw history.
        recent = working_messages[seg.covers_upto - state.get("history_offset", 0):]
        if recent and not isinstance(recent[0], HumanMessage):
            recent = [HumanMessage(content="Continue the authoritative objective using the recorded context."), *recent]
        context = SystemMessage(content=seg.render())
        if approx_tokens([context]) > settings.summary_max_tokens:
            raise ContextBudgetExceeded("Compacted segment exceeds the configured summary allowance.")
        return {"llm_input_messages": _check_context([system, context, *recent], schema_tokens), **updates}

    return hook


def render_and_save_graph_image(
    compiled_graph: Any,
    output_path: str | Path = "agent_graph.png",
    xray: bool = True,
) -> Path | None:
    """Save the native compiled LangGraph graph visualization using its own automatic generator."""
    try:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        compiled_graph.get_graph(xray=xray).draw_mermaid_png(output_file_path=str(target))
        log.info("Saved compiled agent architecture diagram to %s", target)
        return target
    except Exception as exc:
        log.warning("Could not render agent graph image: %s", exc)
        return None


def _make_worker_node(worker, worker_name: str, checkpointer):
    """Map private worker state; capture the subgraph so native introspection can discover it."""
    async def run_worker(state: AgentState):
        worker_id = state["worker_id"]
        inputs = {key: state[key] for key in (
            "task_objective", "task_requirements", "run_context", "todos", "active_skills",
            "skill_text", "worker_instruction") if key in state}
        inputs.update(history_scope=worker_id, messages=[HumanMessage(
            content=state["worker_instruction"], name="supervisor")])
        if state.get("segment"):
            inputs["skill_text"] = (inputs.get("skill_text", "") + "\n\nParent context (historical evidence):\n"
                                    + CompactedSegment(**state["segment"]).render())
        if state.get("worker_result"):
            inputs["skill_text"] = inputs.get("skill_text", "") + "\n\nPrevious worker handoff:\n" + json.dumps(state["worker_result"])
        try:
            result = await worker.ainvoke(inputs)
        except GraphBubbleUp:
            raise  # Native interrupt/resume is not a failed worker result.
        except (Exception, asyncio.CancelledError) as exc:
            snapshot = await worker.aget_state(get_config()) if checkpointer else None
            await save_worker_result(get_store(), state["run_context"], worker_id, worker_name,
                                     snapshot.values if snapshot else inputs,
                                     f"Worker stopped ({type(exc).__name__}).",
                                     "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed")
            raise
        last = next((m for m in reversed(result.get("messages", [])) if isinstance(m, AIMessage)), None)
        response = content_text(last.content) if last else "Worker returned without a final response."
        handoff = await save_worker_result(get_store(), state["run_context"], worker_id, worker_name, result, response)
        capture_updates = {}
        if any(isinstance(m, ToolMessage) and m.name == "ask_user" for m in result.get("messages", [])):
            captured = await capture_memories_node({**result, "memory_capture_upto": 0})
            capture_updates["memory_capture_status"] = captured["memory_capture_status"]

        worker_tool_msgs = [
            m for m in result.get("messages", [])
            if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and bool(getattr(m, "tool_calls", None)))
        ]
        new_trajectory = list(state.get("trajectory", [])) + [
            m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in worker_tool_msgs
        ]

        return {**capture_updates, "worker_result": handoff, "todos": result.get("todos", state.get("todos", [])),
                "task_requirements": result.get("task_requirements", state.get("task_requirements", [])),
                "trajectory": new_trajectory,
                "messages": [AIMessage(content=json.dumps(handoff), name=worker_name)]}

    return run_worker


async def build_graph(
    checkpointer: Any | None = None,
    store: Any | None = None,
    include_mcp: bool = True,
    save_graph_image: bool = False,
    image_path: str | Path = "agent_graph.png",
    session_permissions: set[str] | None = None,
    allow_interactive_approval: bool = True,
):
    """Assemble and compile the multi-agent graph dynamically from AGENT_SPECS."""
    import general_coding_agent.prompts as prompt_mod

    # Even an ephemeral agent needs consistent thread state across calls. The CLI
    # supplies SQLite; standalone use retains checkpoints for this graph's lifetime.
    checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
    model = get_chat_model()
    summarizer = get_chat_model(temperature=0.0)

    builder = StateGraph(AgentState)
    builder.add_node("capture_memories", capture_memories_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_edge(START, "capture_memories")
    builder.add_edge("capture_memories", "supervisor")

    for name, spec in AGENT_SPECS.items():
        tools = await spec.tools_builder(include_mcp=include_mcp) if spec.requires_mcp else await spec.tools_builder()
        # Tier 1 Schema-Level Tool Stripping (P2-06):
        # Dynamically strip unauthorized tools so unpermitted schemas are completely omitted from the model payload
        tools = filter_tools_for_session(
            tools,
            session_permissions=session_permissions,
            allow_interactive_approval=allow_interactive_approval,
        )
        prompt_text = getattr(prompt_mod, spec.prompt_name, AGENT_PROMPT)
        from langchain_core.utils.function_calling import convert_to_openai_tool
        schema_tokens = len(json.dumps([convert_to_openai_tool(tool) for tool in tools])) // 4
        worker_node = create_react_agent(
            model,
            ToolNode(tools, awrap_tool_call=durable_tool_wrapper, handle_tool_errors=False),
            state_schema=AgentState,
            pre_model_hook=_make_compaction_hook(summarizer, prompt_template=prompt_text, schema_tokens=schema_tokens),
            store=store,
        )
        builder.add_node(name, _make_worker_node(worker_node, name, checkpointer))
        builder.add_edge(name, "supervisor")

    # Durable restart requires the CLI's SQLite checkpointer and store together.
    compiled = builder.compile(checkpointer=checkpointer, store=store if store is not None else InMemoryStore())

    if save_graph_image:
        render_and_save_graph_image(compiled, output_path=image_path)

    return compiled


def prepare_run_input(
    user_input: str | Command,
    thread_id: str,
    tenant_id: str = "default_tenant",
    user_id: str | None = None,
    workspace_id: str | None = None,
    existing_state: dict[str, Any] | None = None,
    new_task: bool = False,
) -> tuple[dict[str, Any] | Command, RunContext]:
    """Prepare graph input, establish stable objective, and create/preserve RunContext.

    Shared entry path for a new turn, continuation, or Command(resume=...).
    - A new turn gets fresh turn_id and run_id.
    - Resuming an existing run retains turn_id and run_id.
    - Completed tasks are replaced by the next request; new_task explicitly replaces other tasks.
    - Follow-ups on unfinished tasks preserve the objective and append requirements.
    """
    effective_user_id = user_id or settings.user_id
    effective_workspace_id = workspace_id or str(settings.workspace_root)
    existing = existing_state or {}
    existing_ctx = existing.get("run_context") or {}

    if isinstance(user_input, Command):
        if new_task or not existing_ctx or user_input.resume is None:
            raise ValueError("Resume requires an existing interrupted run and an answer.")
        if user_input.update or user_input.goto or user_input.graph:
            raise ValueError("Only Command(resume=...) is accepted at the run boundary.")
        if isinstance(user_input.resume, str):
            clean_resume = user_input.resume.strip()
            if len(clean_resume) > settings.input_max_chars:
                raise ValueError(f"Request exceeds {settings.input_max_chars} characters")
        return user_input, RunContext(**existing_ctx)

    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("A nonblank text request is required.")

    clean_input = user_input.strip()
    if len(clean_input) > settings.input_max_chars:
        raise ValueError(f"Request exceeds {settings.input_max_chars} characters")

    # New turn
    turn_id = uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    run_ctx = RunContext(
        tenant_id=tenant_id,
        user_id=effective_user_id,
        workspace_id=effective_workspace_id,
        thread_id=thread_id,
        turn_id=turn_id,
        run_id=run_id,
    )

    graph_input: dict[str, Any] = {
        "messages": [HumanMessage(content=clean_input, id=turn_id,
                                  additional_kwargs={"received_at": datetime.now(timezone.utc).isoformat()})],
        "run_context": run_ctx.model_dump(),
        "run_outcome": RunOutcome(status="running").model_dump(),
        "supervisor_iteration": 0,
        "worker_instruction": "",
    }

    # Handle authoritative objective vs follow-up requirements
    active_objective = existing.get("task_objective")
    if new_task or not active_objective or existing.get("run_outcome", {}).get("status") == "completed":
        graph_input.update(task_objective=clean_input, task_requirements=[], todos=[],
                           active_skills=[], skill_text="", worker_result={}, worker_id="")
    else:
        existing_reqs = list(existing.get("task_requirements") or [])
        existing_reqs.append(clean_input)
        graph_input["task_requirements"] = existing_reqs

    return graph_input, run_ctx


async def get_run_state(graph, thread_id: str) -> dict[str, Any]:
    """Read checkpoint state plus authoritative lifecycle metadata, without changing resume tasks.

    Raw graph.aget_state() contains the last node's outcome; use this reader for run status.
    Durable restart requires both the checkpointer and store supplied by the CLI.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config) if graph.checkpointer else None
    state = dict(snap.values or {}) if snap else {}
    if snap:
        state["__interrupt__"] = tuple(i for task in snap.tasks for i in task.interrupts)
    record = await graph.store.aget(("run_lifecycle",), thread_id) if graph.store else None
    if record:
        # A record written before the first checkpoint of a new run cannot resume an old run.
        if state.get("run_context", {}).get("run_id") != record.value["run_context"]["run_id"]:
            state.pop("__interrupt__", None)
        state.update(record.value)
    return state


async def _run(
    graph: Any, user_input: str | Command, thread_id: str, tenant_id: str,
    user_id: str | None, workspace_id: str | None, new_task: bool = False,
) -> AsyncIterator[tuple[str, Any]]:
    """Single lifecycle owner for both invocation APIs; never edits an interrupted checkpoint."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.graph_step_limit}
    state = await get_run_state(graph, thread_id)
    previous_ctx = state.get("run_context")
    if previous_ctx and (
        previous_ctx["tenant_id"] != tenant_id
        or previous_ctx["user_id"] != (user_id or settings.user_id)
        or previous_ctx["workspace_id"] != (workspace_id or str(settings.workspace_root))
    ):
        raise ValueError("Thread identity/workspace does not match this run.")
    if isinstance(user_input, Command):
        if not graph.checkpointer or not state.get("__interrupt__"):
            raise ValueError("There is no saved interrupt to resume.")
    elif state.get("__interrupt__"):
        raise ValueError("Answer the pending interrupt before starting another turn; use a new thread for other work.")
    graph_input, ctx = prepare_run_input(user_input, thread_id, tenant_id, user_id, workspace_id,
                                        state, new_task=new_task)
    config["configurable"]["run_context"] = ctx.model_dump()
    state.pop("__interrupt__", None)
    if isinstance(graph_input, dict):
        state.update({k: v for k, v in graph_input.items() if k != "messages"})

    async def record(status: RunStatusLiteral, reason: str | None = None) -> dict[str, Any]:
        outcome = RunOutcome(status=status, reason=reason).model_dump()
        state["run_outcome"] = outcome
        if graph.store:
            await graph.store.aput(("run_lifecycle",), thread_id, {
                "run_context": ctx.model_dump(), "run_outcome": outcome,
                "task_objective": state.get("task_objective", ""),
                "task_requirements": state.get("task_requirements", []),
            })
        return outcome

    await record("running")
    paused = False
    finished = False
    started = time.perf_counter()
    deadline = None
    try:
        usage = await reserve_run_usage(graph.store, ctx.model_dump(), "start") if graph.store else {}
        remaining = settings.run_timeout_seconds - usage.get("active_seconds", 0)
        if remaining <= 0:
            raise RunBudgetExceeded("Active run time limit reached.")
        deadline = asyncio.timeout(remaining)
        # values makes arun and streaming share identical execution and outcome handling.
        astream_kwargs = {"durability": "sync"} if graph.checkpointer else {}
        async with deadline, aclosing(graph.astream(graph_input, config=config,
                                          stream_mode=["messages", "updates", "values"],
                                          **astream_kwargs)) as events:
            async for mode, chunk in events:
                if mode == "values":
                    state.update(chunk)
                elif mode == "updates" and "__interrupt__" in chunk:
                    if not graph.checkpointer:
                        raise RuntimeError("Human interaction requires a checkpointer; restart with persistence enabled.")
                    interrupts = chunk["__interrupt__"]
                    state["__interrupt__"] = interrupts
                    approval = any(isinstance(i.value, dict) and i.value.get("kind") == "approval"
                                   for i in interrupts)
                    outcome = await record("waiting_for_approval" if approval else "waiting_for_input",
                                           "Awaiting a human response.")
                    paused = True
                    yield "outcome", outcome
                yield mode, chunk
        if not paused:
            outcome = state.get("run_outcome", {})
            status = outcome.get("status", "running")
            if status == "running":
                status, reason = "blocked", "Graph stopped without a completion decision."
            else:
                reason = outcome.get("reason")
            outcome = await record(status, reason)
            finished = True
            yield "outcome", outcome
        else:
            # A final values chunk may still contain the last node's pre-interrupt status.
            state["run_outcome"] = outcome
        finished = True
        yield "state", state
    except (GraphRecursionError, ContextBudgetExceeded, RunBudgetExceeded, OperationBlocked) as exc:
        if isinstance(exc, OperationBlocked):
            status = "blocked"
            reason = str(exc)
        else:
            status = "budget_exhausted"
            reason = "Graph step limit reached." if isinstance(exc, GraphRecursionError) else str(exc)
        outcome = await record(status, reason)
        finished = True
        yield "outcome", outcome
        yield "state", state
    except TimeoutError:
        if deadline is None or not deadline.expired():
            await record("failed", "Execution failed (TimeoutError).")
            raise
        outcome = await record("budget_exhausted", "Active run time limit reached.")
        finished = True
        yield "outcome", outcome
        yield "state", state
    except (asyncio.CancelledError, GeneratorExit):
        if not paused and not finished:
            await record("cancelled", "Run execution was cancelled.")
        raise
    except Exception as exc:
        # Keep provider payloads/secrets out of durable status. Preserve the original exception.
        log.error("Run %s failed (%s)", ctx.run_id, type(exc).__name__)
        await record("failed", f"Execution failed ({type(exc).__name__}).")
        raise
    finally:
        if graph.store:
            state["run_usage"] = await reserve_run_usage(
                graph.store, ctx.model_dump(), "active:" + uuid.uuid4().hex,
                active_seconds=time.perf_counter() - started)


async def arun(
    graph,
    user_input: str | Command,
    thread_id: str,
    tenant_id: str = "default_tenant",
    user_id: str | None = None,
    workspace_id: str | None = None,
    new_task: bool = False,
) -> dict:
    """Run one turn to completion; return the final state. For tests/scripts."""
    result = {}
    async with aclosing(_run(graph, user_input, thread_id, tenant_id, user_id, workspace_id, new_task)) as events:
        async for mode, chunk in events:
            if mode == "state":
                result = chunk
    return result


async def stream_events(
    graph,
    user_input: str | Command,
    thread_id: str,
    tenant_id: str = "default_tenant",
    user_id: str | None = None,
    workspace_id: str | None = None,
    new_task: bool = False,
) -> AsyncIterator[tuple[str, Any]]:
    """Yield (channel, payload): ('token', str) | ('tool', name) | ('interrupt', question) | ('outcome', dict).

    `user_input` is a str for a new turn, or a Command(resume=...) to answer an ask_user pause.
    """
    async with aclosing(_run(graph, user_input, thread_id, tenant_id, user_id, workspace_id, new_task)) as events:
        async for mode, chunk in events:
            if mode == "outcome":
                yield "outcome", chunk
            elif mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") not in AGENT_SPECS:
                    continue
                if isinstance(msg, ToolMessage):
                    yield "tool", msg.name
                elif isinstance(msg, AIMessage):
                    text = content_text(msg.content)
                    if text:
                        yield "token", text
            elif mode == "updates":
                capture_status = next((update["memory_capture_status"] for update in chunk.values()
                                       if isinstance(update, dict) and "memory_capture_status" in update), None)
                if capture_status and capture_status["status"] != "completed":
                    yield "memory", capture_status
                if "__interrupt__" in chunk:
                    value = chunk["__interrupt__"][0].value
                    yield "interrupt", value if isinstance(value, dict) else {"question": str(value), "options": []}
                    return
                for m in (chunk.get("supervisor") or {}).get("messages", []):
                    txt = content_text(m.content)
                    if txt:
                        yield "token", txt
