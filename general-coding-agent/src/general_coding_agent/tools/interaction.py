"""Human-in-the-loop: let the agent ask the user a question mid-run.

Uses LangGraph's interrupt(): the graph pauses, its state is checkpointed, and the run
resumes with the user's answer via Command(resume=...). This is the IDEA doc's
"Executing -> Paused: user interrupts -> resume" edge — a real durable pause, not a guess.
"""

from datetime import datetime, timezone
from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt, Command
from general_coding_agent.state import AgentState


@tool
def ask_user(question: str, state: Annotated[AgentState, InjectedState],
             tool_call_id: Annotated[str, InjectedToolCallId], options: list[str] | None = None) -> Command:
    """Ask the user a clarifying question when the request is ambiguous or missing a decision
    you should not make alone (which framework, how many pages, styling, scope, destructive
    actions). Optionally pass `options` (a few suggested choices) — the user may pick one or
    reply freely. Returns the user's answer. Prefer ONE focused question over many.

    Interrupts the run (durable, checkpointed) and resumes with the user's answer."""
    if not question.strip():
        raise ValueError("Clarification question must be nonblank")
    answer = interrupt({"kind": "question", "question": question, "options": options or []})
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Clarification answer must be nonblank text")
    requirement = f"Clarification: {question}\nUser answer: {answer}"
    requirements = list(state.get("task_requirements", []))
    if requirement not in requirements:
        requirements.append(requirement)
    return Command(update={"task_requirements": requirements, "messages": [ToolMessage(
        content=answer, tool_call_id=tool_call_id, name="ask_user",
        additional_kwargs={"question": question, "received_at": datetime.now(timezone.utc).isoformat()})]})
