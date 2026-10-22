"""The planning tool — this is the agent's "Planner".

Instead of a separate planner node emitting a rigid DAG, the agent writes/updates a todo
list as a tool call (the pattern LangChain's Deep Agents uses). The list lives in graph
state, so it survives across steps and is visible to the UI.
"""

from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from general_coding_agent.state import Todo


@tool
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Record or replace the current plan as a list of todos, each with a `content` string
    and a `status` of pending/in_progress/completed. Call this to plan before acting, and
    again to update statuses as you make progress."""
    done = sum(1 for t in todos if t["status"] == "completed")
    return Command(update={
        "todos": todos,
        "messages": [ToolMessage(f"Plan updated: {done}/{len(todos)} done.", tool_call_id=tool_call_id)],
    })
