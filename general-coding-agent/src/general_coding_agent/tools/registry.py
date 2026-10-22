"""Assemble the tool mix the agent runs with.

One place to add/remove/swap tools. Everything is a LangChain BaseTool, so create_react_agent's
ToolNode executes all of them uniformly. Content-producing tools (fs/CLI/MCP) are wrapped so any
oversized result is offloaded to an artifact; control tools (planner, memory, artifact reader)
return small/structured values and are used as-is.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from langchain_core.tools import BaseTool
from general_coding_agent.state import ToolEffect, ToolPolicy
from general_coding_agent.tools.artifacts import read_artifact, wrap_with_artifacts
from general_coding_agent.tools.fs import fs_edit, fs_read, fs_write, shell_exec
from general_coding_agent.tools.interaction import ask_user
from general_coding_agent.tools.mcp import get_mcp_tools
from general_coding_agent.tools.memory import manage_memory, search_memory, read_history
from general_coding_agent.tools.planning import write_todos

TOOL_POLICIES: dict[str, ToolPolicy] = {
    "fs_read": ToolPolicy(ToolEffect.READ),
    "read_artifact": ToolPolicy(ToolEffect.READ),
    "search_memory": ToolPolicy(ToolEffect.READ),
    "fs_write": ToolPolicy(ToolEffect.WORKSPACE_WRITE),
    "fs_edit": ToolPolicy(ToolEffect.WORKSPACE_WRITE),
    "read_history": ToolPolicy(ToolEffect.READ),
    "manage_memory": ToolPolicy(ToolEffect.MEMORY_WRITE),
    "write_todos": ToolPolicy(ToolEffect.STATE_CONTROL),
    "ask_user": ToolPolicy(ToolEffect.STATE_CONTROL),
    "shell_exec": ToolPolicy(ToolEffect.HOST_EXEC, approval_required=True),
}


def get_tool_policy(tool_name: str) -> ToolPolicy | None:
    """Return static capability/effect policy for tool_name. MCP tools default to EXTERNAL."""
    if tool_name in TOOL_POLICIES:
        return TOOL_POLICIES[tool_name]
    if tool_name.startswith("mcp_") or tool_name.startswith("time_"):
        return ToolPolicy(ToolEffect.EXTERNAL)
    return None


_EFFECT_ALIASES = {
    "READ": "READ",
    "WRITE": "WORKSPACE_WRITE",
    "WORKSPACE_WRITE": "WORKSPACE_WRITE",
    "EXEC": "HOST_EXEC",
    "HOST_EXEC": "HOST_EXEC",
    "STATE": "STATE_CONTROL",
    "STATE_CONTROL": "STATE_CONTROL",
    "EXTERNAL": "EXTERNAL",
    "MEMORY": "MEMORY_WRITE",
    "MEMORY_WRITE": "MEMORY_WRITE",
    "APPROVE": "APPROVE",
}


def filter_tools_for_session(
    tools: list[BaseTool],
    session_permissions: set[str] | None = None,
    allow_interactive_approval: bool = True,
) -> list[BaseTool]:
    """Dynamically filter tools before schema generation and agent binding.

    Completely strips unauthorized, unpermitted, or non-interactive tools from
    the model's request schema.

    Args:
        tools: The source list of BaseTool instances to filter.
        session_permissions: Optional set of allowed permissions or effects
            (e.g., {'READ', 'WORKSPACE_WRITE', 'HOST_EXEC', 'STATE_CONTROL', 'EXTERNAL'}
            or {'CAN_READ', 'CAN_WRITE', 'CAN_EXEC'}). If None, all permissions are granted.
        allow_interactive_approval: If False, omits tools requiring human approval
            (e.g. for headless or automated batch execution).
    """
    normalized_permissions = None
    if session_permissions is not None:
        normalized_permissions = set()
        for p in session_permissions:
            clean = p.upper().removeprefix("CAN_")
            normalized_permissions.add(_EFFECT_ALIASES.get(clean, clean))

    filtered: list[BaseTool] = []
    for t in tools:
        policy = get_tool_policy(t.name)
        if policy is None:
            policy = ToolPolicy(ToolEffect.EXTERNAL)

        # Check approval requirement in automated/non-interactive mode
        if policy.approval_required and not allow_interactive_approval:
            if normalized_permissions is None or "APPROVE" not in normalized_permissions:
                continue

        # Check explicit effect permission
        if normalized_permissions is not None:
            effect_name = policy.effect.value.upper()
            if effect_name not in normalized_permissions:
                continue

        filtered.append(t)

    return filtered


# fs_read is self-limiting (pages with offset/limit) -> must NOT be offloaded, or reading a big
# file would loop: dump -> offload -> read again -> dump -> offload ...
PAGED_TOOLS: list[BaseTool] = [fs_read]
# Tools whose result is a byproduct and can be large -> wrapped with artifact offloading.
CONTENT_TOOLS: list[BaseTool] = [fs_write, fs_edit, shell_exec]
# Small/structured/interrupting results -> used directly. Workers are read-only on memory.
CONTROL_TOOLS: list[BaseTool] = [write_todos, search_memory, read_history, read_artifact, ask_user]


async def build_tools(include_mcp: bool = True) -> list[BaseTool]:
    """Return the full tool list: paged fs_read + offload-wrapped content tools (+ MCP) + control tools."""
    tools = list(PAGED_TOOLS) + [wrap_with_artifacts(t) for t in CONTENT_TOOLS]
    if include_mcp:
        tools += [wrap_with_artifacts(t) for t in await get_mcp_tools()]
    tools += CONTROL_TOOLS
    return tools


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    prompt_name: str
    tools_builder: Callable[..., Awaitable[list[BaseTool]]]
    requires_mcp: bool = False


async def build_planner_tools() -> list[BaseTool]:
    """Read-only tools for planning, exploration, and artifact creation. Zero file mutation tools."""
    return [fs_read, write_todos, search_memory, read_history, read_artifact, ask_user]


async def build_coder_tools(include_mcp: bool = True) -> list[BaseTool]:
    """Full execution and mutation tools for coding and verification."""
    return await build_tools(include_mcp=include_mcp)


AGENT_SPECS: dict[str, AgentSpec] = {
    "planner": AgentSpec(
        name="planner",
        description="Read-only analysis, repo inspection, human clarification (`ask_user`), and planning. Uses `write_todos` to record actionable plans. CANNOT modify source code files. Call when the task is new, complex, ambiguous, or needs investigation/clarification.",
        prompt_name="PLANNER_PROMPT",
        tools_builder=build_planner_tools,
        requires_mcp=False,
    ),
    "coder": AgentSpec(
        name="coder",
        description="Implementation and verification. Has file mutation tools (`fs_edit`, `fs_write`) and execution (`shell_exec`). Call when there is a concrete plan or specific code to write/fix/test.",
        prompt_name="AGENT_PROMPT",
        tools_builder=build_coder_tools,
        requires_mcp=True,
    ),
}


def get_agent_catalogue_prompt() -> str:
    """Dynamically generate the Available Workers documentation for the supervisor prompt."""
    lines = ["Available Workers:"]
    for name, spec in AGENT_SPECS.items():
        lines.append(f"- \"{name}\": {spec.description}")
    lines.append("- \"end\": All tasks are completed, or for direct chit-chat/greetings (e.g. 'hi'). NEVER use 'end' for technical, coding, or ambiguous project requests.")
    return "\n".join(lines)
