"""Graph state + the structured compaction artifact.

Extends the prebuilt create_react_agent state (messages + remaining_steps) with the
channels this agent needs. TypedDict because that is what create_react_agent/ToolNode expect.

Compaction is non-destructive (IDEA_SCALABLE_AGENT.md §"Compaction must be non-destructive"):
originals are archived before the supervisor evicts an active checkpoint prefix. Workers use
a compacted model view. Each history has its own versioned summary and source boundaries.
"""

from enum import StrEnum
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict
from pydantic import BaseModel, Field, ConfigDict
from langgraph.prebuilt.chat_agent_executor import AgentState as _ReactState


class ToolEffect(StrEnum):
    READ = "read"
    WORKSPACE_WRITE = "workspace_write"
    MEMORY_WRITE = "memory_write"
    HOST_EXEC = "host_exec"
    EXTERNAL = "external"
    STATE_CONTROL = "state_control"


@dataclass(frozen=True)
class ToolPolicy:
    effect: ToolEffect
    approval_required: bool = False


class ToolObservation(BaseModel):
    ok: bool
    operation_id: str
    content: str = ""
    exit_code: int | None = None
    artifact_id: str | None = None
    bytes: int = 0
    truncated: bool = False
    error_type: str | None = None


class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]


class MemoryProposal(BaseModel):
    """A source quote, never a model-invented fact; runtime validates its provenance."""
    model_config = ConfigDict(extra="forbid")
    quote: str = Field(min_length=1, max_length=2000)
    scope: Literal["user", "project"]
    memory_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=0)


class MemoryCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposals: list[MemoryProposal] = Field(default_factory=list, max_length=5)


class CompactedSegment(BaseModel):
    """A versioned, structured summary of older messages. The IDEA doc's compaction artifact.

    Prose alone silently drops things; these explicit lists are what a coding agent must not
    lose across compaction cycles (requirements, open work, file paths, unresolved errors)."""

    version: int = Field(default=1, description="Bumped every time the segment is (re)generated")
    covers_upto: int = Field(default=0, description="Number of leading messages this segment summarises")
    summary_id: str = ""
    parent_summary_id: str | None = None
    source_from: str = ""
    source_to: str = ""
    summary: str = Field(default="", description="Dense narrative of the compacted turns")
    preserved_facts: list[str] = Field(default_factory=list, description="User requirements, decisions, constraints")
    open_tasks: list[str] = Field(default_factory=list, description="TODOs / work still pending")
    file_changes: list[str] = Field(default_factory=list, description="Files created/edited (with paths)")
    unresolved_questions: list[str] = Field(default_factory=list, description="Open errors / test failures / questions")

    def render(self) -> str:
        parts = [f"[Compacted context v{self.version}, covering the first {self.covers_upto} messages]", self.summary]
        if self.summary_id:
            parts.append(f"History reference: {self.summary_id}; previous: {self.parent_summary_id or 'none'}. "
                         "Use read_history for original messages or previous summaries when evidence is unclear.")
        for label, items in (
            ("Preserved facts / requirements", self.preserved_facts),
            ("Open tasks", self.open_tasks),
            ("File changes", self.file_changes),
            ("Unresolved questions / errors", self.unresolved_questions),
        ):
            if items:
                parts.append(f"{label}:\n" + "\n".join(f"- {i}" for i in items))
        return "\n\n".join(p for p in parts if p)


RunStatusLiteral = Literal[
    "running",
    "waiting_for_input",
    "waiting_for_approval",
    "completed",
    "failed",
    "cancelled",
    "budget_exhausted",
    "blocked",
]


class RunContext(BaseModel):
    """Execution context and identity metadata for a specific agent run."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    tenant_id: str = Field(default="default_tenant", min_length=1, description="Trusted tenant identity")
    user_id: str = Field(default="local_user", min_length=1, description="Authorized user identifier")
    workspace_id: str = Field(default="default_workspace", min_length=1, description="Target workspace identifier")
    thread_id: str = Field(min_length=1, description="Conversation thread identifier")
    turn_id: str = Field(min_length=1, description="Single user turn identifier")
    run_id: str = Field(min_length=1, description="Unique agent execution run identifier")


class RunOutcome(BaseModel):
    """Explicit terminal or lifecycle outcome of an agent run."""

    status: RunStatusLiteral = Field(default="running", description="Current or terminal lifecycle status")
    reason: str | None = Field(default=None, description="Explanation or evidence for the outcome")


class AgentState(_ReactState):
    """messages + remaining_steps are inherited from _ReactState."""

    task_objective: NotRequired[str]        # Authoritative user objective established at entry
    task_requirements: NotRequired[list[str]]  # Accepted follow-up requirements/corrections
    run_context: NotRequired[dict]          # RunContext.model_dump(); per-run identity metadata
    run_outcome: NotRequired[dict]          # Node outcome; get_run_state overlays durable lifecycle status
    todos: NotRequired[list[Todo]]
    active_skills: NotRequired[list[str]]
    skill_text: NotRequired[str]
    segment: NotRequired[dict]              # CompactedSegment.model_dump(); the compaction artifact
    memories: NotRequired[str]              # rendered long-term memories injected this turn
    supervisor_iteration: NotRequired[int]  # tracks supervisor <-> worker loop iterations
    worker_instruction: NotRequired[str]    # current targeted instruction from supervisor
    worker_id: NotRequired[str]             # Stable delegation ID, preserved when a worker resumes
    history_scope: NotRequired[str]         # Worker invocation; absent for main conversation
    worker_result: NotRequired[dict]        # Runtime-persisted handoff, not verified completion
    main_history_upto: NotRequired[int]     # Number of original parent messages archived
    history_offset: NotRequired[int]        # Global index of messages[0] after archived-prefix eviction
    memory_capture_upto: NotRequired[int]   # Examined parent-message cursor, independent of compaction
    memory_capture_status: NotRequired[dict]
    trajectory: NotRequired[list[dict[str, Any]]] # Cumulative record of worker tool interactions for evidence verification

