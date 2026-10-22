"""Generic artifact offloading for tool results + the read-back tool.

`wrap_with_artifacts` offloads oversized text and JSON observations and returns a bounded
preview plus reference. Small native values and control results (e.g. Command) pass through.
This is the "share directly if small, else offload and let the agent read it back" rule, applied
to every content-producing tool — our fs/CLI tools and MCP tools alike.
"""

import contextvars
import json
from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool, tool
from general_coding_agent.config import settings
from general_coding_agent.memory.artifacts import read_artifact_slice, save_artifact

_current_prompt_tokens: contextvars.ContextVar[int] = contextvars.ContextVar("current_prompt_tokens", default=0)


def set_current_prompt_tokens(tokens: int) -> None:
    """Record current prompt token count for dynamic headroom calculations."""
    _current_prompt_tokens.set(tokens)


def get_current_prompt_tokens() -> int:
    """Get current prompt token count for dynamic headroom calculations."""
    return _current_prompt_tokens.get()


def calculate_spill_threshold(
    current_prompt_tokens: int = 0,
    max_context: int | None = None,
    output_reserve: int | None = None,
    floor: int = 1000,
    ceiling: int = 50_000,
) -> int:
    """Calculate dynamic artifact spill threshold based on remaining prompt token headroom.

    Headroom = ContextLimit - PromptTokens - OutputReserve
    Spills tool results exceeding 25% of available headroom (in chars, approx 4 chars per token).
    Scales threshold between safety floor (1,000 chars default) and ceiling (50,000 chars default).
    Under ample context, threshold scales from 2,000 chars up to 50,000 chars.
    Under nearly-exhausted context, threshold safely drops to 1,000 chars to avoid ContextBudgetExceeded.
    """
    if max_context is None:
        max_context = settings.model_context_tokens
    if output_reserve is None:
        output_reserve = settings.model_output_reserve_tokens + settings.context_safety_tokens

    headroom = max(0, max_context - current_prompt_tokens - output_reserve)
    dynamic_limit = int(headroom * 4 * 0.25)
    return max(floor, min(ceiling, dynamic_limit))


from dataclasses import dataclass


@dataclass(frozen=True)
class SpillRef:
    """Structured reference to an offloaded tool artifact."""
    artifact_id: str
    tool_name: str
    total_lines: int
    total_bytes: int
    total_chars: int
    preview: str

    def format_message(self, prefix: str = "") -> str:
        return (
            f"{prefix}[{self.tool_name} produced {self.total_chars} chars "
            f"({self.total_lines} lines, {self.total_bytes} bytes) — offloaded to artifact {self.artifact_id}.\n"
            f"Preview (head + tail):\n{self.preview}\n...\n"
            f"Call read_artifact('{self.artifact_id}') for full content or intermediate slices.]"
        )


def make_head_tail_preview(text: str, head_lines: int = 15, tail_lines: int = 25) -> str:
    """Extract bounded head + tail preview from text.

    Preserves initial command setup/context (head) and terminal error traceback/summary (tail),
    annotating the omitted line count in the middle.
    """
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return text
    head = "\n".join(lines[:head_lines])
    tail = "\n".join(lines[-tail_lines:])
    omitted = len(lines) - head_lines - tail_lines
    return f"{head}\n\n... [{omitted} lines spilled to artifact] ...\n\n{tail}"


@tool
async def read_artifact(artifact_id: str, offset: int = 0, limit: int = 4000) -> str:
    """Read a slice of a large tool-output artifact that was offloaded from the context.
    Use the artifact id from a '[stored as artifact ...]' message. Page with offset/limit."""
    return read_artifact_slice(artifact_id, offset, limit)


def wrap_with_artifacts(
    t: BaseTool,
    current_tokens: int | Callable[[], int] | None = None,
    max_context: int | None = None,
) -> BaseTool:
    """Wrap content tools with offloading for large text or JSON observations.

    Dynamically scales the offloading threshold based on available token headroom
    in the active conversation context.
    """

    async def _run(**kwargs) -> object:
        result = await t.ainvoke(kwargs)
        # MCP returns JSON content blocks as well as strings. Keep small native values
        # intact; serialize large observations without changing control Commands.
        content = json.dumps(result, ensure_ascii=False) if isinstance(result, (list, dict)) else result

        # Determine prompt tokens to calculate dynamic headroom
        if callable(current_tokens):
            tokens = current_tokens()
        elif isinstance(current_tokens, int):
            tokens = current_tokens
        else:
            tokens = get_current_prompt_tokens()

        threshold = calculate_spill_threshold(tokens, max_context=max_context)

        if isinstance(content, str) and len(content) > threshold:
            meta = save_artifact(content, t.name)
            preview = make_head_tail_preview(content, head_lines=15, tail_lines=25)
            lines_count = len(content.splitlines())
            spill = SpillRef(
                artifact_id=meta["id"],
                tool_name=t.name,
                total_lines=lines_count,
                total_bytes=meta["bytes"],
                total_chars=meta["chars"],
                preview=preview,
            )
            prefix = ""
            if "exit_code=" in content:
                for line in content.splitlines()[:5]:
                    if line.startswith("exit_code="):
                        prefix = f"{line}\n"
                        break
            return spill.format_message(prefix=prefix)
        return result

    return StructuredTool.from_function(
        coroutine=_run, name=t.name, description=t.description, args_schema=t.args_schema,
    )
