"""Interactive CLI for the scalable agent.

Streams the agent's work (tokens, tool calls) and persists each thread via the SQLite
checkpointer. Slash commands: /skills /plan /new /exit
"""

import asyncio
import logging
import sys
import uuid
from contextlib import aclosing

from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel

from general_coding_agent.config import settings
from general_coding_agent.graph import build_graph, get_run_state, render_and_save_graph_image, stream_events
from general_coding_agent.memory.checkpointer import get_checkpointer
from general_coding_agent.memory.store import get_memory_store, project_ns, user_ns
from general_coding_agent.skills.manager import skill_manager

console = Console()
# Silence the noisy (harmless) google-genai AFC advisory log.
logging.getLogger("google_genai").setLevel(logging.ERROR)


def answer_interrupt(payload: dict) -> str:
    """Render the same saved interaction for a live pause or a reopened thread."""
    console.print(f"\nAgent asks: {payload['question']}", markup=False)
    opts = payload.get("options") or []
    for i, opt in enumerate(opts, 1):
        console.print(f"  {i}. {opt}", markup=False)
    raw = console.input("[bold green]You ❯[/bold green] ").strip()
    return opts[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(opts) else raw


async def show_memories(store, offset: int = 0) -> None:
    for label, ns in (("User", user_ns()), ("Project", project_ns())):
        items = await store.asearch(ns, limit=50, offset=offset)
        console.print(f"[bold]{label} memories (offset {offset}, {len(items)} records):[/bold]")
        for it in items:
            console.print(f"  {it.key} revision={it.value.get('revision', 0)} "
                          f"retired={it.value.get('retired', False)} expires={it.value.get('expires_at') or 'never'}: "
                          f"{it.value.get('content', '')}", markup=False)
        if len(items) == 50:
            console.print(f"More records may exist: /memories {offset + 50}", markup=False)


async def show_segment(graph, thread_id: str) -> None:
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    seg = (state.values or {}).get("segment") if state else None
    if not seg:
        console.print("[dim]No compaction segment yet.[/dim]")
        return
    console.print(f"[bold]Compacted segment v{seg['version']}[/bold] (covers {seg['covers_upto']} msgs)")
    console.print(f"[dim]{seg['summary'][:400]}[/dim]")
    for k in ("preserved_facts", "open_tasks", "file_changes", "unresolved_questions"):
        if seg.get(k):
            console.print(f"  [cyan]{k}[/cyan]: {seg[k]}")


async def show_plan(graph, thread_id: str) -> None:
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    todos = (state.values or {}).get("todos") if state else None
    if not todos:
        console.print("[dim]No active plan.[/dim]")
        return
    marks = {"completed": "[green]✓[/green]", "in_progress": "[yellow]⟳[/yellow]", "pending": "[dim]○[/dim]"}
    console.print("[bold]Plan:[/bold]")
    for t in todos:
        console.print(f"  {marks.get(t['status'], '○')} {t['content']}")


async def main() -> None:
    thread_id = sys.argv[1].strip() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else uuid.uuid4().hex[:8]
    console.print(Panel.fit(
        f"[bold cyan]Scalable Agent[/bold cyan]\n"
        f"[dim]provider={settings.llm_provider} model={settings.llm_model} workspace={settings.workspace_root}[/dim]\n"
        f"[dim]thread_id={thread_id}[/dim]\n"
        f"[dim]commands: /skills /plan /memories /segment /graph /new /task <request> /resume <id> /exit[/dim]",
        border_style="cyan",
    ))

    async with get_checkpointer() as checkpointer, get_memory_store() as store:
        graph = await build_graph(checkpointer=checkpointer, store=store)

        while True:
            new_task = False
            try:
                user_input = console.input("\n[bold green]You ❯[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]bye[/dim]")
                break
            if not user_input:
                continue
            if user_input == "/exit":
                break
            if user_input == "/new":
                thread_id = uuid.uuid4().hex[:8]
                console.print(f"[green]new thread:[/green] {thread_id}")
                continue
            if user_input.startswith("/resume"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    thread_id = parts[1].strip()
                    state = await get_run_state(graph, thread_id)
                    msg_count = len(state.get("messages", []))
                    console.print(f"[green]resumed thread:[/green] {thread_id} ({msg_count} msgs in history)")
                    outcome = state.get("run_outcome", {})
                    console.print(f"Run status: {outcome.get('status', 'no saved run')}. {outcome.get('reason') or ''}", markup=False)
                    if state.get("__interrupt__"):
                        try:
                            user_input = Command(resume=answer_interrupt(state["__interrupt__"][0].value))
                        except (KeyboardInterrupt, EOFError):
                            console.print("\n[dim]Thread remains paused.[/dim]")
                            continue
                    else:
                        continue
                else:
                    console.print("[yellow]usage: /resume <thread_id>[/yellow]")
                    continue
            if user_input == "/task":
                console.print("[yellow]usage: /task <request>[/yellow]")
                continue
            if isinstance(user_input, str) and user_input.startswith("/task "):
                new_task = True
                user_input = user_input.split(maxsplit=1)[1]
            if user_input == "/skills":
                for s in skill_manager.list_skills():
                    console.print(f"  • [cyan]{s.name}[/cyan]: {s.description}")
                continue
            if user_input == "/plan":
                await show_plan(graph, thread_id)
                continue
            if isinstance(user_input, str) and user_input.split(maxsplit=1)[0] == "/memories":
                parts = user_input.split()
                if len(parts) > 2 or (len(parts) == 2 and not parts[1].isdigit()):
                    console.print("Usage: /memories [nonnegative offset]", markup=False)
                    continue
                await show_memories(store, int(parts[1]) if len(parts) == 2 else 0)
                continue
            if user_input == "/segment":
                await show_segment(graph, thread_id)
                continue
            if user_input == "/graph":
                out = render_and_save_graph_image(graph, "agent_graph.png")
                if out and out.exists():
                    console.print(f"[green]Saved detailed architecture diagram to:[/green] {out}")
                else:
                    console.print("[yellow]Could not render graph image.[/yellow]")
                continue

            try:
                pending = user_input  # str for a new turn; becomes Command(resume=...) after an ask_user
                while True:
                    interrupted = False
                    async with aclosing(stream_events(graph, pending, thread_id, new_task=new_task)) as events:
                        async for channel, payload in events:
                            if channel == "token":
                                console.print(payload, end="", highlight=False, markup=False)
                            elif channel == "tool":
                                console.print(f"\nTool: {payload}", markup=False)
                            elif channel == "memory":
                                console.print(f"\nMemory capture {payload['status']}; current task state remains saved.", markup=False)
                            elif channel == "interrupt":
                                pending = Command(resume=answer_interrupt(payload))
                                interrupted = True
                                break
                            elif channel == "outcome":
                                status = payload.get("status")
                                if status in ("budget_exhausted", "failed", "blocked", "cancelled"):
                                    console.print(f"\nRun {status}: {payload.get('reason', '')}", markup=False)
                    new_task = False
                    if not interrupted:
                        break
                console.print()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Thread remains available through /resume.[/dim]")
            except asyncio.CancelledError:
                console.print("\n[dim]Run cancelled.[/dim]")
                raise
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive on a bad turn
                console.print(f"\n[red]error:[/red] {exc}")


if __name__ == "__main__":
    asyncio.run(main())
