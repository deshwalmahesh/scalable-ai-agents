"""MCP tool loading via the standard langchain-mcp-adapters client.

One free, no-auth MCP server (time) is wired by default to demonstrate the integration.
Add more servers by extending DEFAULT_MCP_SERVERS. Failure to launch a server is
non-fatal — the agent keeps its local tools.
"""

import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)


class ParentDeathSupervisor:
    """Supervises long-running child subprocesses (e.g. MCP servers) via OS pipe death detection.

    The parent retains the write end of the pipe. If the parent process crashes or is killed
    (e.g., via SIGKILL), the OS kernel automatically closes the write end, causing the read end
    in the child / watcher to immediately receive EOF (0 bytes).
    """

    def __init__(self) -> None:
        self.r_fd, self.w_fd = os.pipe()
        os.set_inheritable(self.w_fd, False)
        os.set_inheritable(self.r_fd, True)
        self._closed = False

    def get_child_fd(self) -> int:
        """File descriptor to pass to child process for inheritance."""
        return self.r_fd

    def start_watcher_thread(self, target_pid: int, poll_interval: float = 0.2) -> threading.Thread:
        """Start a daemon thread that monitors the pipe and terminates target_pid on parent death."""
        def _watcher():
            try:
                with os.fdopen(self.r_fd, "rb", closefd=True) as stream:
                    stream.read(1)  # Blocks until write end is closed (parent death)
            except Exception:
                pass
            # Parent is dead: reap child process
            try:
                os.kill(target_pid, signal.SIGTERM)
                time.sleep(poll_interval)
                os.kill(target_pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        t = threading.Thread(target=_watcher, daemon=True, name=f"parent-death-watcher-{target_pid}")
        t.start()
        return t

    def close(self) -> None:
        """Close parent descriptors."""
        if not self._closed:
            self._closed = True
            try:
                os.close(self.w_fd)
            except OSError:
                pass
            try:
                os.close(self.r_fd)
            except OSError:
                pass


def watch_parent_pipe(read_fd: int, on_death: Callable[[], None] | None = None) -> None:
    """Helper for child processes: blocks on read_fd until parent terminates, then executes on_death."""
    try:
        with os.fdopen(read_fd, "rb", closefd=True) as stream:
            stream.read(1)
    except Exception:
        pass
    if on_death:
        on_death()
    else:
        os._exit(0)

# Each entry is a server the adapter connects to over stdio. `uvx` fetches+runs the
# server on demand, so nothing needs pre-installing.
DEFAULT_MCP_SERVERS: dict[str, dict] = {
    "time": {
        "command": "uvx",
        "args": ["mcp-server-time", "--local-timezone", "UTC"],
        "transport": "stdio",
    },
}


async def get_mcp_tools(servers: dict[str, dict] | None = None) -> list[BaseTool]:
    """Connect to the configured MCP servers and return their tools as LangChain tools."""
    servers = DEFAULT_MCP_SERVERS if servers is None else servers
    if not servers:
        return []
    try:
        return await MultiServerMCPClient(servers).get_tools()
    except Exception as exc:  # noqa: BLE001 - MCP launch is best-effort; degrade gracefully
        log.warning("MCP tools unavailable (%s); continuing with local tools only.", exc)
        return []
