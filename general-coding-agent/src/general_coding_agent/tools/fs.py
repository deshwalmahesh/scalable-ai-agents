"""Filesystem + shell tools, as native LangChain tools.

These are "our" robust CLI tools: every path is confined to workspace_root, writes are
atomic and hash-guarded (optimistic concurrency), outputs are byte-bounded, and shell
runs are time-bounded. They are plain @tool functions, so create_react_agent / ToolNode
execute them directly — no custom broker needed.

TODO WARNING NOTE:
In production environments, arbitrary shell execution and file mutations MUST be run
inside an isolated Docker container, microVM (e.g., Firecracker / gVisor), or a sandboxed
runtime. Executing arbitrary commands directly on the host machine poses critical risks of
destructive deletion, resource exhaustion, and security boundary breakout.
"""

import asyncio
import difflib
import hashlib
import os
import re
import shlex
import shutil
import signal
import sys
import tempfile
from itertools import islice
from pathlib import Path
from langchain_core.tools import tool
from general_coding_agent.config import settings

_DESTRUCTIVE_COMMANDS = {"rm", "rmdir", "sudo", "mkfs", "dd", "shutdown", "reboot", "kill", "killall", "truncate"}
_SYSTEM_ALLOWED_ROOTS = {"/bin", "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/sbin", "/usr/sbin", "/dev"}
_DANGEROUS_EXPANSION_RE = re.compile(r"(\$(?:\\.|\\\n)*[({]|`|<\(.*?\)|>\(.*?\))")
_FORBIDDEN_MUTATIONS = {"export", "unset", "set", "declare", "typeset", "readonly", "local", "alias", "unalias", "source"}


def _ask_human_or_deny(question: str, options: list[str] | None = None, *, ticket: str | None = None) -> bool:
    """Pause via LangGraph interrupt() for human approval; if outside graph or rejected, return False."""
    try:
        from langgraph.types import interrupt
        payload: dict[str, Any] = {"kind": "approval", "question": question, "options": options or ["yes", "no"]}
        if ticket:
            payload["ticket"] = ticket
        res = interrupt(payload)

        # If resumed with a dict, validate cryptographic ticket binding if present
        if isinstance(res, dict):
            if ticket and res.get("ticket") and res["ticket"] != ticket:
                return False  # Ticket mismatch
            approved = res.get("approved", res.get("reply", False))
            return bool(approved is True or str(approved).strip().lower() in ("yes", "y", "approve", "allow"))

        return str(res).strip().lower() in ("yes", "y", "approve", "allow")
    except (RuntimeError, KeyError):
        return False


def _resolve_safe_path(path: str | Path, op: str) -> tuple[Path | None, str | None]:
    raw = Path(path)
    root = settings.workspace_root.resolve()
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not resolved.is_relative_to(root):
        ticket = hashlib.sha256(f"{op}:{resolved}".encode("utf-8")).hexdigest()
        if not _ask_human_or_deny(
            f"Security Alert: Agent attempting {op} outside workspace on '{resolved}'. Allow? [yes/no]",
            options=["yes", "no"],
            ticket=ticket,
        ):
            return None, f"PERMISSION DENIED: {op} on '{path}' outside workspace root was rejected."
    return resolved, None


def _check_command_safety(cmd: str, root: Path) -> tuple[bool, str]:
    root = root.resolve()

    # 1. Pre-flight AST/syntax check: block command substitutions, process substitutions, and backticks
    if _DANGEROUS_EXPANSION_RE.search(cmd):
        return False, "Command substitution, process substitution, or parameter expansion is forbidden in pre-flight checks"

    # 2. Block background job detachment
    if cmd.strip().endswith("&"):
        return False, "Background process execution (&) is forbidden"

    # 3. Robust tokenizer (preserves comment stripping and quotes)
    try:
        tokens = shlex.split(cmd, comments=True)
    except ValueError as e:
        return False, f"Malformed shell syntax: {e}"
    except Exception:
        tokens = cmd.split()

    for i, token in enumerate(tokens):
        if token == "&":
            return False, "Background process execution (&) is forbidden"
        if token in _FORBIDDEN_MUTATIONS:
            return False, f"Variable or shell environment mutation is forbidden: '{token}'"

        # Check file redirection destinations
        if token in (">", ">>", "1>", "2>", "&>"):
            if i + 1 < len(tokens):
                target = tokens[i + 1].strip("><|;&()")
                if target.startswith("~"):
                    return False, f"Redirect target references home directory '{target}' outside workspace"
                if ".." in target:
                    try:
                        resolved = (root / target).resolve()
                        if not resolved.is_relative_to(root):
                            return False, f"Redirect target '{target}' escapes workspace root"
                    except Exception:
                        pass
                if target.startswith("/"):
                    try:
                        resolved = Path(target).resolve()
                        if not resolved.is_relative_to(root):
                            return False, f"Redirect target '{target}' is outside workspace root"
                    except Exception:
                        pass

        clean = token.strip("><|;&()")
        if not clean:
            continue
        cmd_name = Path(clean).name
        if cmd_name in _DESTRUCTIVE_COMMANDS:
            return False, f"Destructive command detected: '{cmd_name}'"
        if clean.startswith("~"):
            return False, f"References home directory '{clean}' outside workspace"
        if ".." in clean:
            try:
                resolved = (root / clean).resolve()
                if not resolved.is_relative_to(root):
                    return False, f"Relative path '{clean}' escapes workspace root"
            except Exception:
                pass
        if clean.startswith("/"):
            try:
                resolved = Path(clean).resolve()
                if not resolved.is_relative_to(root):
                    is_sys = any(str(resolved).startswith(r) for r in _SYSTEM_ALLOWED_ROOTS)
                    is_py = str(resolved).startswith(sys.prefix) or str(resolved).startswith(sys.base_prefix)
                    if not (is_sys or is_py):
                        return False, f"Absolute path '{clean}' is outside workspace root"
            except Exception:
                pass
    return True, "Safe"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _read_text(target: Path) -> str:
    """Bound allocation before decoding; never rewrite invalid bytes as replacement text."""
    with target.open("rb") as stream:
        data = stream.read(settings.file_max_bytes + 1)
    if len(data) > settings.file_max_bytes:
        raise ValueError(f"File exceeds {settings.file_max_bytes} bytes; use a bounded command to inspect it.")
    if b"\x00" in data:
        raise ValueError("Binary file is unsupported by text tools.")
    return data.decode("utf-8")


def _write_text(target: Path, content: str) -> None:
    """Publish a unique sibling temporary file, preserving existing permission bits."""
    data = content.encode("utf-8")
    if len(data) > settings.file_max_bytes:
        raise ValueError(f"Content exceeds {settings.file_max_bytes} bytes.")
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o600
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.chmod(tmp, mode)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)


@tool
async def fs_read(path: str, offset: int = 1, limit: int | None = None) -> str:
    """Read a text file or list a directory, PAGED. Returns up to `limit` lines starting at 1-indexed `offset`
    (default: first page). Big files are never dumped whole — the header shows total lines;
    if there are more, read the next page with a higher offset. Long lines are truncated."""
    if offset < 1 or (limit is not None and limit < 1):
        return "ERROR: offset and limit must be positive."
    target, err = _resolve_safe_path(path, op="read")
    if err or not target:
        return f"ERROR: {err}"
    if not target.exists():
        return f"ERROR: file not found: {path}"
    limit = min(limit or settings.read_page_lines, settings.read_page_lines)
    if target.is_dir():
        # Directory order is filesystem order; avoid materializing/sorting a huge tree.
        with os.scandir(target) as entries:
            page = list(islice(entries, offset - 1, offset + limit))
        names = [entry.name + ("/" if entry.is_dir(follow_symlinks=False) else "") for entry in page[:limit]]
        more = f"\n[more entries: offset={offset + limit}]" if len(page) > limit else ""
        return f"# Directory {path}\n" + "\n".join(names) + more

    try:
        text = _read_text(target)
    except ValueError as exc:
        return f"ERROR: {exc}"
    lines = text.splitlines()
    total = len(lines)
    start = max(1, offset)
    end = min(total, start + limit - 1)

    cap = settings.max_line_chars
    body = "\n".join(
        f"{i}: {lines[i - 1][:cap]}{' …[line truncated]' if len(lines[i - 1]) > cap else ''}"
        for i in range(start, end + 1)
    )
    more = f"\n[lines {start}-{end} of {total}; read the next page with offset={end + 1}]" if end < total else ""
    return f"# {path} (hash={_hash(text)}, total_lines={total})\n{body}{more}"


@tool
async def fs_write(path: str, content: str, expected_hash: str | None = None) -> str:
    """Create a file or fully overwrite one, atomically. Overwriting requires
    expected_hash (from fs_read); the write is rejected if the file changed meanwhile."""
    target, err = _resolve_safe_path(path, op="write")
    if err or not target:
        return f"ERROR: {err}"
    if target.is_dir():
        return f"ERROR: is a directory: {path}"
    if target.exists():
        try:
            current = _read_text(target)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if expected_hash != _hash(current):
            return f"CONFLICT: {path} changed since read (current hash={_hash(current)}); re-read first."
        created = False
    else:
        if expected_hash is not None:
            return f"CONFLICT: {path} no longer exists; re-read before creating it."
        target.parent.mkdir(parents=True, exist_ok=True)
        created = True

    try:
        _write_text(target, content)
    except ValueError as exc:
        return f"ERROR: {exc}"
    return f"{'created' if created else 'overwrote'} {path} (hash={_hash(content)}, bytes={len(content.encode())})"


def _fuzzy_find_and_replace(current: str, target_text: str, replacement: str) -> tuple[str | None, str | None]:
    """Attempt exact match first, then line-level fuzzy matching for whitespace and indentation shifts.

    Returns (new_content, None) on success, or (None, error_message) on failure.
    """
    has_crlf = "\r\n" in current
    norm_current = current.replace("\r\n", "\n")
    norm_target = target_text.replace("\r\n", "\n")
    norm_replacement = replacement.replace("\r\n", "\n")

    count = norm_current.count(norm_target)
    if count == 1:
        res = norm_current.replace(norm_target, norm_replacement, 1)
        if has_crlf:
            res = res.replace("\r\n", "\n").replace("\n", "\r\n")
        return res, None
    if count > 1:
        return None, f"ERROR: target_text matched {count} times in file; add surrounding lines to make it unique."

    current_lines = norm_current.splitlines(keepends=True)
    target_lines = norm_target.splitlines()
    if not target_lines:
        return None, "ERROR: target_text contains no lines."

    n = len(target_lines)

    # Strategy 1: Match with rstrip() (ignores trailing whitespace differences)
    matches = []
    target_rstripped = [line.rstrip() for line in target_lines]
    for i in range(len(current_lines) - n + 1):
        window = [current_lines[i + j].rstrip() for j in range(n)]
        if window == target_rstripped:
            matches.append(i)

    if len(matches) == 1:
        idx = matches[0]
        repl_lines = norm_replacement.splitlines(keepends=True)
        if repl_lines and not repl_lines[-1].endswith("\n") and current_lines[idx + n - 1].endswith("\n"):
            repl_lines[-1] += "\n"
        new_lines = current_lines[:idx] + repl_lines + current_lines[idx + n:]
        res = "".join(new_lines)
        if has_crlf:
            res = res.replace("\r\n", "\n").replace("\n", "\r\n")
        return res, None

    if len(matches) > 1:
        return None, f"ERROR: target_text matched {len(matches)} times (ignoring trailing whitespace); add surrounding lines to make it unique."

    # Strategy 2: Match with strip() and relative indentation shift
    def get_indent(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    non_empty_target = [l for l in target_lines if l.strip()]
    if not non_empty_target:
        return None, "ERROR: target_text contains only whitespace."

    base_target_indent = get_indent(non_empty_target[0])
    target_rel_indents = [(get_indent(l) - base_target_indent) if l.strip() else None for l in target_lines]
    target_stripped = [l.strip() for l in target_lines]

    shift_matches = []
    for i in range(len(current_lines) - n + 1):
        window_raw = current_lines[i:i + n]
        window_stripped = [l.strip() for l in window_raw]
        if window_stripped != target_stripped:
            continue
        first_non_empty = next((l for l in window_raw if l.strip()), None)
        if not first_non_empty:
            continue
        base_file_indent = get_indent(first_non_empty)
        file_rel_indents = [(get_indent(l) - base_file_indent) if l.strip() else None for l in window_raw]
        if file_rel_indents == target_rel_indents:
            shift = base_file_indent - base_target_indent
            shift_matches.append((i, shift))

    if len(shift_matches) == 1:
        idx, shift = shift_matches[0]
        repl_lines = norm_replacement.splitlines(keepends=True)
        adjusted_repl = []
        for r in repl_lines:
            if r.strip():
                if shift > 0:
                    adjusted_repl.append(" " * shift + r)
                elif shift < 0:
                    to_remove = min(abs(shift), get_indent(r))
                    adjusted_repl.append(r[to_remove:])
                else:
                    adjusted_repl.append(r)
            else:
                adjusted_repl.append(r)
        if adjusted_repl and not adjusted_repl[-1].endswith("\n") and current_lines[idx + n - 1].endswith("\n"):
            adjusted_repl[-1] += "\n"
        new_lines = current_lines[:idx] + adjusted_repl + current_lines[idx + n:]
        res = "".join(new_lines)
        if has_crlf:
            res = res.replace("\r\n", "\n").replace("\n", "\r\n")
        return res, None

    if len(shift_matches) > 1:
        return None, f"ERROR: target_text matched {len(shift_matches)} times (with indentation shift); add surrounding lines to make it unique."

    return None, "ERROR: target_text not found in file (even with whitespace and indentation tolerance). Match whitespace/indentation exactly or re-read first."


@tool
async def fs_edit(path: str, target_text: str, replacement: str, expected_hash: str | None = None) -> str:
    """Replace an exact or whitespace-tolerant unique snippet in a file and return the unified diff.
    target_text must match uniquely (include surrounding lines to disambiguate).
    Tolerates trailing whitespace and uniform indentation shifts automatically.
    Pass expected_hash from fs_read for optimistic-concurrency protection."""
    if not target_text:
        return "ERROR: target_text must be nonempty."
    target, err = _resolve_safe_path(path, op="edit")
    if err or not target:
        return f"ERROR: {err}"
    if not target.exists():
        return f"ERROR: file not found: {path}"
    if target.is_dir():
        return f"ERROR: is a directory: {path}"

    try:
        current = _read_text(target)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if expected_hash != _hash(current):
        return f"CONFLICT: {path} changed since read (current hash={_hash(current)}); re-read first."

    new, error_msg = _fuzzy_find_and_replace(current, target_text, replacement)
    if error_msg or new is None:
        return error_msg or f"ERROR: target_text not found in {path}."

    try:
        _write_text(target, new)
    except ValueError as exc:
        return f"ERROR: {exc}"
    diff = "".join(difflib.unified_diff(
        current.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=2,
    ))
    return f"edited {path} (hash={_hash(new)})\n{diff}"


def _build_sandbox_command(
    command: str,
    workspace_root: Path,
    platform: str = sys.platform,
    sandbox_bin: str | None = None,
    bwrap_bin: str | None = None,
) -> tuple[str, list[str]]:
    """Determine sandboxed runner executable and argument list for current OS.

    - macOS: Seatbelt sandbox-exec with compiled Scheme profile.
    - Linux: Bubblewrap (bwrap) with read-only root mount and workspace bind.
    - Fallback: Direct shell execution.
    """
    root = str(workspace_root.resolve())
    if platform == "darwin" and sandbox_bin:
        home = str(Path.home().resolve())
        tmp = str(Path(tempfile.gettempdir()).resolve())
        profile = (
            f"(version 1)\n(allow default)\n"
            f'(deny file-write* (subpath "{home}"))\n'
            f'(allow file-write* (subpath "{root}"))\n'
            f'(allow file-write* (subpath "{tmp}"))\n'
            f'(allow file-write* (subpath "/private/tmp"))\n'
            f'(deny file-read* (subpath "{home}/.ssh"))\n'
            f'(deny file-read* (subpath "{home}/.aws"))\n'
            f'(deny file-read* (subpath "{home}/.gnupg"))\n'
        )
        return sandbox_bin, ["-p", profile, "/bin/sh", "-c", command]
    elif platform.startswith("linux") and bwrap_bin:
        args = [
            "--ro-bind", "/", "/",
            "--bind", root, root,
            "--tmpfs", "/tmp",
            "--proc", "/proc",
            "--dev", "/dev",
            "--unshare-all",
            "--die-with-parent",
            "/bin/sh", "-c", command,
        ]
        return bwrap_bin, args
    return "/bin/sh", ["-c", command]


@tool
async def shell_exec(command: str, timeout_seconds: int | None = None) -> str:
    """Run a shell command in the workspace with a timeout and bounded output.
    Returns exit code plus stdout/stderr."""
    # TODO WARNING NOTE: In production, arbitrary shell execution MUST be isolated in an ephemeral
    # Docker container or microVM. Running arbitrary agent commands directly on the host machine
    # risks host breakout, credential leakage, and destructive deletion.
    if not command.strip() or (timeout_seconds is not None and timeout_seconds < 1):
        return "ERROR: command must be nonblank and timeout_seconds must be positive."
    safe, reason = _check_command_safety(command, settings.workspace_root)
    if not safe:
        ticket = hashlib.sha256(f"shell_exec:{command}".encode("utf-8")).hexdigest()
        approved = _ask_human_or_deny(
            f"Security Alert: Shell command is destructive or references paths outside workspace ({reason}): '{command}'. Allow execution? [yes/no]",
            options=["yes", "no"],
            ticket=ticket,
        )
        if not approved:
            return f"COMMAND REJECTED BY USER: Execution was denied. Reason: {reason}"

    timeout = min(timeout_seconds or settings.command_timeout_seconds, settings.command_timeout_seconds)
    executable, args = _build_sandbox_command(
        command,
        settings.workspace_root,
        platform=sys.platform,
        sandbox_bin=shutil.which("sandbox-exec"),
        bwrap_bin=shutil.which("bwrap"),
    )
    proc = await asyncio.create_subprocess_exec(
        executable,
        *args,
        cwd=str(settings.workspace_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    max_total = settings.command_output_bytes
    stdout_cap = max_total // 3
    stderr_cap = max_total - stdout_cap
    total_drained = 0
    drain_limit = max_total * 10

    async def read_stream(stream, cap: int, stream_name: str) -> bytes:
        nonlocal total_drained
        buf = bytearray()
        truncated = False
        while chunk := await stream.read(65536):
            total_drained += len(chunk)
            if total_drained > drain_limit:
                raise ValueError(f"Command exceeded absolute stream safety limit ({drain_limit} bytes)")
            if len(buf) < cap:
                space = cap - len(buf)
                buf.extend(chunk[:space])
                if len(chunk) > space:
                    truncated = True
            else:
                truncated = True
        if truncated:
            buf.extend(f"\n... [{stream_name} truncated at {cap} bytes]".encode("utf-8"))
        return bytes(buf)

    readers = [
        asyncio.create_task(read_stream(proc.stdout, stdout_cap, "stdout")),
        asyncio.create_task(read_stream(proc.stderr, stderr_cap, "stderr")),
        asyncio.create_task(proc.wait()),
    ]
    try:
        out, err, _ = await asyncio.wait_for(asyncio.gather(*readers), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError, ValueError) as exc:
        # Stop children too: graceful SIGTERM first, escalating to SIGKILL after 500ms grace period.
        if proc.pid:
            try:
                # Stage 1: Send SIGTERM to entire process group
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    # Stage 2: Escalate to SIGKILL if process group remains alive after 500ms
                    os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()
            except ProcessLookupError:
                pass
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        if isinstance(exc, asyncio.CancelledError):
            raise
        if isinstance(exc, ValueError):
            return f"ERROR: command terminated: {exc}"
        return f"ERROR: command timed out after {timeout}s: {command}"
    stdout = out.decode("utf-8", errors="replace")
    stderr = err.decode("utf-8", errors="replace")
    return f"exit_code={proc.returncode}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
