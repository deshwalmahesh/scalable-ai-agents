"""Artifact store: large tool outputs live here, not in the model context.

IDEA_SCALABLE_AGENT.md: "Large command output should not be copied into the model context.
Store it as an artefact and return a bounded preview + artifact ID." This is the local,
filesystem-backed stand-in for Ceph RGW / S3 object storage — same interface, swap the backend.

An artifact has a random id; a sidecar `.meta.json` records provenance
(which tool produced it, size, when). The agent gets `read_artifact(id, offset, limit)` to pull
back exactly the slice it needs.
"""

import json
import time
import uuid
from pathlib import Path
from general_coding_agent.config import settings


def _dir() -> Path:
    d = settings.artifacts_dir
    d = d if d.is_absolute() else (Path.cwd() / d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_artifact(content: str, tool: str = "") -> dict:
    """Persist content and size/provenance metadata; I/O failures propagate."""
    artifact_id = uuid.uuid4().hex[:12]
    base = _dir() / artifact_id
    base.with_suffix(".txt").write_text(content, encoding="utf-8")
    meta = {"id": artifact_id, "tool": tool, "chars": len(content),
            "bytes": len(content.encode("utf-8")), "created": time.time()}
    base.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def read_artifact_slice(artifact_id: str, offset: int = 0, limit: int = 4000) -> str:
    """Return a [offset, offset+limit) character slice of an artifact, or an error string."""
    base_dir = _dir().resolve()
    path = (_dir() / f"{artifact_id}.txt").resolve()
    if not path.is_relative_to(base_dir):
        return f"ERROR: invalid artifact_id (path traversal denied): {artifact_id}"
    if not path.exists():
        return f"ERROR: artifact '{artifact_id}' not found"
    offset = max(0, offset)
    limit = min(max(1, limit), settings.artifact_read_chars)
    # Character offsets cannot be used as UTF-8 byte seeks. Skip in bounded chunks
    # instead of allocating the entire artifact; retain one character of lookahead.
    with path.open(encoding="utf-8", errors="replace") as stream:
        remaining = offset
        skipped = 0
        while remaining:
            part = stream.read(min(remaining, settings.artifact_read_chars))
            if not part:
                break
            remaining -= len(part)
            skipped += len(part)
        chunk = stream.read(limit)
        more = bool(stream.read(1))
    end = skipped + len(chunk)
    tail = f"\n...[more: read_artifact('{artifact_id}', offset={end})]" if more else ""
    return f"[artifact {artifact_id} chars {skipped}-{end}]\n{chunk}{tail}"
