"""Text extraction for PDF/DOCX/TXT. One funnel keyed by extension; blocking parsers
run off the event loop. Page numbers (PDF) are preserved for citations."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO

from docx import Document
from pypdf import PdfReader

SUPPORTED = ("pdf", "docx", "doc", "txt", "md")


@dataclass
class PageText:
    page: int | None
    text: str


class UnsupportedFormat(Exception):
    pass


def extension(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def _pdf(data: bytes) -> list[PageText]:
    reader = PdfReader(BytesIO(data))
    out: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(PageText(page=i, text=text))
    return out


def _docx(data: bytes) -> list[PageText]:
    doc = Document(BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [PageText(page=None, text=text)] if text.strip() else []


def _text(data: bytes) -> list[PageText]:
    text = data.decode("utf-8", errors="replace")
    return [PageText(page=None, text=text)] if text.strip() else []


def _doc(data: bytes) -> list[PageText]:
    # Legacy binary .doc has no clean pure-Python reader; antiword (open-source, in the
    # Docker image) is the standard extractor. It needs a real path, so use a private
    # per-process temp file (not shared state) that is removed immediately.
    if shutil.which("antiword") is None:
        raise UnsupportedFormat("doc (antiword not installed)")
    with tempfile.NamedTemporaryFile(suffix=".doc") as tmp:
        tmp.write(data)
        tmp.flush()
        text = subprocess.run(
            ["antiword", tmp.name], capture_output=True, text=True, timeout=60, check=True
        ).stdout
    return [PageText(page=None, text=text)] if text.strip() else []


def _extract_sync(data: bytes, filename: str) -> list[PageText]:
    ext = extension(filename)
    if ext == "pdf":
        return _pdf(data)
    if ext == "docx":
        return _docx(data)
    if ext == "doc":
        return _doc(data)
    if ext in ("txt", "md"):
        return _text(data)
    raise UnsupportedFormat(ext)


async def extract_text(data: bytes, filename: str) -> list[PageText]:
    return await asyncio.to_thread(_extract_sync, data, filename)
