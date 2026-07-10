"""Split extracted pages into overlapping chunks (proven RecursiveCharacterTextSplitter).
Each chunk keeps its source page and a running index for citation."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.extract import PageText


@dataclass
class Chunk:
    text: str
    page: int | None
    index: int


def chunk_pages(
    pages: list[PageText], chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[Chunk]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or s.chunk_size, chunk_overlap=chunk_overlap or s.chunk_overlap
    )
    chunks: list[Chunk] = []
    for pt in pages:
        for piece in splitter.split_text(pt.text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, page=pt.page, index=len(chunks)))
    return chunks
