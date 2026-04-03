from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
import re

import httpx
from pypdf import PdfReader

from voice_agent.data.knowledge_documents import KnowledgeDocument
from voice_agent.kb import KnowledgeBase

DEFAULT_POLICY_PDF_URL = (
    "https://www.rbcinsurance.com/samplepolicy/pdf/your-term-sample-policy.pdf"
)
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 250
PAGE_MARKER = re.compile(r"Page\s+\d+\s+Specimen", re.IGNORECASE)
EXTRA_SPACES = re.compile(r"[ \t]+")
EXTRA_NEWLINES = re.compile(r"\n{3,}")
SECTION_HEADING = re.compile(r"^[A-Z](?: \d+(?:\.\d+)?)? .+")
TOC_LINE = re.compile(r"^[A-Z](?: \d+)? .+ \d+$")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    source_url: str
    document_count: int
    documents: list[KnowledgeDocument]


def ingest_pdf_into_kb(
    kb: KnowledgeBase,
    *,
    pdf_url: str = DEFAULT_POLICY_PDF_URL,
) -> IngestionResult:
    logger.info("Starting PDF ingestion pdf_url=%s", pdf_url)
    result = build_pdf_documents(pdf_url)
    kb.reset_collection()
    kb.upsert_documents(result.documents)
    logger.info(
        "Completed PDF ingestion pdf_url=%s document_count=%s",
        pdf_url,
        result.document_count,
    )
    return result


def build_pdf_documents(pdf_url: str) -> IngestionResult:
    pdf_bytes = download_pdf(pdf_url)
    pages = extract_pdf_pages(pdf_bytes)
    documents: list[KnowledgeDocument] = []

    for page_number, page_text in enumerate(pages, start=1):
        for chunk_index, chunk in enumerate(chunk_page_text(page_text), start=1):
            chunk_title = extract_chunk_title(chunk, page_number, chunk_index)
            documents.append(
                KnowledgeDocument(
                    id=f"policy-page-{page_number}-chunk-{chunk_index}",
                    title=chunk_title,
                    content=chunk,
                    metadata={
                        "category": "policy",
                        "source_url": pdf_url,
                        "page_number": str(page_number),
                        "section_title": chunk_title,
                    },
                )
            )

    return IngestionResult(
        source_url=pdf_url,
        document_count=len(documents),
        documents=documents,
    )


def download_pdf(pdf_url: str) -> bytes:
    logger.info("Downloading PDF pdf_url=%s", pdf_url)
    response = httpx.get(pdf_url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def extract_pdf_pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [clean_page_text(page.extract_text() or "") for page in reader.pages]
    logger.info("Extracted PDF text page_count=%s", len(pages))
    return pages


def clean_page_text(text: str) -> str:
    text = PAGE_MARKER.sub("", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")
    lines = [EXTRA_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    kept_lines = [
        line
        for line in lines
        if line
        and not line.startswith("YOURTERM ")
        and line not in {"Specimen", "Table of Contents"}
    ]
    if is_table_of_contents(kept_lines):
        return ""
    cleaned = "\n".join(kept_lines)
    cleaned = EXTRA_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()


def chunk_page_text(page_text: str) -> list[str]:
    if len(page_text) < MIN_CHUNK_CHARS:
        return []

    paragraphs = split_into_paragraphs(page_text)
    return [paragraph for paragraph in paragraphs if len(paragraph) >= MIN_CHUNK_CHARS]


def split_into_paragraphs(page_text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.extend(split_long_paragraph(" ".join(current)))
                current = []
            continue

        if is_section_heading(line) and current:
            paragraphs.extend(split_long_paragraph(" ".join(current)))
            current = [line]
            continue

        current.append(line)

    if current:
        paragraphs.extend(split_long_paragraph(" ".join(current)))

    return [paragraph for paragraph in paragraphs if len(paragraph) >= MIN_CHUNK_CHARS]


def split_long_paragraph(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_CHUNK_CHARS:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_length = len(sentence)
        if current and current_length + sentence_length + 1 > MAX_CHUNK_CHARS:
            chunks.append(" ".join(current))
            current = [sentence]
            current_length = sentence_length
            continue
        current.append(sentence)
        current_length += sentence_length + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def is_section_heading(line: str) -> bool:
    return bool(SECTION_HEADING.match(line))


def is_table_of_contents(lines: list[str]) -> bool:
    if not lines:
        return False
    toc_lines = sum(1 for line in lines if TOC_LINE.match(line))
    return lines[0].startswith("Policy Schedule") and toc_lines >= 8


def extract_chunk_title(chunk: str, page_number: int, chunk_index: int) -> str:
    first_line = chunk.splitlines()[0].strip()
    if first_line:
        return first_line[:120]
    return f"Policy PDF page {page_number} chunk {chunk_index}"
