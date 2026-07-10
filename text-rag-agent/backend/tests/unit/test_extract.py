import pytest
from docx import Document as Docx

from app.rag.extract import UnsupportedFormat, extension, extract_text
from tests.helpers import ASSETS


def test_extension_parsing():
    assert extension("Report.PDF") == "pdf"
    assert extension("a.b.TXT") == "txt"
    assert extension("noext") == ""


async def test_txt_reads_text():
    pages = await extract_text((ASSETS / "LONG_DOC.txt").read_bytes(), "LONG_DOC.txt")
    assert pages and pages[0].text.strip()


async def test_md_reads_text():
    pages = await extract_text((ASSETS / "SAMPLE_MD.md").read_bytes(), "SAMPLE_MD.md")
    assert pages and pages[0].text.strip()


async def test_pdf_has_page_numbers():
    pages = await extract_text((ASSETS / "SAMPLE_PDF.pdf").read_bytes(), "SAMPLE_PDF.pdf")
    assert pages and all(p.page for p in pages)


async def test_docx_roundtrip(tmp_path):
    path = tmp_path / "sample.docx"
    doc = Docx()
    doc.add_paragraph("Grounded answers require citations.")
    doc.save(path)
    pages = await extract_text(path.read_bytes(), "sample.docx")
    assert "citations" in pages[0].text


async def test_unsupported_type_raises():
    with pytest.raises(UnsupportedFormat):
        await extract_text(b"whatever", "archive.zip")
