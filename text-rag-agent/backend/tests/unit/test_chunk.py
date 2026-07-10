from app.rag.chunk import chunk_pages
from app.rag.extract import PageText


def test_overlap_and_indexing():
    pages = [PageText(page=1, text="alpha beta gamma delta. " * 200)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.page == 1 and c.text for c in chunks)


def test_pages_preserved_across_sources():
    pages = [PageText(page=1, text="one " * 100), PageText(page=2, text="two " * 100)]
    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=20)
    assert {c.page for c in chunks} == {1, 2}


def test_blank_pages_yield_nothing():
    assert chunk_pages([PageText(page=None, text="   \n  ")]) == []
