"""The anti-hallucination gate: below the score floor => not grounded => no LLM call.
Only the external embed+search are faked; the real gate logic in pipeline.retrieve runs."""

import uuid

import pytest

from app.rag import pipeline
from app.rag.vectorstore import Hit


class _FakeLLM:
    async def embed_query(self, text):
        return [0.1] * 8


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits

    async def search(self, **kwargs):
        return self._hits


@pytest.fixture
def patch_rag(monkeypatch):
    def apply(hits):
        monkeypatch.setattr(pipeline, "get_llm", lambda: _FakeLLM())
        monkeypatch.setattr(pipeline, "get_vector_store", lambda: _FakeStore(hits))

    return apply


def _hit(score):
    return Hit(score=score, text="t", filename="f.pdf", page=1, document_id="d", chunk_index=0)


async def test_grounded_above_floor(patch_rag):
    patch_rag([_hit(0.9)])
    hits, grounded = await pipeline.retrieve(user_id=uuid.uuid4(), question="q", top_k=5, min_score=0.5)
    assert grounded is True and hits


async def test_not_grounded_below_floor(patch_rag):
    patch_rag([_hit(0.2)])
    _, grounded = await pipeline.retrieve(user_id=uuid.uuid4(), question="q", top_k=5, min_score=0.5)
    assert grounded is False


async def test_not_grounded_when_no_hits(patch_rag):
    patch_rag([])
    _, grounded = await pipeline.retrieve(user_id=uuid.uuid4(), question="q", top_k=5, min_score=0.5)
    assert grounded is False
