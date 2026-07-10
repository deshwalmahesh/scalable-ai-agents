"""Qdrant vector store with native hybrid search.

Each chunk is stored as two named vectors: a dense embedding (cosine) and a BM25 sparse
vector (computed locally by fastembed). Queries run Qdrant's Query API with a dense +
sparse `prefetch` fused by Reciprocal Rank Fusion (RRF). A parallel dense-only query
provides the cosine similarity used for the anti-hallucination gate and shown to the user.

Every operation is filtered by `user_id` (tenant isolation). Collection is created lazily
at the real embedding dimension. Transient network errors are retried (the client has none).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from app.config import get_settings
from app.rag.chunk import Chunk

log = structlog.get_logger()

_UPSERT_BATCH = 128
_DENSE = "dense"
_BM25 = "bm25"
_retry = retry(
    retry=retry_if_exception_type((ResponseHandlingException, ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=0.5, max=8),
    reraise=True,
)


@dataclass
class Hit:
    score: float  # dense cosine similarity (shown to the user)
    text: str
    filename: str
    page: int | None
    document_id: str
    chunk_index: int


class VectorStore:
    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)
        self._collection = s.qdrant_collection
        self._bm25_model = s.bm25_model

    async def _ensure_collection(self, dim: int) -> None:
        if await self._client.collection_exists(self._collection):
            # A collection left over from an older, dense-only schema lacks the bm25 sparse
            # vector, so every hybrid upsert would fail with a cryptic Qdrant error. Detect
            # that mismatch and surface an actionable message instead (non-destructive — the
            # operator drops the collection to have it rebuilt with hybrid search).
            info = await self._client.get_collection(self._collection)
            if _BM25 not in (info.config.params.sparse_vectors or {}):
                raise RuntimeError(
                    f"Qdrant collection '{self._collection}' predates hybrid search (missing the "
                    f"'{_BM25}' sparse vector). Delete this collection so it is rebuilt, then re-ingest."
                )
            return
        try:
            await self._client.create_collection(
                self._collection,
                vectors_config={_DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={_BM25: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            await self._client.create_payload_index(
                self._collection,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            log.info("qdrant.collection_created", collection=self._collection, dim=dim)
        except Exception:  # concurrent creator won the race — fine if it now exists
            if not await self._client.collection_exists(self._collection):
                raise

    @staticmethod
    def _filter(user_id: UUID, document_ids: list[UUID] | None) -> models.Filter:
        must: list[models.Condition] = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=str(user_id)))
        ]
        if document_ids:
            must.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=[str(d) for d in document_ids])
                )
            )
        return models.Filter(must=must)

    @_retry
    async def _upsert_batch(self, points: list[models.PointStruct]) -> None:
        await self._client.upsert(self._collection, points=points, wait=True)

    async def upsert(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
        filename: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if not vectors:
            return
        await self._ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    _DENSE: vec,
                    _BM25: models.Document(text=ch.text, model=self._bm25_model),
                },
                payload={
                    "user_id": str(user_id),
                    "document_id": str(document_id),
                    "filename": filename,
                    "page": ch.page,
                    "chunk_index": ch.index,
                    "text": ch.text,
                },
            )
            for ch, vec in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), _UPSERT_BATCH):
            await self._upsert_batch(points[i : i + _UPSERT_BATCH])

    @staticmethod
    def _to_hit(point, score: float) -> Hit:
        pl = point.payload or {}
        return Hit(
            score=score,
            text=pl.get("text", ""),
            filename=pl.get("filename", ""),
            page=pl.get("page"),
            document_id=pl.get("document_id", ""),
            chunk_index=pl.get("chunk_index", 0),
        )

    @_retry
    async def search(
        self,
        *,
        user_id: UUID,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        document_ids: list[UUID] | None = None,
    ) -> list[Hit]:
        """Hybrid (dense + BM25, RRF) ordering; each hit's `.score` is its dense cosine."""
        if not await self._client.collection_exists(self._collection):
            return []
        s = get_settings()
        flt = self._filter(user_id, document_ids)
        over = max(top_k, s.hybrid_prefetch)

        # Dense-only pass: cosine scores for the gate + display.
        dense = await self._client.query_points(
            self._collection,
            query=query_vector,
            using=_DENSE,
            query_filter=flt,
            limit=over,
            with_payload=False,
        )
        cosine = {p.id: p.score for p in dense.points}

        # Hybrid pass: dense + BM25 prefetch fused by RRF.
        hybrid = await self._client.query_points(
            self._collection,
            prefetch=[
                models.Prefetch(query=query_vector, using=_DENSE, limit=over, filter=flt),
                models.Prefetch(
                    query=models.Document(text=query_text, model=self._bm25_model),
                    using=_BM25,
                    limit=over,
                    filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [self._to_hit(p, cosine.get(p.id, p.score)) for p in hybrid.points]

    @_retry
    async def delete_document(self, *, user_id: UUID, document_id: UUID) -> None:
        if not await self._client.collection_exists(self._collection):
            return
        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=self._filter(user_id, [document_id])),
            wait=True,
        )

    async def ping(self) -> None:
        await self._client.get_collections()

    async def aclose(self) -> None:
        await self._client.close()


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


async def aclose_vector_store() -> None:
    global _store
    if _store is not None:
        await _store.aclose()
        _store = None
