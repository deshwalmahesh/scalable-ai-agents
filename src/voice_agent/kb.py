from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

import chromadb

from voice_agent.data.knowledge_documents import KnowledgeDocument
from voice_agent.retrieval import embed_with_jina, local_embedding
from voice_agent.settings import AppSettings

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "within",
    "you",
    "your",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KnowledgeMatch:
    id: str
    title: str
    content: str
    score: float
    vector_score: float
    lexical_score: float
    metadata: dict[str, str]


class KnowledgeBase:
    """Thin wrapper around a Chroma collection for grounded support content."""

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "KnowledgeBase":
        return cls(
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.kb_collection_name,
            chroma_host=settings.chroma_host,
            chroma_api_key=settings.chroma_api_key,
            chroma_tenant=settings.chroma_tenant,
            chroma_database=settings.chroma_database,
            jina_api_key=settings.jina_api_key,
            jina_embedding_model=settings.jina_embedding_model,
            embedding_weight=settings.kb_embedding_weight,
            lexical_weight=settings.kb_lexical_weight,
            min_combined_score=settings.kb_min_combined_score,
            min_vector_score=settings.kb_min_vector_score,
        )

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        *,
        chroma_host: str | None = None,
        chroma_api_key: str | None = None,
        chroma_tenant: str | None = None,
        chroma_database: str | None = None,
        jina_api_key: str | None = None,
        jina_embedding_model: str = "jina-embeddings-v5-text-small",
        embedding_weight: float = 0.7,
        lexical_weight: float = 0.3,
        min_combined_score: float = 0.45,
        min_vector_score: float = 0.5,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._jina_api_key = jina_api_key
        self._jina_embedding_model = jina_embedding_model
        self._embedding_weight = embedding_weight
        self._lexical_weight = lexical_weight
        self._min_combined_score = min_combined_score
        self._min_vector_score = min_vector_score
        if should_use_cloud_client(chroma_api_key, chroma_tenant, chroma_database):
            self._client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=chroma_tenant,
                database=chroma_database,
                cloud_host=chroma_host or "api.trychroma.com",
            )
        else:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def reset_collection(self) -> None:
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def upsert_documents(self, documents: list[KnowledgeDocument]) -> None:
        logger.info(
            "Upserting knowledge documents collection=%s document_count=%s",
            self._collection_name,
            len(documents),
        )
        self._collection.upsert(
            ids=[document.id for document in documents],
            documents=[document.content for document in documents],
            metadatas=[
                {"title": document.title, **document.metadata} for document in documents
            ],
            embeddings=self._embed_texts(
                [document.content for document in documents],
                task="retrieval.passage",
            ),
        )

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, limit: int = 3) -> list[KnowledgeMatch]:
        query_text = query.strip().lower()
        if not query_text:
            return []
        document_count = self._collection.count()
        if document_count == 0:
            logger.warning(
                "Knowledge search skipped because collection is empty collection=%s query=%s",
                self._collection_name,
                query_text,
            )
            return []
        if not self._jina_api_key:
            matches = self._lexical_search(query_text, limit)
            self._log_search_results(query_text, matches, mode="lexical")
            return matches

        matches = self._hybrid_search(query_text, limit, document_count)
        self._log_search_results(query_text, matches, mode="hybrid")
        return matches

    def _embed_texts(self, texts: list[str], task: str) -> list[list[float]]:
        if self._jina_api_key:
            return embed_with_jina(
                texts=texts,
                task=task,
                api_key=self._jina_api_key,
                model=self._jina_embedding_model,
            )
        return [local_embedding(text) for text in texts]

    def _lexical_search(self, query_text: str, limit: int) -> list[KnowledgeMatch]:
        records = self._collection.get(
            limit=max(self._collection.count(), limit),
            include=["documents", "metadatas"],
        )
        ids = records.get("ids", [])
        documents = records.get("documents", []) or []
        metadatas = records.get("metadatas", []) or []
        matches: list[KnowledgeMatch] = []

        for doc_id, content, metadata in zip(ids, documents, metadatas, strict=False):
            metadata = metadata or {}
            title = str(metadata.get("title", ""))
            score = lexical_score(query_text, title, str(content))
            if score <= 0:
                continue
            matches.append(
                KnowledgeMatch(
                    id=str(doc_id),
                    title=title,
                    content=str(content),
                    score=float(score),
                    vector_score=0.0,
                    lexical_score=float(score),
                    metadata={key: str(value) for key, value in metadata.items()},
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def _hybrid_search(
        self,
        query_text: str,
        limit: int,
        document_count: int,
    ) -> list[KnowledgeMatch]:
        query_embedding = self._embed_texts([query_text], task="retrieval.query")[0]
        candidate_limit = min(max(limit * 6, 10), document_count)
        query_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_limit,
        )
        candidate_scores = build_vector_score_map(
            ids=query_results.get("ids", [[]])[0],
            distances=query_results.get("distances", [[]])[0],
        )
        records = self._collection.get(
            limit=document_count,
            include=["documents", "metadatas"],
        )
        matches: list[KnowledgeMatch] = []
        lexical_scores: list[tuple[str, float]] = []

        for doc_id, content, metadata in zip(
            records.get("ids", []),
            records.get("documents", []) or [],
            records.get("metadatas", []) or [],
            strict=False,
        ):
            metadata = metadata or {}
            lexical_scores.append(
                (
                    str(doc_id),
                    lexical_coverage_score(
                        query_text,
                        str(metadata.get("title", "")),
                        str(content),
                    ),
                )
            )

        max_lexical = max((score for _, score in lexical_scores), default=0.0)
        lexical_map = {
            doc_id: (score / max_lexical if max_lexical > 0 else 0.0)
            for doc_id, score in lexical_scores
        }

        for doc_id, content, metadata in zip(
            records.get("ids", []),
            records.get("documents", []) or [],
            records.get("metadatas", []) or [],
            strict=False,
        ):
            metadata = metadata or {}
            title = str(metadata.get("title", ""))
            lexical = lexical_map.get(str(doc_id), 0.0)
            vector = candidate_scores.get(str(doc_id), 0.0)
            combined_score = (
                self._embedding_weight * vector
                + self._lexical_weight * lexical
            )
            if (
                combined_score < self._min_combined_score
                or vector < self._min_vector_score
            ):
                continue
            matches.append(
                KnowledgeMatch(
                    id=str(doc_id),
                    title=title,
                    content=str(content),
                    score=round(combined_score, 4),
                    vector_score=round(vector, 4),
                    lexical_score=round(lexical, 4),
                    metadata={key: str(value) for key, value in metadata.items()},
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def _log_search_results(
        self,
        query_text: str,
        matches: list[KnowledgeMatch],
        *,
        mode: str,
    ) -> None:
        logger.info(
            "Knowledge search completed collection=%s mode=%s query=%s match_count=%s min_combined_score=%s min_vector_score=%s top_matches=%s",
            self._collection_name,
            mode,
            query_text,
            len(matches),
            self._min_combined_score,
            self._min_vector_score,
            [
                {
                    "id": match.id,
                    "title": match.title[:120],
                    "score": match.score,
                    "vector_score": match.vector_score,
                    "lexical_score": match.lexical_score,
                    "page_number": match.metadata.get("page_number", ""),
                }
                for match in matches[:3]
            ],
        )
        if not matches:
            logger.warning(
                "Knowledge search produced no grounded matches collection=%s query=%s",
                self._collection_name,
                query_text,
            )


def should_use_cloud_client(
    chroma_api_key: str | None,
    chroma_tenant: str | None,
    chroma_database: str | None,
) -> bool:
    return bool(
        chroma_api_key
        and chroma_api_key != "YOUR_API_KEY"
        and chroma_tenant
        and chroma_database
    )


def lexical_score(query: str, title: str, content: str) -> int:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0

    combined_text = f"{title} {content}".lower()
    combined_tokens = set(tokenize(combined_text))
    score = 0

    if query in combined_text:
        score += 5

    for token in query_tokens:
        if token in combined_tokens:
            score += 2 if token in title.lower() else 1
            continue
        if any(share_token_stem(token, candidate) for candidate in combined_tokens):
            score += 2 if any(share_token_stem(token, title_token) for title_token in tokenize(title)) else 1

    return score


def lexical_coverage_score(query: str, title: str, content: str) -> float:
    query_tokens = significant_tokens(query)
    if not query_tokens:
        return 0.0

    document_tokens = set(tokenize(f"{title} {content}"))
    matched = 0
    for token in query_tokens:
        if token in document_tokens or any(
            share_token_stem(token, candidate) for candidate in document_tokens
        ):
            matched += 1
    return matched / len(query_tokens)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token.isdigit() or len(token) > 2
    ]


def significant_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def build_vector_score_map(ids: list[object], distances: list[object]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for doc_id, distance in zip(ids, distances, strict=False):
        try:
            numeric_distance = float(distance)
        except (TypeError, ValueError):
            numeric_distance = 1.0
        scores[str(doc_id)] = 1.0 / (1.0 + max(0.0, numeric_distance))
    return scores


def share_token_stem(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) < 6:
        return False
    prefix_length = min(len(left), len(right), 8)
    return left[:prefix_length] == right[:prefix_length]
