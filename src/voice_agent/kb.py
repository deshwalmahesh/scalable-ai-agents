from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import time

import chromadb

from voice_agent.data.knowledge_documents import KnowledgeDocument
from voice_agent.retrieval import embed_texts
from voice_agent.settings import AppSettings

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?;:])\s+|\s+(?=\d+\.)")
LEAD_IN_FRAGMENT_PATTERN = re.compile(r"(:|\b\d+\.)$")
LIST_MARKER_PATTERN = re.compile(r"^\d+\.$")
PREFERRED_SNIPPET_STARTERS = ("If ", "We ", "You ", "There ", "The ", "A ")
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
NON_POLICY_PHRASES = {
    "bye",
    "can you hear",
    "can you hear me",
    "good afternoon",
    "good evening",
    "good morning",
    "goodbye",
    "hello",
    "hey",
    "hi",
    "how are you",
    "ok",
    "okay",
    "sounds good",
    "thank you",
    "thanks",
    "yes",
}
NON_POLICY_TOKENS = {
    "bye",
    "good",
    "hear",
    "hello",
    "hey",
    "hi",
    "how",
    "ok",
    "okay",
    "thanks",
    "there",
    "you",
}
UNSUPPORTED_POLICY_RESPONSE = (
    "I can't answer that from this policy document. Please speak with a human insurance representative."
)
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


@dataclass(frozen=True, slots=True)
class SearchTrace:
    mode: str
    embedding_ms: float
    vector_query_ms: float
    document_load_ms: float
    rerank_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class GroundedRetrieval:
    query: str
    matches: list[KnowledgeMatch]
    trace: SearchTrace
    grounding_message: str


class KnowledgeBase:
    """Thin wrapper around a Chroma collection for grounded support content."""

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "KnowledgeBase":
        return cls(
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.kb_collection_name,
            chroma_provider=settings.chroma_provider,
            chroma_host=settings.chroma_host,
            chroma_api_key=settings.chroma_api_key,
            chroma_tenant=settings.chroma_tenant,
            chroma_database=settings.chroma_database,
            jina_api_key=settings.jina_api_key,
            jina_embedding_model=settings.jina_embedding_model,
            embedding_provider=settings.embedding_provider,
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
        chroma_provider: str | None = None,
        chroma_host: str | None = None,
        chroma_api_key: str | None = None,
        chroma_tenant: str | None = None,
        chroma_database: str | None = None,
        jina_api_key: str | None = None,
        jina_embedding_model: str = "jina-embeddings-v5-text-small",
        embedding_provider: str = "lexical",
        embedding_weight: float = 0.7,
        lexical_weight: float = 0.3,
        min_combined_score: float = 0.45,
        min_vector_score: float = 0.5,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._jina_api_key = jina_api_key
        self._jina_embedding_model = jina_embedding_model
        self._embedding_provider = embedding_provider
        self._embedding_weight = embedding_weight
        self._lexical_weight = lexical_weight
        self._min_combined_score = min_combined_score
        self._min_vector_score = min_vector_score
        use_cloud_client = (
            chroma_provider == "remote"
            if chroma_provider is not None
            else should_use_cloud_client(chroma_api_key, chroma_tenant, chroma_database)
        )
        if use_cloud_client:
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
        payload = {
            "ids": [document.id for document in documents],
            "documents": [document.content for document in documents],
            "metadatas": [
                {"title": document.title, **document.metadata} for document in documents
            ],
        }
        if self._embedding_provider != "lexical":
            payload["embeddings"] = self._embed_texts(
                [document.content for document in documents],
                task="retrieval.passage",
            )
        self._collection.upsert(**payload)

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, limit: int = 3) -> list[KnowledgeMatch]:
        matches, _ = self.search_with_trace(query, limit)
        return matches

    async def async_search_with_trace(
        self,
        query: str,
        limit: int = 3,
    ) -> tuple[list[KnowledgeMatch], SearchTrace]:
        return await asyncio.to_thread(self.search_with_trace, query, limit)

    def search_with_trace(self, query: str, limit: int = 3) -> tuple[list[KnowledgeMatch], SearchTrace]:
        query_text = query.strip().lower()
        if not query_text:
            return [], SearchTrace(
                mode="empty",
                embedding_ms=0.0,
                vector_query_ms=0.0,
                document_load_ms=0.0,
                rerank_ms=0.0,
                total_ms=0.0,
            )
        started_at = time.perf_counter()
        document_count = self._collection.count()
        if document_count == 0:
            logger.warning(
                "Knowledge search skipped because collection is empty collection=%s query=%s",
                self._collection_name,
                query_text,
            )
            return [], SearchTrace(
                mode="empty_collection",
                embedding_ms=0.0,
                vector_query_ms=0.0,
                document_load_ms=0.0,
                rerank_ms=0.0,
                total_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
        if self._embedding_provider == "lexical":
            matches = self._lexical_search(query_text, limit)
            self._log_search_results(query_text, matches, mode="lexical")
            return matches, SearchTrace(
                mode="lexical",
                embedding_ms=0.0,
                vector_query_ms=0.0,
                document_load_ms=0.0,
                rerank_ms=round((time.perf_counter() - started_at) * 1000, 2),
                total_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )

        matches, trace = self._hybrid_search(query_text, limit, document_count)
        self._log_search_results(query_text, matches, mode="hybrid")
        return matches, trace

    def _embed_texts(self, texts: list[str], task: str) -> list[list[float]]:
        return embed_texts(
            texts=texts,
            task=task,
            provider=self._embedding_provider,
            api_key=self._jina_api_key,
            model=self._jina_embedding_model,
        )

    def _lexical_search(self, query_text: str, limit: int) -> list[KnowledgeMatch]:
        records = self._collection.get(
            limit=max(self._collection.count(), limit),
            include=["documents", "metadatas"],
        )
        matches: list[KnowledgeMatch] = []

        for doc_id, content, metadata in zip(
            records.get("ids", []),
            records.get("documents", []) or [],
            records.get("metadatas", []) or [],
            strict=False,
        ):
            resolved_metadata = {
                key: str(value) for key, value in (metadata or {}).items()
            }
            title = resolved_metadata.get("title", "")
            resolved_content = str(content or "")
            score = lexical_score(query_text, title, resolved_content)
            if score <= 0:
                continue
            matches.append(
                KnowledgeMatch(
                    id=str(doc_id),
                    title=title,
                    content=resolved_content,
                    score=float(score),
                    vector_score=0.0,
                    lexical_score=float(score),
                    metadata=resolved_metadata,
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def _hybrid_search(
        self,
        query_text: str,
        limit: int,
        document_count: int,
    ) -> tuple[list[KnowledgeMatch], SearchTrace]:
        started_at = time.perf_counter()
        embedding_started_at = time.perf_counter()
        query_embedding = self._embed_texts([query_text], task="retrieval.query")[0]
        embedding_ms = round((time.perf_counter() - embedding_started_at) * 1000, 2)
        candidate_limit = min(max(limit * 6, 10), document_count)
        vector_query_started_at = time.perf_counter()
        query_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_limit,
            include=["documents", "metadatas", "distances"],
        )
        vector_query_ms = round(
            (time.perf_counter() - vector_query_started_at) * 1000,
            2,
        )
        document_load_ms = 0.0
        matches: list[KnowledgeMatch] = []
        rerank_started_at = time.perf_counter()
        result_ids = [str(value) for value in query_results.get("ids", [[]])[0]]
        result_documents = query_results.get("documents", [[]])[0] or []
        result_metadatas = query_results.get("metadatas", [[]])[0] or []
        candidate_scores = build_vector_score_map(
            ids=result_ids,
            distances=query_results.get("distances", [[]])[0],
        )
        lexical_scores: list[tuple[str, float]] = []

        for doc_id, content, metadata in zip(
            result_ids,
            result_documents,
            result_metadatas,
            strict=False,
        ):
            resolved_metadata = {
                key: str(value) for key, value in (metadata or {}).items()
            }
            lexical_scores.append(
                (
                    doc_id,
                    lexical_coverage_score(
                        query_text,
                        resolved_metadata.get("title", ""),
                        str(content or ""),
                    ),
                )
            )

        max_lexical = max((score for _, score in lexical_scores), default=0.0)
        lexical_map = {
            doc_id: (score / max_lexical if max_lexical > 0 else 0.0)
            for doc_id, score in lexical_scores
        }

        for doc_id, content, metadata in zip(
            result_ids,
            result_documents,
            result_metadatas,
            strict=False,
        ):
            resolved_metadata = {
                key: str(value) for key, value in (metadata or {}).items()
            }
            lexical = lexical_map.get(doc_id, 0.0)
            vector = candidate_scores.get(doc_id, 0.0)
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
                    id=doc_id,
                    title=resolved_metadata.get("title", ""),
                    content=str(content or ""),
                    score=round(combined_score, 4),
                    vector_score=round(vector, 4),
                    lexical_score=round(lexical, 4),
                    metadata=resolved_metadata,
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        rerank_ms = round((time.perf_counter() - rerank_started_at) * 1000, 2)
        total_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return matches[:limit], SearchTrace(
            mode="hybrid",
            embedding_ms=embedding_ms,
            vector_query_ms=vector_query_ms,
            document_load_ms=document_load_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
        )

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


class RetrievalService:
    """Single retrieval boundary for grounded policy context."""

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "RetrievalService":
        return cls(KnowledgeBase.from_settings(settings))

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._knowledge_base = knowledge_base

    def count(self) -> int:
        return self._knowledge_base.count()

    def should_ground(self, query: str) -> bool:
        normalized_query = " ".join(tokenize(query))
        if not normalized_query:
            return False
        if normalized_query in NON_POLICY_PHRASES:
            return False
        tokens = normalized_query.split()
        return not (
            len(tokens) <= 4
            and tokens
            and all(token in NON_POLICY_TOKENS for token in tokens)
        )

    async def async_ground(
        self,
        query: str,
        limit: int = 1,
    ) -> GroundedRetrieval:
        return await asyncio.to_thread(self.ground, query, limit)

    def ground(self, query: str, limit: int = 1) -> GroundedRetrieval:
        normalized_query = query.strip()
        if not self.should_ground(normalized_query):
            return GroundedRetrieval(
                query=normalized_query,
                matches=[],
                trace=SearchTrace(
                    mode="skipped",
                    embedding_ms=0.0,
                    vector_query_ms=0.0,
                    document_load_ms=0.0,
                    rerank_ms=0.0,
                    total_ms=0.0,
                ),
                grounding_message=(
                    "This user turn is conversational small talk or a connection check. "
                    "Answer briefly and naturally without citing policy facts."
                ),
            )
        matches, trace = self._knowledge_base.search_with_trace(normalized_query, limit)
        return GroundedRetrieval(
            query=normalized_query,
            matches=matches,
            trace=trace,
            grounding_message=build_grounding_message(
                query=normalized_query,
                matches=matches,
            ),
        )


def build_grounding_message(
    *,
    query: str,
    matches: list[KnowledgeMatch],
) -> str:
    if not matches:
        return (
            "No grounded policy excerpts matched this user turn. "
            "If the user is asking for policy facts or policy terms, say: "
            f"'{UNSUPPORTED_POLICY_RESPONSE}' "
            "If the user is greeting you or making small talk, answer briefly and naturally."
        )

    excerpts: list[str] = []
    for index, match in enumerate(matches, start=1):
        source_url = match.metadata.get("source_url", "")
        snippet = select_grounding_snippet(query, match.content)
        excerpts.append(
            f"[Excerpt {index}] Title: {match.title}\n"
            f"Content: {snippet}\n"
            f"Source: {source_url}"
        )

    return (
        "Grounding for this turn only. Answer using only the policy facts below. "
        "Do not mention tools, retrieval, or internal process. "
        "If the excerpts support the user's question, answer directly and briefly. "
        "When the excerpts give an exact number, date, or time window, repeat that exact value and do not broaden it into a range or a typical industry practice. "
        "If they support only part of the question, answer the supported part and briefly say what remains unclear instead of refusing. "
        "If an excerpt gives one concrete condition under which a claim or benefit is not paid, state that condition directly even if the user asked more broadly. "
        "Only refuse when the excerpts do not support the answer at all.\n\n"
        f"User question: {query}\n\n" + "\n\n".join(excerpts)
    )


def select_grounding_snippet(query: str, content: str) -> str:
    raw_sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(content) if sentence.strip()]
    sentences: list[str] = []
    index = 0
    while index < len(raw_sentences):
        sentence = raw_sentences[index]
        if LIST_MARKER_PATTERN.fullmatch(sentence) and index + 1 < len(raw_sentences):
            sentences.append(f"{sentence} {raw_sentences[index + 1]}")
            index += 2
            continue
        sentences.append(sentence)
        index += 1

    if len(sentences) <= 1:
        return content.strip()
    if not significant_tokens(query):
        return sentences[0]

    scored_sentences: list[tuple[int, float]] = []
    for index, sentence in enumerate(sentences):
        score = lexical_coverage_score_from_tokens(query, set(tokenize(sentence)))
        if any(token.isdigit() for token in tokenize(sentence)):
            score += 0.1
        scored_sentences.append((index, score))

    best_index, best_score = max(scored_sentences, key=lambda item: item[1])
    if best_score <= 0:
        return sentences[0]
    snippet = sentences[best_index]
    if best_index + 1 < len(sentences) and LEAD_IN_FRAGMENT_PATTERN.search(snippet):
        snippet = f"{snippet} {sentences[best_index + 1]}"
    return clean_grounding_snippet_text(snippet)


def clean_grounding_snippet_text(snippet: str) -> str:
    text = " ".join(snippet.split()).strip()
    if not text:
        return ""

    for starter in PREFERRED_SNIPPET_STARTERS:
        if text.startswith(starter):
            return text

    candidate_positions = [
        text.find(f" {starter}")
        for starter in PREFERRED_SNIPPET_STARTERS
    ]
    candidate_positions = [position for position in candidate_positions if position > 0]
    if not candidate_positions:
        return text
    return text[min(candidate_positions) + 1 :]


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
    return lexical_coverage_score_from_tokens(query, document_tokens)


def lexical_coverage_score_from_tokens(query: str, document_tokens: set[str]) -> float:
    query_tokens = significant_tokens(query)
    if not query_tokens:
        return 0.0

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
