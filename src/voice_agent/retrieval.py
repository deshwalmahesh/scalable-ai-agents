from __future__ import annotations

import threading
from typing import Literal

import httpx

JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
LOCAL_JINA_MODEL = "jinaai/jina-embeddings-v5-text-small"

_JINA_CLIENT: httpx.Client | None = None
_LOCAL_MODEL = None
_LOCAL_MODEL_LOCK = threading.Lock()


def _get_jina_client() -> httpx.Client:
    global _JINA_CLIENT
    if _JINA_CLIENT is None:
        _JINA_CLIENT = httpx.Client(timeout=30.0)
    return _JINA_CLIENT


def _get_local_model():
    global _LOCAL_MODEL
    with _LOCAL_MODEL_LOCK:
        if _LOCAL_MODEL is None:
            from sentence_transformers import SentenceTransformer

            # Tested on Apple M4. Let sentence-transformers pick the device automatically.
            _LOCAL_MODEL = SentenceTransformer(
                LOCAL_JINA_MODEL,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": "auto"},
            )
        return _LOCAL_MODEL


def warm_local_model() -> None:
    _get_local_model()


def embed_texts(
    *,
    texts: list[str],
    task: str,
    provider: Literal["local", "remote"],
    model: str,
    api_key: str | None = None,
) -> list[list[float]]:
    if provider == "remote":
        if not api_key:
            raise ValueError("JINA_API_KEY is required when EMBEDDING_PROVIDER=remote")
        return _embed_with_jina(texts=texts, task=task, api_key=api_key, model=model)

    return _embed_with_local_model(texts=texts, task=task)


def _embed_with_jina(
    *,
    texts: list[str],
    task: str,
    api_key: str,
    model: str,
) -> list[list[float]]:
    response = _get_jina_client().post(
        JINA_EMBEDDINGS_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": model,
            "task": task,
            "normalized": True,
            "input": texts,
        },
    )
    response.raise_for_status()
    payload = response.json()
    data = sorted(payload.get("data", []), key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def _embed_with_local_model(
    *,
    texts: list[str],
    task: str,
) -> list[list[float]]:
    if task == "retrieval.query":
        prompt_name = "query"
    elif task == "retrieval.passage":
        prompt_name = "document"
    else:
        raise ValueError(f"Unsupported local embedding task: {task}")

    embeddings = _get_local_model().encode(
        sentences=texts,
        task="retrieval",
        prompt_name=prompt_name,
        normalize_embeddings=True,
    )
    return [[float(value) for value in row] for row in embeddings]
