from __future__ import annotations

import hashlib
import math
import re

import httpx

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
LOCAL_EMBEDDING_DIMENSION = 64
JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"


def embed_with_jina(
    *,
    texts: list[str],
    task: str,
    api_key: str,
    model: str,
) -> list[list[float]]:
    response = httpx.post(
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
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    data = sorted(payload.get("data", []), key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def local_embedding(text: str) -> list[float]:
    vector = [0.0] * LOCAL_EMBEDDING_DIMENSION
    for token in TOKEN_PATTERN.findall(text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % LOCAL_EMBEDDING_DIMENSION
        sign = -1.0 if digest[1] % 2 else 1.0
        weight = 1.0 + (digest[2] / 255.0)
        vector[index] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [round(value / norm, 8) for value in vector]
