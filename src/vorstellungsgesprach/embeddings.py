"""Embedding backends.

Default backend is Google Gemini (``models/text-embedding-004``, or the
current Gemini embedding model). A deterministic hash-based ``local``
backend is provided ONLY for offline smoke tests (no network / no API key).
Local embeddings carry no semantic meaning; in that mode retrieval quality
comes almost entirely from the BM25 lexical channel.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import List

from vorstellungsgesprach import conf

logger = logging.getLogger("vorstellungsgesprach.embeddings")


# --------------------------------------------------------------------------
# Local deterministic embeddings (offline test only)
# --------------------------------------------------------------------------
def _local_embed(text: str, dim: int = conf.LOCAL_EMBEDDING_DIM) -> List[float]:
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# --------------------------------------------------------------------------
# Gemini embeddings
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client(api_key=conf.GEMINI_API_KEY)


def _gemini_embed(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    client = _gemini_client()
    resp = client.models.embed_content(
        model=conf.GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config={"task_type": task_type},
    )
    return [e.values for e in resp.embeddings]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents (job postings)."""
    if not texts:
        return []
    if conf.use_gemini_embeddings():
        out: List[List[float]] = []
        total = len(texts)
        batch_size = 100
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            out.extend(_gemini_embed(batch, task_type="RETRIEVAL_DOCUMENT"))
            logger.info("Embedded %d/%d documents (Gemini)", min(i + batch_size, total), total)
        return out
    return [_local_embed(t) for t in texts]


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    if conf.use_gemini_embeddings():
        return _gemini_embed([text], task_type="RETRIEVAL_QUERY")[0]
    return _local_embed(text)


def backend_name() -> str:
    return "gemini" if conf.use_gemini_embeddings() else "local(test)"