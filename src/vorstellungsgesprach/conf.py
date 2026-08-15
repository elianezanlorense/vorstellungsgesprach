"""Configuration for the Vorstellungsgesprach RAG app.

Reads settings from environment variables so nothing sensitive is hardcoded.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # carrega automaticamente as variáveis do arquivo .env, se existir

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
#GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "models/gemini-3.7-flash")

# Dimension used by the local deterministic fallback embedder (offline tests).
LOCAL_EMBEDDING_DIM = int(os.environ.get("LOCAL_EMBEDDING_DIM", "256"))


def use_gemini_embeddings() -> bool:
    """Whether to use the real Gemini backend (True) or the local fallback (False)."""
    return bool(GEMINI_API_KEY)