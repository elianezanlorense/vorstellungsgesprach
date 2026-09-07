"""Runtime services for the Streamlit application."""

from __future__ import annotations

from pathlib import Path

import chromadb
from google import genai
from sentence_transformers import SentenceTransformer

from src.load_store_data import load_data
from src.prompts import ANSWER_PROMPT
from src.rag import TechnicalGermanRAG


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOPICS_PATH = PROJECT_ROOT / "data" / "raw" / "topics.json"
CHROMA_PATH = PROJECT_ROOT / "data" / "processed" / "chroma_db"

COLLECTION_NAME = "concepts_de"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

GENERATION_MODEL = "models/gemini-flash-lite-latest"


def build_rag_assistant(
    api_key: str,
) -> TechnicalGermanRAG:
    """Load the existing collection and create the RAG assistant."""
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found.")

    if not TOPICS_PATH.is_file():
        raise FileNotFoundError(
            f"Topics file not found: {TOPICS_PATH}"
        )

    if not CHROMA_PATH.is_dir():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {CHROMA_PATH}. "
            "Run the pipeline before starting the application."
        )

    topics = load_data(TOPICS_PATH)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' is empty."
        )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    gemini_client = genai.Client(
        api_key=api_key
    )

    return TechnicalGermanRAG(
        collection=collection,
        embed_model=embedding_model,
        llm_client=gemini_client,
        topics=topics,
        model=GENERATION_MODEL,
        answer_prompt=ANSWER_PROMPT,
    )


def answer_query(
    rag_assistant: TechnicalGermanRAG,
    query: str,
) -> dict:
    """Answer one user query using the existing RAG assistant."""
    return rag_assistant.rag(
        query=query,
        model_name=GENERATION_MODEL,
        n_results=1,
    )