"""Evaluate all available Gemini models with all evaluation queries.

Usage from the project root:
    uv run python scripts/evaluate_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_all_models
from src.evaluation_queries import evaluation_queries
from src.load_store_data import load_data
from src.prompts import ANSWER_PROMPT
from src.rag import TechnicalGermanRAG


TOPICS_PATH = "../data/raw/topics.json"
CHROMA_PATH = "../data/processed/chroma_db"
OUTPUT_DIR = "../outputs"

COLLECTION_NAME = "concepts_de"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


GENERATION_MODEL = "models/gemini-flash-lite-latest"
JUDGE_MODEL = "models/gemini-flash-latest"

CANDIDATE_MODELS = [
    "models/gemini-flash-lite-latest",
    "models/gemini-3.6-flash",
]

def main() -> None:
    topics_path = (SCRIPT_DIR / TOPICS_PATH).resolve()
    chroma_path = (SCRIPT_DIR / CHROMA_PATH).resolve()
    output_dir = (SCRIPT_DIR / OUTPUT_DIR).resolve()
    env_path = (SCRIPT_DIR / "../.env").resolve()

    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY was not found in {env_path}")
    if not topics_path.is_file():
        raise RuntimeError(f"Topics file not found: {topics_path}")
    if not chroma_path.is_dir():
        raise RuntimeError(f"ChromaDB directory not found: {chroma_path}")

    topics = load_data(topics_path)
    gemini_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    rag = TechnicalGermanRAG(
        collection=collection,
        embed_model=embedding_model,
        llm_client=gemini_client,
        topics=topics,
        model=GENERATION_MODEL,
        answer_prompt=ANSWER_PROMPT,
    )

    _, summary = evaluate_all_models(
        rag=rag,
        gemini_client=gemini_client,
        evaluation_queries=evaluation_queries,
        topics=topics,
        output_dir=output_dir,
        judge_model=JUDGE_MODEL,
    )

    print("\nModel ranking:")
    for position, result in enumerate(summary, start=1):
        print(
            f"{position}. {result['model']} | "
            f"adjusted={result['adjusted_score']:.3f} | "
            f"overall={result['overall']:.3f} | "
            f"success={result['success_rate']:.3f} | "
            f"latency={result['average_latency_seconds']:.3f}s"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
