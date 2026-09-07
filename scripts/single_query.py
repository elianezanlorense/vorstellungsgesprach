"""Run one random evaluation query against all available Gemini models.

Usage from the project root:
    uv run python scripts/single_query.py
"""

from __future__ import annotations

import os
import random
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

from src.evaluation import evaluation_queries
from src.load_store_data import load_data
from src.models import list_available_chat_models
from src.rag import TechnicalGermanRAG
from src.prompts import ANSWER_PROMPT


TOPICS_PATH = "../data/raw/topics.json"
CHROMA_PATH = "../data/processed/chroma_db"
OUTPUT_PATH = "../outputs/query_random.txt"

COLLECTION_NAME = "concepts_de"
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
GENERATION_MODEL = "models/gemini-flash-lite-latest"

def main() -> None:
    topics_path = (SCRIPT_DIR / TOPICS_PATH).resolve()
    chroma_path = (SCRIPT_DIR / CHROMA_PATH).resolve()
    output_path = (SCRIPT_DIR / OUTPUT_PATH).resolve()
    env_path = (SCRIPT_DIR / "../.env").resolve()

    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY was not found in {env_path}")

    if not topics_path.is_file():
        raise RuntimeError(f"Topics file not found: {topics_path}")

    if not chroma_path.is_dir():
        raise RuntimeError(f"ChromaDB directory not found: {chroma_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    topics = load_data(topics_path)

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' is empty.")

    gemini_client = genai.Client(api_key=api_key)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    rag = TechnicalGermanRAG(
        collection=collection,
        embed_model=embedding_model,
        llm_client=gemini_client,
        topics=topics,
        model=GENERATION_MODEL,
        answer_prompt=ANSWER_PROMPT,
    )

    selected_query = random.choice(evaluation_queries)
    available_models = list_available_chat_models(gemini_client)

    query_random = rag.compare_models(
        evaluation_queries=[selected_query],
        candidate_models=available_models,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for index, result in enumerate(query_random, start=1):
            file.write(f"\n[{index}/{len(query_random)}] Modell: {result['model']}\n")
            file.write(f"Frage: {result['query']}\n")
            file.write(f"Thema erkannt: {result['topic']}\n")
            file.write(f"Intro: {result['intro']}\n")
            file.write("Phrasen:\n")
            for phrase in result["phrases"]:
                file.write(f"  • {phrase}\n")
            file.write("-" * 80 + "\n")

    print(f"Selected query: {selected_query['query']}")
    print(f"Expected ID: {selected_query['expected_id']}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()