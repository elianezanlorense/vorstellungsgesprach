"""Sanity check: run one random evaluation query against every available
Gemini chat model, and save the results to outputs/query_randon.txt.

Not part of the production pipeline (main.py) — use this to quickly check
if any candidate model is failing (e.g. a preview model returning a 503)
before running the full comparison in evaluate_models.py.

Usage (from the project root):
    python notebooks/sanity_check.py
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


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluation_queries
from src.load_store_data import load_data
from src.models import list_available_chat_models
from src.rag import TechnicalGermanRAG


TOPICS_PATH = "../data/raw/topics.json"
CHROMA_PATH = "../data/processed/chroma_db"

COLLECTION_NAME = "concepts_de"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATION_MODEL = "models/gemini-flash-lite-latest"

ANSWER_PROMPT = """
Du hilfst einer Person dabei, technisches Deutsch zu lernen.

Basierend auf dem folgenden Konzept, beantworte die Nutzerfrage NICHT mit
einem langen Fließtext. Gib stattdessen zurück:
1. Eine sehr kurze Einleitung (maximal ein Satz)
2. Die wichtigsten Ausdrücke aus dem Konzept, die man sich merken sollte

Nutzerfrage: {query}

Konzept:
Thema: {topic}
Frage: {question_de}
Antwort: {answer_de}
Wichtige Ausdrücke: {phrases}

Antworte ausschließlich mit einem gültigen JSON-Objekt:
{{"intro": "...", "phrases": ["...", "...", "..."]}}
""".strip()

client_gemini = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
def main() -> None:
    load_dotenv("../.env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in the environment.")

    if not os.path.isdir(CHROMA_PATH):
        raise RuntimeError(
            f"ChromaDB directory not found: {CHROMA_PATH}. "
            "Run 'python main.py pipeline' first."
        )

    topics = load_data(TOPICS_PATH)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' is empty. "
            "Run 'python main.py pipeline' first."
        )

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

    query_randon = rag.compare_models(
        evaluation_queries=[selected_query],
        candidate_models=available_models,
    )

    with open("../outputs/query_randon.txt", "w", encoding="utf-8") as f:
        for i, result in enumerate(query_randon, start=1):
            f.write(f"\n[{i}/{len(query_randon)}] Modell: {result['model']}\n")
            f.write(f"Frage: {result['query']}\n")
            f.write(f"Thema erkannt: {result['topic']}\n")
            f.write(f"Intro: {result['intro']}\n")
            f.write("Phrasen:\n")
            for phrase in result["phrases"]:
                f.write(f"  • {phrase}\n")
            f.write("-" * 80 + "\n")

    print(f"Selected query: {selected_query['query']}")
    print("Results saved to: ../outputs/query_randon.txt")


if __name__ == "__main__":
    main()