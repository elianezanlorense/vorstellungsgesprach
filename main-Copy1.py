"""Ingestion and evaluation pipeline for the Technical German RAG project."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent

TOPICS_PATH = "../data/raw/topics.json"
CHROMA_PATH = '../data/processed/chroma_db'
OUTPUT_DIR = '../outputs'

from build_documents import build_documents
from evaluation import evaluate_retrieval,evaluation_queries,hit_rate,mean_reciprocal_rank,compare_models_all_queries
from load_store_data import load_data
from models import list_available_chat_models
from rag import TechnicalGermanRAG




COLLECTION_NAME = "concepts_de"
EMBEDDING_MODEL = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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


def create_gemini_client() -> genai.Client:
    """Create an authenticated Gemini client."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in the environment.")

    return genai.Client(api_key=api_key)


def create_embedding_model() -> SentenceTransformer:
    """Load the multilingual local embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def open_collection():
    """Open or create the persistent ChromaDB collection."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def ingest_topics(
    topics: list[dict],
    embedding_model: SentenceTransformer,
    collection,
) -> int:
    """Build, embed, and upsert all technical concepts."""
    documents = build_documents(topics)
    texts = [document["text"] for document in documents]
    vectors = embedding_model.encode(texts, normalize_embeddings=True)

    collection.upsert(
        ids=[document["id"] for document in documents],
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=[document["metadata"] for document in documents],
    )

    return collection.count()


def run_retrieval_evaluation(
    embedding_model: SentenceTransformer,
    collection,
) -> dict[str, float]:
    """Run retrieval evaluation and save individual query results."""
    results = evaluate_retrieval(
        evaluation_queries=evaluation_queries,
        model=embedding_model,
        collection=collection,
        output_path=OUTPUT_DIR / "evaluation_queries.txt",
    )

    metrics = {
        "hit_rate_at_1": hit_rate(results, k=1),
        "hit_rate_at_3": hit_rate(results, k=3),
        "mrr": mean_reciprocal_rank(results),
    }

    metrics_path = OUTPUT_DIR / "retrieval_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as file:
        file.write(f"Embedding model: {EMBEDDING_MODEL}\n")
        file.write(f"Hit Rate@1: {metrics['hit_rate_at_1']:.3f}\n")
        file.write(f"Hit Rate@3: {metrics['hit_rate_at_3']:.3f}\n")
        file.write(f"MRR: {metrics['mrr']:.3f}\n")

    return metrics


def create_rag(
    topics: list[dict],
    embedding_model: SentenceTransformer,
    collection,
    gemini_client: genai.Client,
) -> TechnicalGermanRAG:
    """Create the Technical German RAG application."""
    return TechnicalGermanRAG(
        collection=collection,
        embed_model=embedding_model,
        llm_client=gemini_client,
        topics=topics,
        model=GENERATION_MODEL,
        answer_prompt=ANSWER_PROMPT,
    )


def save_model_results(results: list[dict], output_path: Path) -> None:
    """Save model responses without printing them to the terminal."""
    with output_path.open("w", encoding="utf-8") as file:
        for index, result in enumerate(results, start=1):
            file.write(f"[{index}/{len(results)}] Modell: {result['model']}\n")
            file.write(f"Frage: {result['query']}\n")
            file.write(f"Expected ID: {result.get('expected_id', '')}\n")
            file.write(f"Retrieved ID: {result.get('retrieved_id', '')}\n")
            file.write(
                f"Retrieval correct: {result.get('retrieval_correct', '')}\n"
            )
            file.write(f"Thema erkannt: {result['topic']}\n")
            file.write(f"Intro: {result['intro']}\n")
            file.write("Phrasen:\n")

            for phrase in result["phrases"]:
                file.write(f"  - {phrase}\n")

            file.write("-" * 80 + "\n")


def compare_models_for_query(
    rag: TechnicalGermanRAG,
    gemini_client: genai.Client,
    query_item: dict[str, str],
    output_path: Path,
) -> list[dict]:
    """Run one evaluation query against every available Gemini chat model.

    Used as a quick sanity check (e.g. to see if a preview model like
    gemini-3-flash-preview returns a 503) before running the full
    comparison with `models-all`.
    """
    candidate_models = list_available_chat_models(gemini_client)
    results = rag.compare_models(
        evaluation_queries=[query_item],
        candidate_models=candidate_models,
    )
    save_model_results(results, output_path)
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=("ingest", "retrieval", "models", "models-all", "all"),
        help="Pipeline step to execute.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=None,
        help=(
            "Zero-based evaluation-query index for model comparison "
            "(used by the 'models' step). When omitted, one query is "
            "selected randomly."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help=(
            "Limit the number of evaluation_queries used by the "
            "'models-all' step. When omitted, all queries are used."
        ),
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=None,
        dest="exclude_models",
        help=(
            "Chat model name to exclude from the 'models-all' step "
            "(e.g. an unstable preview model). Can be repeated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Execute the selected pipeline step."""
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topics = load_data(TOPICS_PATH)
    embedding_model = create_embedding_model()
    collection = open_collection()

    if args.step in {"ingest", "all"}:
        count = ingest_topics(topics, embedding_model, collection)
        print(f"Indexed concepts: {count}")

    if args.step in {"retrieval", "all"}:
        metrics = run_retrieval_evaluation(embedding_model, collection)
        print(f"Hit Rate@1: {metrics['hit_rate_at_1']:.3f}")
        print(f"Hit Rate@3: {metrics['hit_rate_at_3']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")

    if args.step in {"models", "all"}:
        gemini_client = create_gemini_client()
        rag = create_rag(topics, embedding_model, collection, gemini_client)

        if args.query_index is None:
            selected_query = random.choice(evaluation_queries)
        else:
            selected_query = evaluation_queries[args.query_index]

        compare_models_for_query(
            rag=rag,
            gemini_client=gemini_client,
            query_item=selected_query,
            output_path=OUTPUT_DIR / "model_comparison.txt",
        )
        print(f"Selected query: {selected_query['query']}")
        print(f"Expected ID: {selected_query['expected_id']}")
        print(f"Results saved to: {OUTPUT_DIR / 'model_comparison.txt'}")

    if args.step == "models-all":
        gemini_client = create_gemini_client()
        rag = create_rag(topics, embedding_model, collection, gemini_client)

        exclude_models = (
            set(args.exclude_models) if args.exclude_models else None
        )

        compare_models_all_queries(
            rag=rag,
            gemini_client=gemini_client,
            evaluation_queries=evaluation_queries,
            output_dir=OUTPUT_DIR,
            max_queries=args.max_queries,
            exclude_models=exclude_models,
        )
        print(
            f"Results saved to: "
            f"{OUTPUT_DIR / 'model_comparison_all_queries.txt'}"
        )
        print(f"Ranking saved to: {OUTPUT_DIR / 'model_ranking.txt'}")


if __name__ == "__main__":
    main()