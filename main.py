"""Ingestion and retrieval evaluation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from build_documents import build_documents
from embeddings import create_embeddings
from evaluation import (
    evaluate_retrieval,
    hit_rate,
    mean_reciprocal_rank,
)
from evaluation_queries import evaluation_queries
from load_store_data import (
    load_data,
    store_documents,
)


TOPICS_PATH = "data/raw/topics.json"
CHROMA_PATH = "data/processed/chroma_db"
EVALUATION_OUTPUT_PATH = "outputs/evaluation_queries.txt"

COLLECTION_NAME = "concepts_de"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def run_pipeline(
    data_path: str | Path = TOPICS_PATH,
    chroma_path: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    evaluation_output_path: str | Path = EVALUATION_OUTPUT_PATH,
    embedding_model_name: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Run ingestion, indexing and retrieval evaluation."""
    data_path = PROJECT_ROOT / data_path
    chroma_path = PROJECT_ROOT / chroma_path
    evaluation_output_path = (
        PROJECT_ROOT / evaluation_output_path
    )

    topics = load_data(data_path)

    documents = build_documents(topics)

    model, vectors = create_embeddings(
        documents=documents,
        model_name=embedding_model_name,
    )

    collection = store_documents(
        documents=documents,
        vectors=vectors,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )

    retrieval_results = evaluate_retrieval(
        evaluation_queries=evaluation_queries,
        model=model,
        collection=collection,
        output_path=evaluation_output_path,
        n_results=3,
    )

    metrics = {
        "hit_rate_at_1": hit_rate(
            retrieval_results,
            k=1,
        ),
        "hit_rate_at_3": hit_rate(
            retrieval_results,
            k=3,
        ),
        "mrr": mean_reciprocal_rank(
            retrieval_results
        ),
    }

    return {
        "topics": topics,
        "documents": documents,
        "model": model,
        "vectors": vectors,
        "collection": collection,
        "retrieval_results": retrieval_results,
        "metrics": metrics,
    }


def print_pipeline_results(
    pipeline: dict[str, Any],
) -> None:
    """Print the ingestion and evaluation results."""
    collection = pipeline["collection"]
    metrics = pipeline["metrics"]

    print(
        f"Documents indexed: {collection.count()}"
    )

    print(
        f"Collection IDs: "
        f"{collection.get()['ids']}"
    )

    print(
        f"Hit Rate@1: "
        f"{metrics['hit_rate_at_1']:.3f}"
    )

    print(
        f"Hit Rate@3: "
        f"{metrics['hit_rate_at_3']:.3f}"
    )

    print(
        f"MRR: {metrics['mrr']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Technical German RAG pipeline."
        )
    )

    parser.add_argument(
        "command",
        choices=["pipeline"],
        help="Command to execute.",
    )

    args = parser.parse_args()

    if args.command == "pipeline":
        pipeline = run_pipeline()
        print_pipeline_results(pipeline)


if __name__ == "__main__":
    main()