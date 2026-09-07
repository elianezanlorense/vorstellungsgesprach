"""Evaluate vector retrieval with all predefined evaluation queries.

Usage from the project root:
    uv run python scripts/evaluation_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_retrieval, hit_rate, mean_reciprocal_rank
from src.evaluation_queries import evaluation_queries

CHROMA_PATH = "../data/processed/chroma_db"
COLLECTION_NAME = "concepts_de"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Nome curto do modelo, usado no nome do arquivo de saída para não
# sobrescrever relatórios de outras abordagens (ex.: TF-IDF, outro embedding).
EMBEDDING_MODEL_SLUG = EMBEDDING_MODEL.split("/")[-1]
OUTPUT_PATH = f"../outputs/evaluation_queries_{EMBEDDING_MODEL_SLUG}.txt"


def main() -> None:
    chroma_path = (SCRIPT_DIR / CHROMA_PATH).resolve()
    output_path = (SCRIPT_DIR / OUTPUT_PATH).resolve()

    if not chroma_path.is_dir():
        raise RuntimeError(f"ChromaDB directory not found: {chroma_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    available_collections = [
        collection.name for collection in chroma_client.list_collections()
    ]

    if COLLECTION_NAME not in available_collections:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' was not found. "
            f"Available collections: {available_collections}"
        )

    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' is empty.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    # `evaluate_retrieval` grava um arquivo interno simples (query/expected/
    # retrieved) que não usamos diretamente — mandamos para um path
    # descartável e escrevemos o relatório completo abaixo, uma única vez.
    raw_output_path = output_path.with_suffix(".raw.txt")

    retrieval_results = evaluate_retrieval(
        evaluation_queries=evaluation_queries,
        model=model,
        collection=collection,
        output_path=raw_output_path,
    )

    hit_rate_at_1 = hit_rate(retrieval_results, k=1)
    hit_rate_at_3 = hit_rate(retrieval_results, k=3)
    mrr = mean_reciprocal_rank(retrieval_results)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("VECTOR RETRIEVAL EVALUATION\n")
        file.write("=" * 80 + "\n")
        file.write(f"Embedding model: {EMBEDDING_MODEL}\n")
        file.write(f"Documents indexed: {collection.count()}\n")
        file.write(f"Evaluation queries: {len(retrieval_results)}\n")
        file.write(f"Hit Rate@1: {hit_rate_at_1:.3f}\n")
        file.write(f"Hit Rate@3: {hit_rate_at_3:.3f}\n")
        file.write(f"MRR: {mrr:.3f}\n")
        file.write("=" * 80 + "\n")

        for index, result in enumerate(retrieval_results, start=1):
            first_id = result["retrieved_ids"][0]
            correct_at_1 = first_id == result["expected_id"]
            correct_at_3 = result["expected_id"] in result["retrieved_ids"][:3]

            file.write(f"\n[{index}/{len(retrieval_results)}]\n")
            file.write(f"Query: {result['query']}\n")
            file.write(f"Expected: {result['expected_id']}\n")
            file.write(f"Retrieved: {result['retrieved_ids']}\n")
            file.write(f"First result: {first_id}\n")
            file.write(f"Correct@1: {correct_at_1}\n")
            file.write(f"Correct@3: {correct_at_3}\n")
            file.write("-" * 80 + "\n")

    # Remove o arquivo intermediário descartável.
    raw_output_path.unlink(missing_ok=True)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Documents indexed: {collection.count()}")
    print(f"Evaluation queries: {len(retrieval_results)}")
    print(f"Hit Rate@1: {hit_rate_at_1:.3f}")
    print(f"Hit Rate@3: {hit_rate_at_3:.3f}")
    print(f"MRR: {mrr:.3f}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()