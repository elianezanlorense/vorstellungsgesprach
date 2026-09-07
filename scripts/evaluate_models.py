"""Evaluate every available Gemini model with every evaluation query.

This script:

1. Runs every evaluation query against every available Gemini model.
2. Evaluates each generated answer with a fixed judge model.
3. Saves each result immediately to a JSONL file.
4. Supports resuming an interrupted evaluation.
5. Produces a CSV summary ranked by adjusted score.

Usage from the project root:
    uv run python scripts/evaluate_models.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

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
from src.prompts import ANSWER_PROMPT, JUDGE_PROMPT


TOPICS_PATH = "../data/raw/topics.json"
CHROMA_PATH = "../data/processed/chroma_db"
RESULTS_PATH = "../outputs/model_evaluation.jsonl"
SUMMARY_PATH = "../outputs/model_evaluation_summary.csv"

COLLECTION_NAME = "concepts_de"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_GENERATION_MODEL = "models/gemini-flash-lite-latest"
JUDGE_MODEL = "models/gemini-flash-latest"

MAX_RETRIES = 3
MAX_RETRIES_UNAVAILABLE = 1
REQUEST_DELAY_SECONDS = 1.0






def judge_answer(
    gemini_client,
    query: str,
    expected_id: str,
    retrieved_id: str,
    concept: dict,
    generated_answer: str,
) -> dict:
    """Evaluate one generated answer with a fixed judge model."""

    prompt = JUDGE_PROMPT.format(
        query=query,
        expected_id=expected_id,
        retrieved_id=retrieved_id,
        topic=concept["topic"],
        reference_answer=concept["answer_de"],
        reference_phrases=", ".join(concept["phrases"]),
        generated_answer=generated_answer,
    )

    response = call_with_retry(
        lambda: gemini_client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            },
        )
    )

    scores = json.loads(response.text)

    score_fields = [
        "faithfulness",
        "relevance",
        "language_quality",
        "phrase_quality",
        "overall",
    ]

    for field in score_fields:
        value = float(scores.get(field, 1))
        scores[field] = max(1.0, min(5.0, value))

    scores["reason"] = str(
        scores.get("reason", "")
    )

    return scores





def save_result(
    results_path: Path,
    result: dict,
) -> None:
    """Append one evaluation result immediately."""

    with results_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


def create_summary(
    results: dict[tuple[str, str], dict],
    candidate_models: list[str],
    query_count: int,
    summary_path: Path,
) -> list[dict]:
    """Aggregate evaluation metrics for every model."""

    grouped_results = defaultdict(list)

    for result in results.values():
        grouped_results[result["model"]].append(
            result
        )

    summary = []

    for model_name in candidate_models:
        model_results = grouped_results.get(
            model_name,
            [],
        )

        successful_results = [
            result
            for result in model_results
            if result.get("status") == "success"
        ]

        success_count = len(successful_results)
        success_rate = (
            success_count / query_count
            if query_count
            else 0.0
        )

        retrieval_correct_count = sum(
            result.get(
                "retrieval_correct",
                False,
            )
            for result in successful_results
        )

        retrieval_accuracy = (
            retrieval_correct_count / success_count
            if success_count
            else 0.0
        )

        def average(field: str) -> float:
            if not successful_results:
                return 0.0

            return sum(
                float(result["judge"][field])
                for result in successful_results
            ) / len(successful_results)

        average_faithfulness = average(
            "faithfulness"
        )

        average_relevance = average(
            "relevance"
        )

        average_language_quality = average(
            "language_quality"
        )

        average_phrase_quality = average(
            "phrase_quality"
        )

        average_overall = average(
            "overall"
        )

        average_latency = (
            sum(
                float(result["latency_seconds"])
                for result in successful_results
            ) / success_count
            if success_count
            else 0.0
        )

        adjusted_score = (
            average_overall * success_rate
        )

        summary.append(
            {
                "model": model_name,
                "completed": len(model_results),
                "successful": success_count,
                "success_rate": success_rate,
                "retrieval_accuracy": retrieval_accuracy,
                "faithfulness": average_faithfulness,
                "relevance": average_relevance,
                "language_quality": average_language_quality,
                "phrase_quality": average_phrase_quality,
                "overall": average_overall,
                "adjusted_score": adjusted_score,
                "average_latency_seconds": average_latency,
            }
        )

    summary.sort(
        key=lambda item: item["adjusted_score"],
        reverse=True,
    )

    fieldnames = [
        "model",
        "completed",
        "successful",
        "success_rate",
        "retrieval_accuracy",
        "faithfulness",
        "relevance",
        "language_quality",
        "phrase_quality",
        "overall",
        "adjusted_score",
        "average_latency_seconds",
    ]

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary)

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every available Gemini model against every "
            "evaluation query."
        )
    )

    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        metavar="MODEL_NAME",
        help=(
            "Model name to exclude from the run (as returned by "
            "list_available_chat_models). Can be passed multiple "
            "times, or as a single comma-separated value, e.g. "
            "--exclude-model models/gemma-4-31b-it "
            "--exclude-model models/gemini-3.6-flash, or "
            "--exclude-model models/gemma-4-31b-it,models/gemini-3.6-flash"
        ),
    )

    args = parser.parse_args()

    excluded_models: set[str] = set()

    for raw_value in args.exclude_model:
        for name in raw_value.split(","):
            name = name.strip()

            if name:
                excluded_models.add(name)

    args.excluded_models = excluded_models

    return args


def main() -> None:
    args = parse_args()

    topics_path = (SCRIPT_DIR / TOPICS_PATH).resolve()
    chroma_path = (SCRIPT_DIR / CHROMA_PATH).resolve()
    results_path = (SCRIPT_DIR / RESULTS_PATH).resolve()
    summary_path = (SCRIPT_DIR / SUMMARY_PATH).resolve()
    env_path = (SCRIPT_DIR / "../.env").resolve()

    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"GEMINI_API_KEY was not found in {env_path}"
        )

    if not topics_path.is_file():
        raise RuntimeError(
            f"Topics file not found: {topics_path}"
        )

    if not chroma_path.is_dir():
        raise RuntimeError(
            f"ChromaDB directory not found: {chroma_path}"
        )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    topics = load_data(topics_path)

    topics_by_id = {
        topic["id"]: topic
        for topic in topics
    }

    gemini_client = genai.Client(
        api_key=api_key
    )

    chroma_client = chromadb.PersistentClient(
        path=str(chroma_path)
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

    rag = TechnicalGermanRAG(
        collection=collection,
        embed_model=embedding_model,
        llm_client=gemini_client,
        topics=topics,
        model=DEFAULT_GENERATION_MODEL,
        answer_prompt=ANSWER_PROMPT,
    )

    all_available_models = list_available_chat_models(
        gemini_client
    )

    unknown_excluded_models = (
        args.excluded_models - set(all_available_models)
    )

    if unknown_excluded_models:
        print(
            "Warning: the following --exclude-model values do not "
            "match any available model and will be ignored: "
            + ", ".join(sorted(unknown_excluded_models))
        )

    available_models = [
        model_name
        for model_name in all_available_models
        if model_name not in args.excluded_models
    ]

    if args.excluded_models:
        print(
            f"Excluding {len(args.excluded_models & set(all_available_models))} "
            f"model(s): "
            + ", ".join(
                sorted(
                    args.excluded_models
                    & set(all_available_models)
                )
            )
        )

    existing_results = load_existing_results(
        results_path
    )

    total_evaluations = (
        len(available_models)
        * len(evaluation_queries)
    )

    print(
        f"Models: {len(available_models)}"
    )

    print(
        f"Queries: {len(evaluation_queries)}"
    )

    print(
        f"Total evaluations: {total_evaluations}"
    )

    completed = 0

    for model_name in available_models:
        for evaluation_item in evaluation_queries:
            completed += 1

            query = evaluation_item["query"]
            expected_id = evaluation_item["expected_id"]
            result_key = (model_name, query)

            previous_result = existing_results.get(
                result_key
            )

            if (
                previous_result
                and previous_result.get("status") == "success"
            ):
                print(
                    f"[{completed}/{total_evaluations}] "
                    f"Skipped: {model_name}"
                )
                continue

            print(
                f"[{completed}/{total_evaluations}] "
                f"Evaluating: {model_name}"
            )

            started_at = time.perf_counter()

            try:
                generated = call_with_retry(
                    lambda: rag.rag(
                        query=query,
                        model_name=model_name,
                        n_results=1,
                    )
                )

                latency_seconds = (
                    time.perf_counter()
                    - started_at
                )

                intro = generated.get(
                    "intro",
                    "",
                )

                phrases = generated.get(
                    "phrases",
                    [],
                )

                generated_answer = (
                    intro
                    + "\n"
                    + "\n".join(
                        f"- {phrase}"
                        for phrase in phrases
                    )
                ).strip()

                retrieved_id = generated.get(
                    "retrieved_id",
                    "",
                )

                reference_concept = topics_by_id[
                    expected_id
                ]

                judge_scores = judge_answer(
                    gemini_client=gemini_client,
                    query=query,
                    expected_id=expected_id,
                    retrieved_id=retrieved_id,
                    concept=reference_concept,
                    generated_answer=generated_answer,
                )

                result = {
                    "status": "success",
                    "model": model_name,
                    "query": query,
                    "expected_id": expected_id,
                    "retrieved_id": retrieved_id,
                    "retrieval_correct": (
                        retrieved_id == expected_id
                    ),
                    "topic": generated.get(
                        "topic",
                        "",
                    ),
                    "intro": intro,
                    "phrases": phrases,
                    "latency_seconds": round(
                        latency_seconds,
                        3,
                    ),
                    "judge_model": JUDGE_MODEL,
                    "judge": judge_scores,
                }

            except Exception as error:
                latency_seconds = (
                    time.perf_counter()
                    - started_at
                )

                result = {
                    "status": "error",
                    "model": model_name,
                    "query": query,
                    "expected_id": expected_id,
                    "latency_seconds": round(
                        latency_seconds,
                        3,
                    ),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }

            save_result(
                results_path,
                result,
            )

            existing_results[
                result_key
            ] = result

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    summary = create_summary(
        results=existing_results,
        candidate_models=available_models,
        query_count=len(evaluation_queries),
        summary_path=summary_path,
    )

    print("\nModel ranking:")

    for position, model_result in enumerate(
        summary,
        start=1,
    ):
        print(
            f"{position}. "
            f"{model_result['model']} | "
            f"adjusted={model_result['adjusted_score']:.3f} | "
            f"overall={model_result['overall']:.3f} | "
            f"success={model_result['success_rate']:.3f} | "
            f"latency={model_result['average_latency_seconds']:.3f}s"
        )

    print(
        f"\nDetailed results saved to: {results_path}"
    )

    print(
        f"Summary saved to: {summary_path}"
    )


if __name__ == "__main__":
    main()