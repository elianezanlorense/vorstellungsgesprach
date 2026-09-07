"""Retrieval and LLM evaluation utilities."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from google import genai

from src.models import list_available_chat_models
from src.prompts import ANSWER_PROMPT, JUDGE_PROMPT

if TYPE_CHECKING:
    from src.rag import TechnicalGermanRAG


SCORE_FIELDS = (
    "faithfulness",
    "relevance",
    "language_quality",
    "phrase_quality",
    "overall",
)


def evaluate_retrieval(
    evaluation_queries: list[dict[str, str]],
    model: Any,
    collection: Any,
    output_path: str | Path,
    n_results: int = 3,
) -> list[dict]:
    """Run retrieval queries and save the results."""
    if not evaluation_queries:
        return []
    if collection.count() == 0:
        raise ValueError("The collection is empty.")

    result_limit = min(n_results, collection.count())
    results = []

    for item in evaluation_queries:
        query_vector = model.encode(item["query"], normalize_embeddings=True)
        search_results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=result_limit,
        )
        results.append(
            {
                "query": item["query"],
                "expected_id": item["expected_id"],
                "retrieved_ids": search_results["ids"][0],
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(f"Query: {result['query']}\n")
            file.write(f"Expected: {result['expected_id']}\n")
            file.write(f"Retrieved: {result['retrieved_ids']}\n\n")

    return results


def hit_rate(results: list[dict], k: int) -> float:
    """Calculate Hit Rate at K."""
    if not results:
        return 0.0
    return sum(
        result["expected_id"] in result["retrieved_ids"][:k]
        for result in results
    ) / len(results)


def mean_reciprocal_rank(results: list[dict]) -> float:
    """Calculate Mean Reciprocal Rank."""
    if not results:
        return 0.0

    reciprocal_ranks = []
    for result in results:
        retrieved_ids = result["retrieved_ids"]
        expected_id = result["expected_id"]
        reciprocal_ranks.append(
            1 / (retrieved_ids.index(expected_id) + 1)
            if expected_id in retrieved_ids
            else 0.0
        )
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compare_models_single_query(
    rag: "TechnicalGermanRAG",
    gemini_client: genai.Client,
    evaluation_queries: list[dict[str, str]],
    output_path: str | Path,
    candidate_models: list[str] | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Run one random evaluation query against candidate models."""
    if not evaluation_queries:
        raise ValueError("The evaluation query list is empty.")

    models = candidate_models or list_available_chat_models(gemini_client)
    selected_query = random.choice(evaluation_queries)
    results = rag.compare_models(
        evaluation_queries=[selected_query],
        candidate_models=models,
    )
    _save_detailed_log(results, Path(output_path))
    return selected_query, results


def evaluate_all_models(
    rag: "TechnicalGermanRAG",
    gemini_client: genai.Client,
    evaluation_queries: list[dict[str, str]],
    topics: list[dict[str, Any]],
    output_dir: str | Path,
    judge_model: str,
    candidate_models: list[str] | None = None,
    max_queries: int | None = None,
    max_retries: int = 3,
    request_delay_seconds: float = 1.0,
) -> tuple[list[dict], list[dict]]:
    """Evaluate candidate models, judge their answers, and save a ranking."""
    models = candidate_models or list_available_chat_models(gemini_client)
    queries = evaluation_queries[:max_queries] if max_queries else evaluation_queries
    topics_by_id = {topic["id"]: topic for topic in topics}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "model_evaluation.jsonl"
    summary_path = output_dir / "model_evaluation_summary.csv"

    saved_results = _load_existing_results(results_path)
    total = len(models) * len(queries)
    current = 0

    _save_model_summary(saved_results, models, len(queries), summary_path)

    for model_name in models:
        for item in queries:
            current += 1
            query = item["query"]
            expected_id = item["expected_id"]
            key = (model_name, query)

            if saved_results.get(key, {}).get("status") == "success":
                print(f"[{current}/{total}] Skipped: {model_name}")
                continue

            print(f"[{current}/{total}] Evaluating: {model_name}")
            started_at = time.perf_counter()

            try:
                generated = _call_with_retry(
                    lambda: rag.rag(
                        query=query,
                        model_name=model_name,
                        n_results=1,
                    ),
                    max_retries,
                )
                retrieved_id = generated.get("retrieved_id", "")
                intro = generated.get("intro", "")
                phrases = generated.get("phrases", [])
                answer = "\n".join(
                    [intro, *(f"- {phrase}" for phrase in phrases)]
                ).strip()
                scores = _judge_answer(
                    gemini_client=gemini_client,
                    judge_model=judge_model,
                    query=query,
                    expected_id=expected_id,
                    retrieved_id=retrieved_id,
                    concept=topics_by_id[expected_id],
                    generated_answer=answer,
                    max_retries=max_retries,
                )
                result = {
                    "status": "success",
                    "model": model_name,
                    "query": query,
                    "expected_id": expected_id,
                    "retrieved_id": retrieved_id,
                    "retrieval_correct": retrieved_id == expected_id,
                    "topic": generated.get("topic", ""),
                    "intro": intro,
                    "phrases": phrases,
                    "latency_seconds": round(time.perf_counter() - started_at, 3),
                    "judge_model": judge_model,
                    "judge": scores,
                }
            except Exception as error:
                result = {
                    "status": "error",
                    "model": model_name,
                    "query": query,
                    "expected_id": expected_id,
                    "latency_seconds": round(time.perf_counter() - started_at, 3),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }

            _append_result(results_path, result)
            saved_results[key] = result
            _save_model_summary(saved_results, models, len(queries), summary_path)
            time.sleep(request_delay_seconds)

    summary = _save_model_summary(saved_results, models, len(queries), summary_path)
    return list(saved_results.values()), summary


def _judge_answer(
    gemini_client: genai.Client,
    judge_model: str,
    query: str,
    expected_id: str,
    retrieved_id: str,
    concept: dict,
    generated_answer: str,
    max_retries: int,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        query=query,
        expected_id=expected_id,
        retrieved_id=retrieved_id,
        topic=concept["topic"],
        reference_answer=concept["answer_de"],
        reference_phrases=", ".join(concept["phrases"]),
        generated_answer=generated_answer,
    )
    response = _call_with_retry(
        lambda: gemini_client.models.generate_content(
            model=judge_model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        ),
        max_retries,
    )
    scores = json.loads(response.text)
    for field in SCORE_FIELDS:
        scores[field] = max(1.0, min(5.0, float(scores.get(field, 1))))
    scores["reason"] = str(scores.get("reason", ""))
    return scores


def _call_with_retry(function: Callable[[], Any], max_retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return function()
        except Exception as error:
            last_error = error
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
    if last_error is None:
        raise RuntimeError("The request failed without an exception.")
    raise last_error


def _load_existing_results(path: Path) -> dict[tuple[str, str], dict]:
    results = {}
    if not path.is_file():
        return results
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if result.get("model") and result.get("query"):
                results[(result["model"], result["query"])] = result
    return results


def _append_result(path: Path, result: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


def _save_detailed_log(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for index, result in enumerate(results, start=1):
            file.write(f"[{index}/{len(results)}] Model: {result['model']}\n")
            file.write(f"Query: {result['query']}\n")
            file.write(f"Expected ID: {result.get('expected_id', '')}\n")
            file.write(f"Retrieved ID: {result.get('retrieved_id', '')}\n")
            file.write(f"Topic: {result.get('topic', '')}\n")
            file.write(f"Intro: {result.get('intro', '')}\n")
            file.write("Phrases:\n")
            for phrase in result.get("phrases", []):
                file.write(f"  - {phrase}\n")
            file.write("-" * 80 + "\n")


def _save_model_summary(
    results: dict[tuple[str, str], dict],
    models: list[str],
    query_count: int,
    output_path: Path,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results.values():
        if result.get("model") in models:
            grouped[result["model"]].append(result)

    summary = []
    for model_name in models:
        model_results = grouped.get(model_name, [])
        successful = [r for r in model_results if r.get("status") == "success"]
        count = len(successful)

        def average(field: str) -> float:
            return (
                sum(float(r["judge"][field]) for r in successful) / count
                if count
                else 0.0
            )

        success_rate = count / query_count if query_count else 0.0
        overall = average("overall")
        retrieval_accuracy = (
            sum(bool(r.get("retrieval_correct")) for r in successful) / count
            if count
            else 0.0
        )
        latency = (
            sum(float(r["latency_seconds"]) for r in successful) / count
            if count
            else 0.0
        )
        summary.append(
            {
                "model": model_name,
                "completed": len(model_results),
                "successful": count,
                "errors": len(model_results) - count,
                "success_rate": round(success_rate, 4),
                "retrieval_accuracy": round(retrieval_accuracy, 4),
                "faithfulness": round(average("faithfulness"), 4),
                "relevance": round(average("relevance"), 4),
                "language_quality": round(average("language_quality"), 4),
                "phrase_quality": round(average("phrase_quality"), 4),
                "overall": round(overall, 4),
                "adjusted_score": round(overall * success_rate, 4),
                "average_latency_seconds": round(latency, 4),
            }
        )

    summary.sort(key=lambda item: item["adjusted_score"], reverse=True)
    fieldnames = list(summary[0]) if summary else ["model"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    return summary
