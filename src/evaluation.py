from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google import genai

from models import list_available_chat_models

if TYPE_CHECKING:
    from rag import TechnicalGermanRAG


def evaluate_retrieval(
    evaluation_queries: list[dict[str, str]],
    model: Any,
    collection: Any,
    output_path: str | Path,
    n_results: int = 3,
) -> list[dict]:
    """Run retrieval queries and save the results to a text file."""
    retrieval_results = []
    result_limit = min(n_results, collection.count())
    for item in evaluation_queries:
        query_vector = model.encode(
            item["query"],
            normalize_embeddings=True,
        )
        search_results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=result_limit,
        )
        retrieved_ids = search_results["ids"][0]
        retrieval_results.append({
            "query": item["query"],
            "expected_id": item["expected_id"],
            "retrieved_ids": retrieved_ids,
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in retrieval_results:
            file.write(f"Query: {result['query']}\n")
            file.write(f"Expected: {result['expected_id']}\n")
            file.write(f"Retrieved: {result['retrieved_ids']}\n")
            file.write("\n")
    return retrieval_results


def hit_rate(results: list[dict], k: int) -> float:
    hits = []
    for result in results:
        top_k = result["retrieved_ids"][:k]
        hit = result["expected_id"] in top_k
        hits.append(hit)
    return sum(hits) / len(hits)


def mean_reciprocal_rank(results: list[dict]) -> float:
    reciprocal_ranks = []
    for result in results:
        retrieved_ids = result["retrieved_ids"]
        expected_id = result["expected_id"]
        if expected_id in retrieved_ids:
            rank = retrieved_ids.index(expected_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ---------------------------------------------------------------------------
# LLM evaluation: compara múltiplos modelos Gemini contra as evaluation_queries
# ---------------------------------------------------------------------------


def compare_models_all_queries(
    rag: "TechnicalGermanRAG",
    gemini_client: genai.Client,
    evaluation_queries: list[dict[str, str]],
    output_dir: Path,
    max_queries: int | None = None,
    exclude_models: set[str] | None = None,
) -> list[dict]:
    """Roda todas (ou um subconjunto) das evaluation_queries contra
    todos os modelos candidatos e salva log detalhado + ranking agregado.

    Args:
        rag: instância de TechnicalGermanRAG já configurada.
        gemini_client: cliente Gemini autenticado.
        evaluation_queries: lista completa de queries de avaliação.
        output_dir: pasta onde salvar os outputs (ex.: OUTPUT_DIR).
        max_queries: se definido, usa só as N primeiras queries
            (útil pra não estourar rate limit / custo).
        exclude_models: nomes de modelos a excluir manualmente
            (ex.: previews instáveis que já sabemos que dão 503).

    Returns:
        Lista de resultados individuais (uma entrada por query x modelo).
    """
    candidate_models = list_available_chat_models(gemini_client)

    if exclude_models:
        candidate_models = [
            model for model in candidate_models if model not in exclude_models
        ]

    queries = evaluation_queries
    if max_queries is not None:
        queries = queries[:max_queries]

    all_results = rag.compare_models(
        evaluation_queries=queries,
        candidate_models=candidate_models,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "model_comparison_all_queries.txt"
    _save_detailed_log(all_results, log_path)

    summary = _summarize_by_model(all_results)
    summary_path = output_dir / "model_ranking.txt"
    _save_ranking(summary, len(queries), summary_path)

    return all_results


def _save_detailed_log(results: list[dict], output_path: Path) -> None:
    """Salva o log detalhado de cada (query, modelo) em arquivo."""
    with output_path.open("w", encoding="utf-8") as file:
        for index, result in enumerate(results, start=1):
            file.write(f"[{index}/{len(results)}] Modell: {result['model']}\n")
            file.write(f"Frage: {result['query']}\n")
            file.write(f"Expected ID: {result.get('expected_id', '')}\n")
            file.write(f"Retrieved ID: {result.get('retrieved_id', '')}\n")
            file.write(
                f"Retrieval correct: {result.get('retrieval_correct', '')}\n"
            )
            file.write(f"Thema erkannt: {result.get('topic', '')}\n")
            file.write(f"Intro: {result.get('intro', '')}\n")
            file.write("Phrasen:\n")

            for phrase in result.get("phrases", []):
                file.write(f"  - {phrase}\n")

            file.write("-" * 80 + "\n")


def _summarize_by_model(results: list[dict]) -> dict[str, dict[str, Any]]:
    """Agrega, por modelo: total de queries, acertos de retrieval e
    falhas (quando 'retrieval_correct' vier vazio, indicando erro)."""
    summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "failed": 0}
    )

    for result in results:
        model = result["model"]
        summary[model]["total"] += 1

        retrieval_correct = result.get("retrieval_correct", "")
        if retrieval_correct == "":
            summary[model]["failed"] += 1
        elif retrieval_correct:
            summary[model]["correct"] += 1

    return dict(summary)


def _save_ranking(
    summary: dict[str, dict[str, Any]],
    total_queries: int,
    output_path: Path,
) -> None:
    """Salva o ranking final dos modelos, ordenado por hit rate de
    retrieval (descendente), com falhas reportadas separadamente."""
    ranked = sorted(
        summary.items(),
        key=lambda item: (
            item[1]["correct"] / item[1]["total"] if item[1]["total"] else 0
        ),
        reverse=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"Total de evaluation_queries usadas: {total_queries}\n")
        file.write("=" * 80 + "\n")

        for model, stats in ranked:
            total = stats["total"]
            correct = stats["correct"]
            failed = stats["failed"]
            hit_rate_value = correct / total if total else 0.0

            file.write(f"Modell: {model}\n")
            file.write(f"  Queries avaliadas: {total}\n")
            file.write(
                f"  Retrieval correto: {correct}/{total} "
                f"({hit_rate_value:.1%})\n"
            )
            file.write(f"  Falhas (erro/sem resposta): {failed}\n")
            file.write("-" * 80 + "\n")