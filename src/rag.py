ANSWER_PROMPT = """Du hilfst einer Person, sich auf ein Data-Science-Vorstellungsgespräch auf Deutsch vorzubereiten.

Basierend auf dem folgenden Konzept, beantworte die Nutzerfrage NICHT mit einem Fließtext. Gib stattdessen zurück:
1. Eine sehr kurze Einleitung (max. 1 Satz)
2. Die wichtigsten Ausdrücke (Phrasen) aus dem Konzept, die man sich zum Lernen merken sollte

Nutzerfrage: {query}

Konzept:
Thema: {topic}
Frage: {question_de}
Antwort: {answer_de}
Wichtige Ausdrücke: {phrases}

Antworte NUR mit einem JSON-Objekt, ohne Markdown-Formatierung:
{{"intro": "...", "phrases": ["...", "...", "..."]}}
"""


import json
from collections import defaultdict
from typing import Any

from google.genai import types


class TechnicalGermanRAG:
    """RAG application for technical German concepts."""

    def __init__(
        self,
        collection,
        embed_model,
        llm_client,
        topics: list[dict],
        model: str,
        answer_prompt: str,
    ):
        self.collection = collection
        self.embed_model = embed_model
        self.llm_client = llm_client
        self.model = model
        self.answer_prompt = answer_prompt

        self.concepts_by_id = {
            concept["id"]: concept
            for concept in topics
        }

    def search(
        self,
        query: str,
        n_results: int = 1,
    ) -> dict:
        """Retrieve concepts from ChromaDB using vector search."""
        query_vector = self.embed_model.encode(
            query,
            normalize_embeddings=True,
        )

        return self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(n_results, self.collection.count()),
        )

    def find_concept_by_id(self, concept_id: str) -> dict:
        """Return the complete original concept by ID."""
        concept = self.concepts_by_id.get(concept_id)

        if concept is None:
            raise ValueError(
                f"Concept ID not found: {concept_id}"
            )

        return concept

    def build_prompt(
        self,
        query: str,
        concept: dict,
    ) -> str:
        """Build the generation prompt using the retrieved concept."""
        return self.answer_prompt.format(
            query=query,
            topic=concept.get("topic", ""),
            question_de=concept.get("question_de", ""),
            answer_de=concept.get("answer_de", ""),
            phrases=", ".join(concept.get("phrases", [])),
        )

    def generate(
        self,
        prompt: str,
        model_name: str | None = None,
    ) -> dict:
        """Generate a structured answer with Gemini."""
        selected_model = model_name or self.model

        response = self.llm_client.models.generate_content(
            model=selected_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        return json.loads(response.text)

    def rag(
        self,
        query: str,
        model_name: str | None = None,
        n_results: int = 1,
    ) -> dict:
        """Run vector retrieval followed by LLM generation."""
        selected_model = model_name or self.model
        search_results = self.search(query, n_results=n_results)

        retrieved_id = search_results["ids"][0][0]
        metadata = search_results["metadatas"][0][0]
        distance = search_results["distances"][0][0]

        concept = self.find_concept_by_id(retrieved_id)
        prompt = self.build_prompt(query, concept)
        generated = self.generate(prompt, selected_model)

        return {
            "query": query,
            "model": selected_model,
            "retrieved_id": retrieved_id,
            "topic": metadata.get(
                "topic",
                concept.get("topic", ""),
            ),
            "distance": distance,
            "intro": generated.get("intro", ""),
            "phrases": generated.get("phrases", []),
            "context": concept.get("answer_de", ""),
        }

    def compare_models(
        self,
        evaluation_queries: list[dict],
        candidate_models: list[str],
    ) -> list[dict]:
        """Run the RAG flow with multiple generation models."""
        all_results = []

        for model_name in candidate_models:
            for item in evaluation_queries:
                try:
                    result = self.rag(
                        query=item["query"],
                        model_name=model_name,
                        n_results=1,
                    )

                    result["expected_id"] = item["expected_id"]
                    result["retrieval_correct"] = (
                        result["retrieved_id"]
                        == item["expected_id"]
                    )

                    all_results.append(result)

                except Exception as error:
                    print(
                        f"Error with model {model_name} "
                        f"for query '{item['query']}': {error}"
                    )

        return all_results

    def judge_results(
        self,
        results: list[dict],
        judge_model: str,
        judge_answer_fn,
    ) -> list[dict]:
        """Evaluate generated answers using an LLM judge."""
        judged_results = []

        for result in results:
            answer_text = (
                result["intro"]
                + "\n"
                + "\n".join(result["phrases"])
            )

            try:
                scores = judge_answer_fn(
                    query=result["query"],
                    answer=answer_text,
                    context=result["context"],
                    judge_model=judge_model,
                )

                judged_results.append({
                    **result,
                    **scores,
                })

            except Exception as error:
                print(
                    f"Error judging result from "
                    f"{result['model']}: {error}"
                )

        return judged_results

    @staticmethod
    def summarize_by_model(
        judged_results: list[dict],
    ) -> dict[str, float]:
        """Calculate the average overall score per model."""
        scores_by_model = defaultdict(list)

        for result in judged_results:
            if result.get("overall") is not None:
                scores_by_model[result["model"]].append(
                    result["overall"]
                )

        summary = {
            model_name: sum(scores) / len(scores)
            for model_name, scores in scores_by_model.items()
            if scores
        }

        return dict(
            sorted(
                summary.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )