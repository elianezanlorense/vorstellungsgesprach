"""
Pipeline: pergunta do usuário -> retrieval vetorial (ChromaDB) -> geração de
resposta formatada via Gemini -> comparação entre modelos (LLM evaluation).

Pensado para ser colado em células do seu notebook, reaproveitando as
variáveis que você já tem: `client_gemini` (genai.Client), `model` (SentenceTransformer),
`collection` (ChromaDB), `topics` (lista de conceitos carregada de topics.json),
`evaluation_queries` (lista de perguntas de teste com expected_id).
"""

import json


# --------------------------------------------------------------------------
# 1. Geração da resposta final a partir do conceito recuperado
# --------------------------------------------------------------------------

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


def generate_answer(client_gemini, query: str, concept: dict, model_name: str) -> dict:
    """Chama o Gemini para gerar a resposta formatada (intro + phrases) a partir de um conceito."""
    prompt = ANSWER_PROMPT.format(
        query=query,
        topic=concept.get("topic", ""),
        question_de=concept.get("question_de", ""),
        answer_de=concept.get("answer_de", ""),
        phrases=", ".join(concept.get("phrases", [])),
    )
    response = client_gemini.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return json.loads(response.text)


# --------------------------------------------------------------------------
# 2. Fluxo completo: retrieval (vetorial) -> geração
# --------------------------------------------------------------------------

def find_concept_by_id(topics: list[dict], concept_id: str) -> dict:
    """Busca o conceito original completo (question_de, answer_de, phrases) pelo id."""
    for concept in topics:
        if concept["id"] == concept_id:
            return concept
    raise ValueError(f"Concept id não encontrado: {concept_id}")


def answer_query(
    client_gemini,
    query: str,
    collection,
    embed_model,
    topics: list[dict],
    gen_model: str,
    n_results: int = 1,
) -> dict:
    """Executa o fluxo completo: embed da query -> retrieval no Chroma -> geração via Gemini."""
    query_vector = embed_model.encode(query, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_results,
    )

    top_id = results["ids"][0][0]
    top_metadata = results["metadatas"][0][0]

    concept = find_concept_by_id(topics, top_id)
    generated = generate_answer(client_gemini, query, concept, model_name=gen_model)

    return {
        "query": query,
        "model": gen_model,
        "retrieved_id": top_id,
        "topic": top_metadata.get("topic", concept.get("topic", "")),
        "intro": generated.get("intro", ""),
        "phrases": generated.get("phrases", []),
    }


# --------------------------------------------------------------------------
# 3. Comparação entre modelos (LLM evaluation)
# --------------------------------------------------------------------------

def run_model_comparison(
    client_gemini,
    collection,
    embed_model,
    topics: list[dict],
    evaluation_queries: list[dict],
    candidate_models: list[str],
) -> list[dict]:
    """Roda o fluxo completo para cada pergunta de teste, em cada modelo candidato."""
    all_results = []
    for model_name in candidate_models:
        for item in evaluation_queries:
            try:
                result = answer_query(
                    client_gemini=client_gemini,
                    query=item["query"],
                    collection=collection,
                    embed_model=embed_model,
                    topics=topics,
                    gen_model=model_name,
                )
                result["expected_id"] = item["expected_id"]
                result["retrieval_correct"] = result["retrieved_id"] == item["expected_id"]
                all_results.append(result)
            except Exception as e:
                print(f"Erro com modelo {model_name} na query '{item['query']}': {e}")
    return all_results


def judge_results(client_gemini, results: list[dict], judge_model: str, judge_answer_fn) -> list[dict]:
    """Usa uma função de julgamento (ex: evaluation.judge_answer) para pontuar cada resposta gerada.

    `judge_answer_fn` deve ter assinatura compatível com:
        judge_answer_fn(query=..., answer=..., context=..., judge_model=...) -> dict
    retornando pelo menos as chaves 'faithfulness', 'helpfulness', 'overall'.
    """
    judged = []
    for result in results:
        answer_text = result["intro"] + " " + " | ".join(result["phrases"])
        try:
            scores = judge_answer_fn(
                query=result["query"],
                answer=answer_text,
                context=result["topic"],
                judge_model=judge_model,
            )
            judged.append({**result, **scores})
        except Exception as e:
            print(f"Erro ao julgar resultado ({result['model']}, '{result['query']}'): {e}")
    return judged


def summarize_by_model(judged_results: list[dict]) -> dict:
    """Agrega os scores por modelo (média de overall) para decidir o vencedor."""
    from collections import defaultdict

    by_model = defaultdict(list)
    for r in judged_results:
        if r.get("overall") is not None:
            by_model[r["model"]].append(r["overall"])

    summary = {
        model: sum(scores) / len(scores)
        for model, scores in by_model.items()
        if scores
    }
    return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))