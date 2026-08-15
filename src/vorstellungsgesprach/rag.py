"""RAG pipeline: retrieval (local ONNX embeddings) + generation (Gemini chat)."""
from __future__ import annotations

from typing import Any

from vorstellungsgesprach import conf, embeddings


SYSTEM_PROMPT = """Du bist ein Assistent, der Kandidaten bei der Vorbereitung auf \
IT-Vorstellungsgespräche in Deutschland hilft. Nutze AUSSCHLIESSLICH den bereitgestellten \
Kontext (echte Stellenanzeigen), um Fragen zu beantworten. Wenn die Information im Kontext \
nicht vorhanden ist, sage das ehrlich, anstatt etwas zu erfinden. Antworte auf Deutsch, \
klar und praxisnah."""


def retrieve(collection, query: str, n_results: int = 3) -> list[dict[str, Any]]:
    """Busca os documentos mais relevantes para a pergunta."""
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=n_results)

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "metadata": meta, "distance": distance})
    return hits


def build_prompt(query: str, hits: list[dict[str, Any]]) -> str:
    """Monta o prompt com o contexto recuperado."""
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        context_blocks.append(
            f"[Quelle {i}] {meta['title']} bei {meta['company']}:\n{hit['text']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""Kontext (echte Stellenanzeigen):

{context}

Frage: {query}

Antworte basierend auf dem obigen Kontext und zitiere die Quelle (z.B. [Quelle 1])."""


def answer(collection, query: str, model: str | None = None, n_results: int = 3) -> dict[str, Any]:
    """Pipeline completo: retrieval + geração de resposta."""
    from google import genai

    model = model or conf.GEMINI_CHAT_MODEL

    hits = retrieve(collection, query, n_results=n_results)
    prompt = build_prompt(query, hits)

    client = genai.Client(api_key=conf.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"system_instruction": SYSTEM_PROMPT},
    )

    return {
        "query": query,
        "model": model,
        "answer": response.text,
        "sources": hits,
    }