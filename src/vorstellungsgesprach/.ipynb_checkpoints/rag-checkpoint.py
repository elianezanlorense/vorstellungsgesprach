"""RAG pipeline: retrieval (local ONNX embeddings) + generation (Gemini chat)."""

from __future__ import annotations

from typing import Any

from google import genai

from vorstellungsgesprach import conf, embeddings

SYSTEM_PROMPT = """Du bist ein Assistent, der beim Aufbau von Fachvokabular für Vorstellungsgespräche hilft.

Basierend auf den folgenden Stellenanzeigen-Ausschnitten, die "{query}" erwähnen, liste JEDE wörtliche Textstelle auf, in der der Begriff vorkommt. Gib KEINE Zusammenfassung oder Erklärung — nur die Originalformulierungen aus dem Text.

Format pro Eintrag:
**[Firma]**: "[wörtliche Textstelle mit dem Begriff]"

Kontext:
{context}
"""

def retrieve(
    collection,
    query: str,
    n_results: int = 3,
) -> list[dict[str, Any]]:
    """Busca os documentos mais relevantes para a pergunta."""
    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
    )

    hits = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": distance,
            }
        )

    return hits


def build_prompt(
    query: str,
    hits: list[dict[str, Any]],
) -> str:
    """Monta o prompt com o contexto recuperado."""
    context_blocks = []

    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]

        context_blocks.append(
            f"[Quelle {index}] "
            f"{metadata.get('title', 'Unbekannter Titel')} bei "
            f"{metadata.get('company', 'Unbekanntes Unternehmen')}:\n"
            f"{hit['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""Kontext (echte Stellenanzeigen):

{context}

Frage: {query}

Antworte basierend auf dem obigen Kontext und zitiere die Quelle (z. B. [Quelle 1])."""


def answer(
    collection,
    query: str,
    model: str | None = None,
    n_results: int = 3,
) -> dict[str, Any]:
    """Pipeline completo: retrieval + geração de resposta."""
    selected_model = model or conf.GEMINI_CHAT_MODEL

    hits = retrieve(
        collection=collection,
        query=query,
        n_results=n_results,
    )

    prompt = build_prompt(
        query=query,
        hits=hits,
    )

    client = genai.Client(api_key=conf.GEMINI_API_KEY)

    chat = client.chats.create(
        model=selected_model,
        config={
            "system_instruction": SYSTEM_PROMPT,
        },
    )

    response = chat.send_message(prompt)

    return {
        "query": query,
        "model": selected_model,
        "answer": response.text,
        "sources": hits,
    }