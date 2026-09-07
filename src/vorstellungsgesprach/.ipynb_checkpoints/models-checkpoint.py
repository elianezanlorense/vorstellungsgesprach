"""Compara respostas do RAG usando todos os modelos Gemini disponíveis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chromadb
from google import genai

from vorstellungsgesprach import conf, rag


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "vagas_ti"
DEFAULT_QUESTION = "Welche Jobs erfordern Erfahrung mit Machine Learning?"

_SKIP_SUBSTRINGS = [
    "embedding",
    "imagen",
    "veo",
    "tts",
    "audio",
    "robotics"
]


def list_available_chat_models(client: genai.Client) -> list[str]:
    """Retorna somente os modelos que conseguem gerar texto."""
    available_models: list[str] = []

    for model in client.models.list():
        model_name = model.name

        if not model_name:
            continue

        if any(
            substring in model_name.casefold()
            for substring in _SKIP_SUBSTRINGS
        ):
            continue

        try:
            chat = client.chats.create(model=model_name)
            response = chat.send_message("Reply only with OK.")

            if response.text:
                available_models.append(model_name)

        except Exception:
            continue

    return available_models


def compare_models(question: str) -> list[dict[str, Any]]:
    """Executa a pergunta e exibe somente as respostas bem-sucedidas."""
    client = genai.Client(api_key=conf.GEMINI_API_KEY)
    available_models = list_available_chat_models(client)

    if not available_models:
        print("There is no model available.")
        return []

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
    )

    results: list[dict[str, Any]] = []

    for model_name in available_models:
        try:
            result = rag.answer(
                collection=collection,
                query=question,
                model=model_name,
            )

            results.append(result)

            print(f"\n{'=' * 80}")
            print(f"Modelo: {model_name}")
            print(f"{'=' * 80}")
            print(result["answer"])

        except Exception:
            continue

    return results


def main() -> None:
    """Executa a comparação pelo terminal."""
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_QUESTION
    )

    compare_models(question)


if __name__ == "__main__":
    main()