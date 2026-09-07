def build_documents(concepts: list[dict]) -> list[dict]:
    """Builds a list of documents from a list of concepts."""
    documents = []

    for concept in concepts:
        tags = concept.get("tags", [])
        phrases = concept.get("phrases", [])

        text = (
            f"Thema: {concept['topic']}\n"
            f"Frage: {concept['question_de']}\n"
            f"Antwort: {concept['answer_de']}\n"
            f"Tags: {', '.join(tags)}\n"
            f"Wichtige Ausdrücke: {', '.join(phrases)}"
        )

        documents.append({
            "id": concept["id"],
            "text": text,
            "metadata": {
                "topic": concept["topic"],
                "tags": ", ".join(tags),
            },
        })

    return documents



