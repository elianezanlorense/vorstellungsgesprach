def chunk_description(text: str, max_chars: int = 4000, overlap: int = 200) -> list[str]:
    """Quebra a descrição em pedaços se for muito longa; senão, retorna como está."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap  # overlap evita cortar contexto no meio de uma frase importante

    return chunks


def build_documents(jobs: list[dict]) -> list[dict]:
    """Transforma cada vaga em um ou mais documentos, dependendo do tamanho."""
    documents = []
    for job in jobs:
        description_chunks = chunk_description(job["description"])

        for i, chunk in enumerate(description_chunks):
            text = f"Titel: {job['title']}\nUnternehmen: {job['company']}\n\n{chunk}"
            documents.append({
                "id": f"{job['id']}_chunk{i}" if len(description_chunks) > 1 else job["id"],
                "text": text,
                "metadata": {
                    "job_id": job["id"],
                    "title": job["title"],
                    "company": job["company"],
                    "chunk_index": i,
                    "total_chunks": len(description_chunks),
                },
            })
    return documents

