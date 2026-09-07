from sentence_transformers import SentenceTransformer
import numpy as np

def create_embeddings(
    documents: list[dict],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> tuple[SentenceTransformer, np.ndarray]:
    """Load the embedding model and generate document vectors."""

    if not documents:
        raise ValueError("No documents were provided.")

    model = SentenceTransformer(model_name)

    texts = [
        document["text"]
        for document in documents
    ]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return model, vectors