import json
from pathlib import Path
from typing import Any
import numpy as np
import chromadb

def load_data(path: str | Path) -> list[dict[str, Any]]:
    """Load job records from a JSON file.

    The JSON file must contain a list of job records. Each job record is
    represented as a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        A list of job-record dictionaries.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the JSON does not contain a list.
        json.JSONDecodeError: If the file does not contain valid JSON.
    """
    path = Path(path)

    # Check whether the input file exists.
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Read and deserialize the JSON file.
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # The dataset must be a list containing individualrecords.
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list ofrecords, got {type(data).__name__}"
        )

    # Ensure that every item in the list is a dictionary.
    if not all(isinstance(concept, dict) for concept in data):
        raise ValueError("Every concept must be a JSON object")

    return data



def store_documents(
    documents: list[dict[str, Any]],
    vectors: np.ndarray,
    chroma_path: str | Path,
    collection_name: str,
):
    """Store document embeddings in ChromaDB."""

    if not documents:
        raise ValueError("No documents were provided.")

    if len(documents) != len(vectors):
        raise ValueError(
            "The number of documents and vectors must be equal."
        )

    chroma_path = Path(chroma_path).resolve()

    # Cria data/processed/chroma_db caso ainda não exista
    chroma_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    client_chroma = chromadb.PersistentClient(
        path=str(chroma_path)
    )

    collection = client_chroma.get_or_create_collection(
        name=collection_name,
    )

    collection.upsert(
        ids=[
            document["id"]
            for document in documents
        ],
        embeddings=vectors.tolist(),
        documents=[
            document["text"]
            for document in documents
        ],
        metadatas=[
            document["metadata"]
            for document in documents
        ],
    )

    return collection