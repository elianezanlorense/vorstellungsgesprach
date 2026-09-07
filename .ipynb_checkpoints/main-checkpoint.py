"""Ingestion and evaluation pipeline for the Technical German RAG project."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent

data_path = "../data/raw/topics.json"
CHROMA_PATH = '../data/processed/chroma_db'
#output = '../outputs'

from src.build_documents import build_documents
from src.evaluation import evaluate_retrieval,evaluation_queries,hit_rate,mean_reciprocal_rank,compare_models_all_queries
from src.embedding import create_embeddings
from src.load_store_data import load_data,store_documents
from src.models import list_available_chat_models
from src.rag import TechnicalGermanRAG




COLLECTION_NAME = "concepts_de"
EMBEDDING_MODEL = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
GENERATION_MODEL = "models/gemini-flash-lite-latest"

ANSWER_PROMPT = """
Du hilfst einer Person dabei, technisches Deutsch zu lernen.

Basierend auf dem folgenden Konzept, beantworte die Nutzerfrage NICHT mit
einem langen Fließtext. Gib stattdessen zurück:
1. Eine sehr kurze Einleitung (maximal ein Satz)
2. Die wichtigsten Ausdrücke aus dem Konzept, die man sich merken sollte

Nutzerfrage: {query}

Konzept:
Thema: {topic}
Frage: {question_de}
Antwort: {answer_de}
Wichtige Ausdrücke: {phrases}

Antworte ausschließlich mit einem gültigen JSON-Objekt:
{{"intro": "...", "phrases": ["...", "...", "..."]}}
""".strip()


def run_pipeline(
    evaluation_queries,
    data_path=data_path,
    chroma_path="../data/processed/chroma_db",
    collection_name="concepts_de",
    evaluation_output_path="../outputs/evaluation_queries.txt",
    embedding_model_name=(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    ),
):
    topics = load_data(data_path)

    documents = build_documents(topics)

    model, vectors = create_embeddings(
        documents=documents,
        model_name=embedding_model_name,
    )

    collection = store_documents(
        documents=documents,
        vectors=vectors,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )

    retrieval_results = evaluate_retrieval(
        evaluation_queries=evaluation_queries,
        model=model,
        collection=collection,
        output_path=evaluation_output_path,
    )

    return {
        "topics": topics,
        "documents": documents,
        "model": model,
        "vectors": vectors,
        "collection": collection,
        "retrieval_results": retrieval_results,
    }