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

TOPICS_PATH = "../data/raw/topics.json"
CHROMA_PATH = '../data/processed/chroma_db'
OUTPUT_DIR = '../outputs'

from build_documents import build_documents
from evaluation import evaluate_retrieval,evaluation_queries,hit_rate,mean_reciprocal_rank,compare_models_all_queries
from load_store_data import load_data
from models import list_available_chat_models
from rag import TechnicalGermanRAG




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


