import sys
from pathlib import Path
print(Path.cwd())
import chromadb
from typing import Any
from vorstellungsgesprach.utils import load_data, add_language_metadata,remove_duplicate_jobs
from vorstellungsgesprach import conf
from vorstellungsgesprach import embeddings
from vorstellungsgesprach import evaluation
from vorstellungsgesprach import conf, rag
from vorstellungsgesprach.models import list_available_chat_models
from dotenv import load_dotenv
from google import genai
import os

def prepare_jobs(path: str | Path, min_language_confidence: float = 0.80) -> list[dict[str, Any]]:
    """Carrega, padroniza, filtra vagas em alemão e remove duplicatas."""
    raw_jobs = load_data(path)

    jobs = [
        {
            "id": str(job["id"]),
            "description": job.get("description", ""),
            "title": job.get("title", ""),
            "company": job.get("companyName", job.get("company", ""))
        }
        for job in raw_jobs
    ]

    jobs = add_language_metadata(jobs)

    german_jobs = [
        job
        for job in jobs
        if job["detected_language"] == "german"
        and job["language_confidence"] >= min_language_confidence
    ]

    return remove_duplicate_jobs(german_jobs)