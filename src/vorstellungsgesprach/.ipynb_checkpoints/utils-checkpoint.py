import json
from pathlib import Path
from typing import Any
import re

from lingua import Language, LanguageDetectorBuilder


_detector = LanguageDetectorBuilder.from_languages(
    Language.GERMAN,
    Language.ENGLISH,
).build()

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

    # The dataset must be a list containing individual job records.
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list of job records, got {type(data).__name__}"
        )

    # Ensure that every item in the list is a dictionary.
    if not all(isinstance(job, dict) for job in data):
        raise ValueError("Every job record must be a JSON object")

    return data

def add_language_metadata(
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add the detected language and confidence to every job record."""
    enriched_jobs = []

    for job in jobs:
        description = str(job.get("description", "")).strip()
        enriched_job = job.copy()

        if len(description) < 100:
            enriched_job["detected_language"] = "unknown"
            enriched_job["language_confidence"] = 0.0
        else:
            confidence_values = _detector.compute_language_confidence_values(
                description
            )

            best_match = confidence_values[0]

            enriched_job["detected_language"] = best_match.language.name.lower()
            enriched_job["language_confidence"] = best_match.value

        enriched_jobs.append(enriched_job)

    return enriched_jobs


def normalize_text(text: str | None) -> str:
    """Normalize text for duplicate comparison."""
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip().casefold()

def regex_text(text: str | None) -> str:
    """Remove gender markers such as (m/w/d), (all genders), (gn), m/w/d"""
    if not text:
        return ""

    return re.sub(r"\s*(?:\(\s*(?:[a-z]/[a-z]/[a-z]|all genders?|gn)\s*\)|[a-z]/[a-z]/[a-z]|[0-9]{4})\s*", " ", text, flags=re.IGNORECASE).strip()

def remove_duplicate_jobs(
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove jobs with the same company and description."""
    unique_jobs = []
    seen = set()

    for job in jobs:
        job["company"] = normalize_text(job.get("company"))
        job["description_lower"] = normalize_text(job.get("description"))
        job["title"] = normalize_text(job.get("title"))
        job["title"] = regex_text(job.get("title"))
        # Create a unique key that ignores ID, title, and city.
        duplicate_key = (job["company"], job["description_lower"])

        if not job["description"] or duplicate_key in seen:
            continue

        seen.add(duplicate_key)
        unique_jobs.append(job)

    return unique_jobs