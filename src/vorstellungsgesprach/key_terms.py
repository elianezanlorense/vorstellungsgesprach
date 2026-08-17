from collections import Counter, defaultdict
from typing import Any

from google import genai
from pydantic import BaseModel, Field

from vorstellungsgesprach import conf


class Competency(BaseModel):
    canonical_name: str = Field(description="Nome universal e curto da competência, por exemplo Python, Machine Learning ou Model Evaluation.")
    original_expression: str = Field(description="Expressão exata encontrada na vaga, preservando o alemão original.")
    category: str = Field(description="Categoria inferida, como programming_language, tool, methodology, cloud, database, soft_skill ou qualification.")
    requirement_level: str = Field(description="Nível inferido, como basic, good, advanced, expert ou unspecified.")
    context: str = Field(description="Frase completa da vaga na qual a competência aparece.")


class JobAnalysis(BaseModel):
    canonical_role: str = Field(description="Função profissional universal inferida a partir do título e da descrição.")
    seniority: str = Field(description="Nível inferido: intern, junior, mid, senior, lead ou unspecified.")
    competencies: list[Competency]


EXTRACTION_PROMPT = """Analysiere die folgende deutsche Stellenanzeige semantisch.

Extrahiere alle fachlichen Kompetenzen, Technologien, Methoden, Tätigkeiten,
Qualifikationen und relevanten Soft Skills.

Normalisiere unterschiedliche Formulierungen automatisch auf denselben
kanonischen Begriff. Zum Beispiel müssen „Python“, „Python-Kenntnisse“ und
„Erfahrung mit Python“ den canonical_name „Python“ erhalten.

Der canonical_name soll kurz, stabil und nicht als vollständiger Satz
formuliert sein. original_expression und context müssen exakt aus der
Stellenanzeige übernommen werden. Erfinde keine Kompetenzen."""


def analyze_job(job: dict[str, Any], model: str = "models/gemini-flash-lite-latest") -> dict[str, Any]:
    client = genai.Client(api_key=conf.GEMINI_API_KEY)
    text = f"Titel: {job['title']}\nUnternehmen: {job['company']}\n\n{job['description']}"

    response = client.models.generate_content(
        model=model,
        contents=f"{EXTRACTION_PROMPT}\n\n{text}",
        config={
            "response_mime_type": "application/json",
            "response_schema": JobAnalysis,
        },
    )

    analysis = JobAnalysis.model_validate_json(response.text)

    return {
        **job,
        "canonical_role": analysis.canonical_role,
        "seniority": analysis.seniority,
        "competencies": [item.model_dump() for item in analysis.competencies],
    }


def analyze_jobs(jobs: list[dict[str, Any]], model: str = "models/gemini-flash-lite-latest") -> list[dict[str, Any]]:
    results = []

    for index, job in enumerate(jobs, start=1):
        print(f"Analysiere Stelle {index}/{len(jobs)}: {job['title']}")
        results.append(analyze_job(job, model=model))

    return results