"""LLM-as-judge evaluation for comparing model outputs."""

from __future__ import annotations

import re

from google import genai

from vorstellungsgesprach import conf


JUDGE_SYSTEM_PROMPT = """Du bist ein strenger Gutachter für Antworten eines \
Recherche-Assistenten für IT-Stellenanzeigen. Bewerte die gegebene Antwort anhand \
von zwei Kriterien, jeweils von 1 (schlecht) bis 5 (ausgezeichnet):

- FAITHFULNESS: Basiert die Antwort ausschließlich auf dem gegebenen Kontext, ohne \
Informationen zu erfinden?
- HELPFULNESS: Ist die Antwort konkret, gut strukturiert und nützlich für die Frage?

Antworte NUR in diesem exakten Format, ohne zusätzlichen Text:
FAITHFULNESS: <Zahl>
HELPFULNESS: <Zahl>"""


def judge_answer(
    query: str,
    answer: str,
    context: str,
    judge_model: str,
) -> dict:
    """Avalia uma resposta usando o modelo informado."""
    client = genai.Client(
        api_key=conf.GEMINI_API_KEY,
    )

    prompt = f"""Frage: {query}

Kontext (was dem Modell zur Verfügung stand):
{context}

Zu bewertende Antwort:
{answer}"""

    chat = client.chats.create(
        model=judge_model,
        config={
            "system_instruction": JUDGE_SYSTEM_PROMPT,
        },
    )

    response = chat.send_message(prompt)

    faithfulness = _extract_score(
        response.text,
        "FAITHFULNESS",
    )

    helpfulness = _extract_score(
        response.text,
        "HELPFULNESS",
    )

    overall = None

    if faithfulness is not None and helpfulness is not None:
        overall = (faithfulness + helpfulness) / 2

    return {
        "faithfulness": faithfulness,
        "helpfulness": helpfulness,
        "overall": overall,
    }


def _extract_score(
    text: str,
    label: str,
) -> float | None:
    """Extrai uma nota de 1 a 5."""
    match = re.search(
        rf"{label}:\s*([1-5])(?:\.0)?",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return float(match.group(1))