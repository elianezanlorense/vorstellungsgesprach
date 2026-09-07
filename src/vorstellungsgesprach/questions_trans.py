"""Extract Q&A pairs from an interview-questions markdown file and translate
them to German using Gemini.

Usage:
    uv run python scripts/translate_questions.py path/to/questions.md
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors

from vorstellungsgesprach import conf

TRANSLATE_SYSTEM_PROMPT = """Du bist ein professioneller Übersetzer für technische \
Interviewfragen im Bereich Data Science / Machine Learning. Übersetze den gegebenen \
englischen Text ins Deutsche. Behalte Fachbegriffe (z.B. "Overfitting", "Gradient \
Descent"), die im Deutschen ebenfalls gebräuchlich sind, wenn eine wörtliche \
Übersetzung unnatürlich klingen würde. Antworte NUR mit der Übersetzung, ohne \
zusätzlichen Text, Anführungszeichen oder Erklärungen."""

MAX_RETRIES = 5


def extract_questions(md_text: str) -> list[dict]:
    """Extrai pares (número, pergunta, resposta) de um markdown no formato
    '### Qn: pergunta ###' seguido do texto da resposta até o próximo '### Q'."""
    pattern = re.compile(
        r"###\s*Q(\d+):\s*(.+?)\s*###\s*\n(.*?)(?=###\s*Q\d+:|\Z)",
        re.DOTALL,
    )

    results = []
    for match in pattern.finditer(md_text):
        number, question, answer = match.groups()
        # remove markdown de imagem/links longos que não fazem sentido traduzir
        answer_clean = re.sub(r"!\[.*?\]\(.*?\)", "", answer).strip()
        results.append(
            {
                "number": int(number),
                "question_en": question.strip(),
                "answer_en": answer_clean,
            }
        )
    return results


def translate_text(text: str, client: genai.Client, model: str) -> str:
    if not text.strip():
        return text

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=text,
                config={"system_instruction": TRANSLATE_SYSTEM_PROMPT},
            )
            return response.text.strip()
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < MAX_RETRIES:
                wait = 30 * attempt  # espera crescente: 30s, 60s, 90s...
                print(f"  Rate limit atingido, aguardando {wait}s (tentativa {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Não foi possível traduzir após múltiplas tentativas.")


def translate_questions(
    questions: list[dict], model: str | None = None, output_path: Path | None = None
) -> list[dict]:
    model = model or conf.GEMINI_CHAT_MODEL
    client = genai.Client(api_key=conf.GEMINI_API_KEY)

    translated = []
    for i, item in enumerate(questions, start=1):
        print(f"Übersetze Frage {item['number']} ({i}/{len(questions)})...")
        question_de = translate_text(item["question_en"], client, model)
        answer_de = translate_text(item["answer_en"], client, model)
        translated.append(
            {
                "number": item["number"],
                "question_en": item["question_en"],
                "question_de": question_de,
                "answer_en": item["answer_en"],
                "answer_de": answer_de,
            }
        )
        # salva progresso a cada pergunta, para não perder trabalho se algo falhar
        if output_path:
            output_path.write_text(
                json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return translated


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: uv run python scripts/translate_questions.py caminho/arquivo.md")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    md_text = md_path.read_text(encoding="utf-8")

    questions = extract_questions(md_text)
    print(f"{len(questions)} Fragen gefunden.\n")

    output_path = Path("data/processed/questions_de.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    translated = translate_questions(
    questions,
    model="models/gemini-flash-lite-latest",
    output_path=output_path,
)

    print(f"\nFertig. Gespeichert unter {output_path}")