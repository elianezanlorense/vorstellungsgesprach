"""Streamlit interface for the Technical German RAG assistant."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from load_data import load_data
from rag import TechnicalGermanRAG


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOPICS_PATH = "../data/raw/topics.json"
CHROMA_PATH = "../data/processed/chroma_db"

COLLECTION_NAME = "concepts_de"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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


st.set_page_config(
    page_title="Technisches Deutsch",
    page_icon="📚",
    layout="centered",
)

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Initialization (read-only: assumes `python main.py ingest` already ran)
# ---------------------------------------------------------------------------

@st.cache_resource
def create_rag_assistant() -> TechnicalGermanRAG:
    """Load the RAG resources from data already ingested by main.py."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in the environment.")

    if not TOPICS_PATH.is_file():
        raise FileNotFoundError(f"Topics file not found: {TOPICS_PATH}")

    if not CHROMA_PATH.is_dir():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {CHROMA_PATH}. "
            "Run 'python main.py ingest' first."
        )

    topics = load_data(TOPICS_PATH)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    available = [item.name for item in chroma_client.list_collections()]

    if COLLECTION_NAME not in available:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found. "
            f"Available: {available}. Run 'python main.py ingest' first."
        )

    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' is empty. "
            "Run 'python main.py ingest' first."
        )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    gemini_client = genai.Client(api_key=api_key)

    return TechnicalGermanRAG(
        collection=collection,
        embed_model=embedding_model,
        llm_client=gemini_client,
        topics=topics,
        model=GENERATION_MODEL,
        answer_prompt=ANSWER_PROMPT,
    )


try:
    rag_assistant = create_rag_assistant()
except Exception as error:
    st.error(f"Application initialization failed: {error}")
    st.stop()


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


def render_answer(result: dict) -> None:
    """Render a generated answer."""
    intro = result.get("intro", "")
    phrases = result.get("phrases", [])
    answer_de = result.get("answer_de", "")

    if intro:
        st.write(intro)

    if phrases:
        st.markdown("#### Wichtige Ausdrücke")
        for phrase in phrases:
            st.markdown(f"- {phrase}")

    if answer_de:
        with st.expander("Vollständige Referenzantwort"):
            st.write(answer_de)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.title("📚 Technisches Deutsch")
st.write("Lerne technische Begriffe und wichtige Ausdrücke auf Deutsch.")
st.caption(
    "Stelle eine Frage auf Deutsch. Die Anwendung sucht das passende "
    "Konzept in der Wissensbasis und erzeugt eine verständliche Antwort."
)

if st.button("Verlauf löschen"):
    st.session_state.history = []
    st.rerun()

for history_item in st.session_state.history:
    with st.chat_message("user"):
        st.write(history_item["query"])
    with st.chat_message("assistant"):
        render_answer(history_item)

query = st.chat_input("Stelle eine Frage zu einem technischen Konzept.")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Suche nach einem passenden Konzept..."):
            try:
                result = rag_assistant.rag(
                    query=query,
                    model_name=GENERATION_MODEL,
                    n_results=1,
                )
            except Exception as error:
                st.error(f"Die Anfrage konnte nicht verarbeitet werden: {error}")
                st.stop()

        result["query"] = query
        render_answer(result)

    st.session_state.history.append(result)