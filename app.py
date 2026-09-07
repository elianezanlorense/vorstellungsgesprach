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

TOPICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "topics.json"
)

CHROMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chroma_db"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from load_store_data import load_data
from rag import TechnicalGermanRAG
from vorstellungsgesprach import db


COLLECTION_NAME = "concepts_de"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

GENERATION_MODEL = "models/gemini-flash-lite-latest"


ANSWER_PROMPT = """
Du bist ein Assistent für technisches Deutsch.

Beantworte die Anfrage ausschließlich anhand des bereitgestellten
technischen Konzepts.

Anfrage:
{query}

Thema:
{topic}

Referenzfrage:
{question_de}

Referenzantwort:
{answer_de}

Wichtige Ausdrücke:
{phrases}

Gib ausschließlich gültiges JSON in diesem Format zurück:

{{
  "intro": "Eine kurze Erklärung auf Deutsch.",
  "phrases": [
    "Wichtiger Ausdruck 1",
    "Wichtiger Ausdruck 2"
  ]
}}
""".strip()


st.set_page_config(
    page_title="Technisches Deutsch",
    page_icon="📚",
    layout="centered",
)

load_dotenv(PROJECT_ROOT / ".env")

db.init_db()


def validate_topics(topics: list[dict]) -> None:
    """Validate the technical concept dataset."""
    required_fields = {
        "id",
        "question_de",
        "answer_de",
        "topic",
        "tags",
        "phrases",
    }

    if not topics:
        raise ValueError("The topics dataset is empty.")

    concept_ids = set()

    for index, concept in enumerate(topics):
        if not isinstance(concept, dict):
            raise ValueError(
                f"Concept at position {index} is not a JSON object."
            )

        missing_fields = required_fields - concept.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))

            raise ValueError(
                f"Concept at position {index} is missing: {missing}. "
                f"Available fields: {list(concept.keys())}"
            )

        concept_id = concept["id"]

        if concept_id in concept_ids:
            raise ValueError(f"Duplicate concept ID: {concept_id}")

        concept_ids.add(concept_id)

        if not isinstance(concept["tags"], list):
            raise ValueError(
                f"The 'tags' field of {concept_id} must be a list."
            )

        if not isinstance(concept["phrases"], list):
            raise ValueError(
                f"The 'phrases' field of {concept_id} must be a list."
            )


@st.cache_resource
def create_rag_assistant() -> TechnicalGermanRAG:
    """Initialize and cache the RAG resources."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in the environment."
        )

    if not TOPICS_PATH.exists():
        raise FileNotFoundError(
            f"Topics file not found: {TOPICS_PATH}"
        )

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {CHROMA_PATH}"
        )

    topics = load_data(TOPICS_PATH)
    validate_topics(topics)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
    )

    available_collections = [
        item.name
        for item in chroma_client.list_collections()
    ]

    if COLLECTION_NAME not in available_collections:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' was not found. "
            f"Available collections: {available_collections}"
        )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
    )

    if collection.count() == 0:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' is empty."
        )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    gemini_client = genai.Client(
        api_key=api_key
    )

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


if "history" not in st.session_state:
    st.session_state.history = []


def render_answer(result: dict) -> None:
    """Render the generated answer, with feedback buttons."""
    intro = result.get("intro", "")
    phrases = result.get("phrases", [])
    context = result.get("context", "") or result.get("answer_de", "")

    if intro:
        st.markdown(intro)

    if phrases:
        st.markdown("#### Wichtige Ausdrücke")

        for phrase in phrases:
            st.markdown(f"- {phrase}")

    if context:
        with st.expander("Vollständige Referenzantwort"):
            st.write(context)

    interaction_id = result.get("interaction_id")
    if interaction_id is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍", key=f"up_{interaction_id}"):
                db.set_feedback(interaction_id, "up")
                st.toast("Danke für dein Feedback!")
        with col2:
            if st.button("👎", key=f"down_{interaction_id}"):
                db.set_feedback(interaction_id, "down")
                st.toast("Danke für dein Feedback!")


st.title("📚 Technisches Deutsch")

st.write(
    "Lerne technische Begriffe und wichtige Ausdrücke auf Deutsch."
)

st.caption(
    "Stelle eine Frage auf Deutsch. Die Anwendung sucht ein passendes "
    "Konzept in der Wissensbasis und erzeugt eine Antwort."
)


if st.button("Verlauf löschen"):
    st.session_state.history = []
    st.rerun()


for history_item in st.session_state.history:
    with st.chat_message("user"):
        st.write(history_item["query"])

    with st.chat_message("assistant"):
        render_answer(history_item)


query = st.chat_input(
    "Stelle eine Frage zu einem technischen Konzept."
)


if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner(
            "Suche nach einem passenden Konzept..."
        ):
            try:
                result = rag_assistant.rag(
                    query=query,
                    model_name=GENERATION_MODEL,
                    n_results=1,
                )

            except Exception as error:
                st.error(
                    "Die Anfrage konnte nicht verarbeitet werden."
                )
                st.exception(error)
                st.stop()

        result["query"] = query

        interaction_id = db.log_interaction(
            query=query,
            answer=(result.get("intro", "") + " " + " | ".join(result.get("phrases", []))).strip(),
            model=GENERATION_MODEL,
            sources=[{"title": result.get("topic", ""), "company": ""}],
        )
        result["interaction_id"] = interaction_id

        render_answer(result)

    st.session_state.history.append(result)