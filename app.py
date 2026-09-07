"""Streamlit interface for the Technical German RAG assistant."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from src.app_service import (
    answer_query,
    build_rag_assistant,
)


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


st.set_page_config(
    page_title="Technisches Deutsch",
    page_icon="📚",
    layout="centered",
)


def get_api_key() -> str:
    """Read the API key locally or from Streamlit Cloud."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            api_key = ""

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found."
        )

    return api_key


@st.cache_resource
def get_rag_assistant():
    """Create and cache the application assistant."""
    return build_rag_assistant(
        api_key=get_api_key()
    )


try:
    rag_assistant = get_rag_assistant()

except Exception as error:
    st.error(
        f"Application initialization failed: {error}"
    )
    st.stop()


if "history" not in st.session_state:
    st.session_state.history = []


if "feedback" not in st.session_state:
    st.session_state.feedback = {}


def submit_feedback(
    result_id: str,
    query: str,
) -> None:
    """Save feedback in the current Streamlit session."""
    rating = st.session_state.get(
        f"rating_{result_id}"
    )

    comment = st.session_state.get(
        f"comment_{result_id}",
        "",
    ).strip()

    if rating is None and not comment:
        st.warning(
            "Bitte wähle eine Bewertung oder "
            "schreibe einen Kommentar."
        )
        return

    st.session_state.feedback[result_id] = {
        "query": query,
        "rating": rating,
        "comment": comment,
    }

    st.toast("Danke für dein Feedback!")


def render_answer(result: dict) -> None:
    """Render an answer and its feedback form."""
    intro = result.get("intro", "")
    phrases = result.get("phrases", [])
    context = (
        result.get("context", "")
        or result.get("answer_de", "")
    )

    result_id = result["result_id"]
    query = result["query"]

    if intro:
        st.markdown(intro)

    if phrases:
        st.markdown("#### Wichtige Ausdrücke")

        for phrase in phrases:
            st.markdown(f"- {phrase}")

    if context:
        with st.expander(
            "Vollständige Referenzantwort"
        ):
            st.write(context)

    with st.expander("Feedback"):
        st.feedback(
            "thumbs",
            key=f"rating_{result_id}",
        )

        st.text_area(
            "Dein Kommentar",
            placeholder="Was können wir verbessern?",
            key=f"comment_{result_id}",
        )

        st.button(
            "Absenden",
            key=f"submit_{result_id}",
            on_click=submit_feedback,
            args=(result_id, query),
        )


st.title("📚 Technisches Deutsch")

st.write(
    "Lerne technische Begriffe und wichtige "
    "Ausdrücke auf Deutsch."
)

st.caption(
    "Stelle eine Frage auf Deutsch. Die Anwendung "
    "sucht ein passendes Konzept in der Wissensbasis "
    "und erzeugt eine Antwort."
)


if st.button("Verlauf löschen"):
    st.session_state.history = []
    st.session_state.feedback = {}
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
            "Suche nach einemSuche nach einem passenden Konzept..."
        ):
            try:
                result = answer_query(
                    rag_assistant=rag_assistant,
                    query=query,
                )

            except Exception as error:
                st.error(
                    "Die Anfrage konnte nicht "
                    "verarbeitet werden."
                )
                st.exception(error)
                st.stop()

        result["query"] = query
        result["result_id"] = str(uuid4())

        render_answer(result)

    st.session_state.history.append(result)