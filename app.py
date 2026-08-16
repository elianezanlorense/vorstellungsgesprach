"""Streamlit interface for the Vorstellungsgesprach RAG assistant."""

from __future__ import annotations

import chromadb
import streamlit as st

from vorstellungsgesprach import db, rag


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "vagas_ti"
MODEL_NAME = "models/gemini-flash-lite-latest"
N_RESULTS = 4


st.set_page_config(
    page_title="Vorstellungsgesprach Assistent",
    page_icon="💼",
)

db.init_db()


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
    )

    return client.get_collection(
        name=COLLECTION_NAME,
    )


collection = get_collection()

st.title("💼 Vorstellungsgesprach Assistent")

st.caption(
    "Stelle Fragen zu echten IT-Stellenanzeigen "
    "in Deutschland – auf Deutsch."
)


if "history" not in st.session_state:
    st.session_state.history = []


def render_answer(entry: dict) -> None:
    st.write(entry["answer"])

    with st.expander("Quellen"):
        for source in entry["sources"]:
            metadata = source["metadata"]

            st.markdown(
                f"**{metadata['title']}** "
                f"@ {metadata['company']}"
            )

            st.caption(
                source["text"][:300] + "..."
            )

    interaction_id = entry["interaction_id"]
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍", key=f"up_{interaction_id}"):
            db.set_feedback(interaction_id, "up")
            st.toast("Danke für dein Feedback!")

    with col2:
        if st.button("👎", key=f"down_{interaction_id}"):
            db.set_feedback(interaction_id, "down")
            st.toast("Danke für dein Feedback!")


for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])

    with st.chat_message("assistant"):
        render_answer(entry)


query = st.chat_input(
    "Deine Frage (z. B. „Welche Jobs erfordern Python?“)"
)


if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Suche in Stellenanzeigen..."):
            try:
                result = rag.answer(
                    collection=collection,
                    query=query,
                    model=MODEL_NAME,
                    n_results=N_RESULTS,
                )
            except Exception:
                st.error(
                    "Die Anfrage-Quota wurde überschritten. Bitte warte "
                    "kurz und versuche es erneut."
                )
                st.stop()

        interaction_id = db.log_interaction(
            query=result["query"],
            answer=result["answer"],
            model=result.get("model"),
            sources=[
                {"title": s["metadata"]["title"], "company": s["metadata"]["company"]}
                for s in result["sources"]
            ],
        )
        result["interaction_id"] = interaction_id

        render_answer(result)

    st.session_state.history.append(result)