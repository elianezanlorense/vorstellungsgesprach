"""Streamlit interface for the Vorstellungsgesprach RAG assistant."""

from __future__ import annotations

import chromadb
import streamlit as st

from vorstellungsgesprach import rag


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "vagas_ti"
MODEL_NAME = "models/gemma-4-26b-a4b-it"
N_RESULTS = 20


st.set_page_config(
    page_title="Vorstellungsgesprach Assistent",
    page_icon="💼",
)


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


for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])

    with st.chat_message("assistant"):
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


query = st.chat_input(
    "Deine Frage (z. B. „Welche Jobs erfordern Python?“)"
)


if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Suche in Stellenanzeigen..."):
            result = rag.answer(
                collection=collection,
                query=query,
                model=MODEL_NAME,
                n_results=N_RESULTS,
            )

        st.write(result["answer"])

        with st.expander("Quellen"):
            for source in result["sources"]:
                metadata = source["metadata"]

                st.markdown(
                    f"**{metadata['title']}** "
                    f"@ {metadata['company']}"
                )

                st.caption(
                    source["text"][:300] + "..."
                )

        col1, col2 = st.columns(2)

        with col1:
            st.button(
                "👍",
                key=f"up_{len(st.session_state.history)}",
            )

        with col2:
            st.button(
                "👎",
                key=f"down_{len(st.session_state.history)}",
            )

    st.session_state.history.append(result)