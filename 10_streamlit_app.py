"""
Streamlit web UI for the ASX RAG assistant.

Reuses the exact same retrieval + generation logic as 09_rag_chatbot.py
(imported directly, not duplicated) -- this file only adds the web
interface on top: a chat-style input, answer display, and an expandable
sources panel showing exactly which chunks were used and their relevance.

Usage: streamlit run 10_streamlit_app.py
"""

import os
import importlib.util

import streamlit as st
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Import functions from 09_rag_chatbot.py directly (filename starts with a
# digit, so normal `import` syntax doesn't work -- load it explicitly).
_spec = importlib.util.spec_from_file_location("rag_chatbot", "09_rag_chatbot.py")
rag_chatbot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag_chatbot)


st.set_page_config(
    page_title="ASX Annual Report Assistant",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Connecting to knowledge base...")
def load_chroma_collection():
    client = chromadb.PersistentClient(path=rag_chatbot.CHROMA_DB_PATH)
    return client.get_collection(name=rag_chatbot.COLLECTION_NAME)


@st.cache_resource(show_spinner="Connecting to Groq...")
def load_groq_client():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found. Make sure your .env file has "
            "GROQ_API_KEY=your_key_here in this folder, then restart the app."
        )
        st.stop()
    return Groq(api_key=api_key)


def render_sources(sources):
    """Show retrieved sources in an expandable panel, grouped clearly with
    relevance flags, so the person can verify where each answer came from."""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
        for s in sources:
            flag = " ⚠️ low relevance" if s["low_relevance"] else ""
            st.markdown(
                f"**{s['label']}** — {s['company']}, page {s['page']} "
                f"(distance: {s['distance']:.3f}){flag}"
            )
            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))
            st.divider()


def main():
    st.title("📊 ASX Annual Report Assistant")
    st.caption(
        "Ask questions about CBA, BHP, CSL, Woolworths, and Telstra's FY2025 "
        "annual reports. Answers are generated only from the retrieved report "
        "content, with sources shown below each answer."
    )

    model = load_embedding_model()
    collection = load_chroma_collection()
    groq_client = load_groq_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay prior turns
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("mode_note"):
                st.caption(msg["mode_note"])
            if msg["role"] == "assistant" and msg.get("fallback_note"):
                st.caption(f"⚠️ {msg['fallback_note']}")
            if msg["role"] == "assistant" and msg.get("citation_warnings"):
                for w in msg["citation_warnings"]:
                    st.warning(f"⚠️ {w}")
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    query = st.chat_input("Ask about net profit, revenue, dividends, risks...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching reports and generating answer..."):
                result = rag_chatbot.get_answer(model, collection, groq_client, query)
            st.markdown(result["answer"])
            if result["mode_note"]:
                st.caption(result["mode_note"])
            if result.get("fallback_note"):
                st.caption(f"⚠️ {result['fallback_note']}")
            if result.get("citation_warnings"):
                for w in result["citation_warnings"]:
                    st.warning(f"⚠️ {w}")
            render_sources(result["sources"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "mode_note": result["mode_note"],
            "fallback_note": result.get("fallback_note"),
            "citation_warnings": result.get("citation_warnings"),
            "sources": result["sources"],
        })

    with st.sidebar:
        st.subheader("About this assistant")
        st.markdown(
            "This is a Retrieval-Augmented Generation (RAG) system built over "
            "five ASX-listed companies' FY2025 annual reports:\n\n"
            "- Commonwealth Bank of Australia (CBA)\n"
            "- BHP Group\n"
            "- CSL Limited\n"
            "- Woolworths Group\n"
            "- Telstra Group\n\n"
            "**Stack:** PDF extraction (pdfplumber) → chunking → "
            "sentence-transformers embeddings → ChromaDB → Groq "
            "(Llama 3.3 70B) for answer generation."
        )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
