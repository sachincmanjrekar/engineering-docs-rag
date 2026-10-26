"""Streamlit chat UI that talks to the FastAPI backend."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Engineering Docs Q&A", page_icon="📘")
st.title("📘 Engineering Document Q&A")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF manual or datasheet", type="pdf")
    if uploaded and st.button("Ingest"):
        with st.spinner("Extracting, chunking and embedding..."):
            r = requests.post(f"{API_URL}/ingest", files={"file": (uploaded.name, uploaded.getvalue())}, timeout=300)
        if r.ok:
            st.success(f"Ingested {r.json()['chunks']} chunks from {r.json()['pages']} pages")
        else:
            st.error(r.json().get("detail", r.text))
    docs = requests.get(f"{API_URL}/documents", timeout=30).json()["documents"]
    scope = st.selectbox("Search in", ["All documents", *docs])

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if question := st.chat_input("Ask about your documents..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Searching..."):
        payload = {"question": question, "source": None if scope == "All documents" else scope}
        r = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
        if not r.ok:
            st.error(r.text)
        else:
            body = r.json()
            st.markdown(body["answer"])
            for i, s in enumerate(body["sources"], start=1):
                with st.expander(f"[{i}] {s['source']} - page {s['page']} - {s['section']}"):
                    st.write(s["text"])
            st.session_state.history.append({"role": "assistant", "content": body["answer"]})
