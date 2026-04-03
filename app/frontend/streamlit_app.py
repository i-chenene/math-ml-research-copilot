import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Math/ML Research Copilot", layout="wide")
st.title("Math/ML Research Copilot")
st.caption("Upload papers, ask grounded questions, and compare two papers with citations.")

st.subheader("1) Upload PDFs")
uploaded_files = st.file_uploader(
    "Select one or more PDF papers", type=["pdf"], accept_multiple_files=True
)
if st.button("Ingest PDFs", disabled=not uploaded_files):
    files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
    response = requests.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=120)
    if response.ok:
        st.success("Documents ingested successfully.")
        st.json(response.json())
    else:
        st.error(f"Upload failed: {response.text}")

st.divider()
st.subheader("2) Ingested documents")
docs_resp = requests.get(f"{BACKEND_URL}/documents", timeout=30)
documents = docs_resp.json() if docs_resp.ok else []
st.dataframe(documents, use_container_width=True)

st.divider()
st.subheader("3) Ask questions with citations")
question = st.text_input("Your question")
top_k = st.slider("Top-k chunks", min_value=1, max_value=15, value=5)
if st.button("Ask", disabled=not question.strip()):
    payload = {"question": question, "top_k": top_k}
    answer_resp = requests.post(f"{BACKEND_URL}/qa/ask", json=payload, timeout=120)
    if answer_resp.ok:
        result = answer_resp.json()
        st.markdown("**Answer**")
        st.write(result["answer"])
        st.markdown("**Citations**")
        for citation in result["citations"]:
            st.code(citation)
        with st.expander("Retrieval debug"):
            st.json(result["retrieved_chunks"])
    else:
        st.error(answer_resp.text)

st.divider()
st.subheader("4) Compare two papers")
if documents:
    options = {f'{d["id"]}: {d["name"]}': d["id"] for d in documents}
    col1, col2 = st.columns(2)
    with col1:
        left_key = st.selectbox("Paper A", list(options.keys()), key="paper_a")
    with col2:
        right_key = st.selectbox("Paper B", list(options.keys()), key="paper_b")
    if st.button("Compare papers", disabled=left_key == right_key):
        compare_payload = {
            "document_a_id": options[left_key],
            "document_b_id": options[right_key],
        }
        compare_resp = requests.post(
            f"{BACKEND_URL}/papers/compare", json=compare_payload, timeout=120
        )
        if compare_resp.ok:
            st.json(compare_resp.json())
        else:
            st.error(compare_resp.text)
else:
    st.info("Upload at least two papers to compare.")
