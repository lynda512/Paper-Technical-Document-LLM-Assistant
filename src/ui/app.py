import streamlit as st
from pathlib import Path
import time

from src.ingestion.pdf_loader import load_pdf
from src.preprocessing.chunker import chunk_text
from src.rag_pipeline.pipeline import RAGPipeline, RAGConfig

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_papers"
PERSIST_DIR = DATA_DIR / "vector_store"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_pipeline():
    cfg = RAGConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    
        llm_model="sshleifer/tiny-gpt2",

        persist_dir=str(PERSIST_DIR),
    )
    return RAGPipeline(cfg)

pipeline = get_pipeline()

st.title("Paper LLM Assistant (RAG)")

uploaded_file = st.file_uploader("Upload a PDF paper", type=["pdf"])
if uploaded_file is not None:
    dest = RAW_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.read())
    with st.spinner("Indexing paper..."):
        text = load_pdf(dest)
        chunks = chunk_text(text, doc_id=dest.stem)
        pipeline.index_chunks(chunks)
    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

question = st.text_input("Ask a question about the indexed papers")
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        t0 = time.time()
        result = pipeline.answer(question)
        dt = int((time.time() - t0) * 1000)
    st.subheader("Answer")
    st.write(result["answer"])
    st.caption(f"Latency: {dt} ms, top-{len(result['source_docs'])} chunks used.")
