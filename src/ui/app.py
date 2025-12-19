import streamlit as st
from pathlib import Path
import time

from src.ingestion.pdf_loader import load_pdf_pages # Updated to page-level
from src.preprocessing.chunker import RecursiveChunker # Professional chunking
from src.rag_pipeline.pipeline import RAGPipeline, RAGConfig

# 1. Configuration & Styling
st.set_page_config(page_title="AI Engineering Assistant", layout="wide")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_papers"
PERSIST_DIR = DATA_DIR / "vector_store"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_pipeline():
    cfg = RAGConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        # Use a model capable of following instructions (e.g., Zephyr or Llama)
        llm_model="HuggingFaceH4/zephyr-7b-beta", 
        persist_dir=str(PERSIST_DIR),
        top_k=4
    )
    return RAGPipeline(cfg)

pipeline = get_pipeline()
chunker = RecursiveChunker(chunk_size=800, chunk_overlap=150)

# 2. Sidebar - Application Info
with st.sidebar:
    st.header("System Status")
    st.info("Model: Zephyr-7B (Quantized)")
    st.success("Vector Store: Connected")
    st.markdown("---")
    st.write("This tool demonstrates **Trustworthy AI** by providing source citations for every answer.")

# 3. Main Interface
st.title("📄 Paper Technical Document Assistant")
st.markdown("Analyze research and industry papers with verifiable AI insights.")

uploaded_file = st.file_uploader("Upload a PDF paper", type=["pdf"])

if uploaded_file is not None:
    dest = RAW_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing document structure..."):
        # Load pages and chunk with metadata
        pages = load_pdf_pages(dest) 
        all_chunks = []
        for page in pages:
            # Pass the page metadata to the chunker
            chunks = chunker.chunk_text(page["text"], doc_metadata=page)
            all_chunks.extend(chunks)
        
        pipeline.index_chunks(all_chunks)
    st.success(f"Successfully indexed {len(all_chunks)} granular chunks.")

# 4. Question & Verified Answer
question = st.text_input("Enter your research question:")

if st.button("Generate Insight") and question:
    with st.spinner("Consulting knowledge base..."):
        result = pipeline.answer(question)
        
    # UI layout for "Trustworthy" output
    st.subheader("Analysis")
    st.write(result["answer"])
    
    # 5. Source Transparency (New Section)
    with st.expander("View Source Citations"):
        for i, meta in enumerate(result["citations"]):
            st.markdown(f"**Source {i+1}:** {meta['source']} (Page {meta['page']})")
            st.caption(result["source_docs"][i][:300] + "...") # Show snippet

    st.divider()
    st.caption(f"Performance Metrics: Latency {result['latency_ms']}ms | Evidence Count: {result['retrieval_count']}")