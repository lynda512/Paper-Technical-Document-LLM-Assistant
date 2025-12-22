import streamlit as st
from pathlib import Path
import time
import shutil

# Professionalized modules for document ingestion and processing
from src.ingestion.pdf_loader import load_pdf_pages 
from src.preprocessing.chunker import RecursiveChunker 
from src.rag_pipeline.pipeline import RAGPipeline, RAGConfig

# --- 1. Configuration & Persistence ---
st.set_page_config(page_title="AI Engineering Assistant", layout="wide")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_papers"
PERSIST_DIR = DATA_DIR / "vector_store"

# Ensure directories exist for persistent storage
RAW_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_pipeline():
    """Initializes the RAG pipeline with optimized local model configurations."""
    cfg = RAGConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        # Points to the Llama 3 8B Instruct model running in LM Studio
        llm_model="llama-3-8b-local", 
        persist_dir=str(PERSIST_DIR),
        top_k=4 
    )
    return RAGPipeline(cfg)

# Initialize components
pipeline = get_pipeline()
chunker = RecursiveChunker(chunk_size=800, chunk_overlap=150)

# --- 2. Sidebar - Status & Maintenance ---
with st.sidebar:
    st.header("System Status")
    st.info("LLM: Llama 3 8B (via Local Inference)")
    st.success("Vector Store: Persistent ChromaDB")
    st.markdown("---")
    
    # Feature to prevent stale data collisions
    if st.button("Clear Vector Database"):
        if PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)
            PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            st.warning("Database cleared. Please re-upload your document.")
            time.sleep(1)
            st.rerun()

# --- 3. Main Interface & Ingestion ---
st.title("📄 Paper Technical Document Assistant")
st.markdown("Extract verifiable insights from research documents with page-level citations.")

uploaded_file = st.file_uploader("Upload a PDF paper", type=["pdf"])

if uploaded_file is not None:
    dest = RAW_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    
    with st.spinner("Processing document structure..."):
        # Load text page-by-page to preserve citation metadata
        pages = load_pdf_pages(dest) 
        
        all_chunks = []
        # Corrected Indentation: Iterating through pages to create granular chunks
        for page_data in pages:
            # Passes page-specific metadata (source/page) into the chunker
            chunks = chunker.chunk_text(page_data["text"], doc_metadata=page_data)
            all_chunks.extend(chunks)
            
        # Index all newly created chunks into the vector store
        pipeline.index_chunks(all_chunks)
        
    st.success(f"Successfully indexed {len(all_chunks)} granular chunks from {uploaded_file.name}")

# --- 4. Retrieval & Analysis ---
question = st.text_input("Enter your research question:")

if st.button("Generate Insight") and question:
    with st.spinner("Analyzing knowledge base..."):
        # Pipeline returns a unified result dictionary including answer and citations
        result = pipeline.answer(question)
        
    st.subheader("Analysis")
    # Displays the generated synthesis from Llama 3
    st.write(result.get("answer", "No answer generated."))
    
    # --- 5. Source Transparency Section ---
    with st.expander("View Source Citations"):
        citations = result.get("citations", [])
        source_docs = result.get("source_docs", [])
        
        if not citations:
            st.info("No relevant passages found in current context.")
        else:
            for i, meta in enumerate(citations):
                # Displays accurate source and page metadata
                src_name = meta.get('source', 'Unknown')
                pg_num = meta.get('page', 'N/A')
                st.markdown(f"**Source {i+1}:** {src_name} (Page {pg_num})")
                
                # Show document snippet for verification
                if i < len(source_docs):
                    st.caption(source_docs[i][:300] + "...") 
    st.divider()
    st.caption(f"Metrics: Latency {result.get('latency_ms', 0)}ms | Evidence Count: {result.get('retrieval_count', 0)}")