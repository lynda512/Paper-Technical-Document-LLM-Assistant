
# AI Engineering Assistant for Technical PDFs

This project is a **RAG-based assistant** for research papers and technical documents.  
Users upload a PDF, the system builds a semantic index with page‑level metadata, and a local LLM generates answers **with explicit source citations**.

The goal is to move from a quick RAG demo to a **hardened, observable assistant** that surfaces verifiable insights instead of opaque answers.

---

## Features

-  **PDF ingestion with page metadata**  
  - Parses PDFs page‑by‑page and preserves `source` (file name) and `page` for each chunk.
  - Uses a `RecursiveChunker` to create overlapping text chunks suitable for semantic search.

-  **Semantic retrieval over ChromaDB**  
  - Embeddings via `sentence-transformers/all-MiniLM-L6-v2`.
  - Persistent ChromaDB index (HNSW cosine) with upsert and metadata filtering for future extensions.
-  **Local LLM question answering**  
  - `LLMClient` talks to a locally hosted instruction model (e.g. Llama 3 8B in LM Studio) via HTTP.
  - Prompts built from a system instruction (`SYSTEM_PROMPT`) plus a formatted context containing page‑level citations.

-  **Trust & transparency**  
  - Every retrieved chunk is tagged with `[Source: <file>, Page: <n>]` and passed into the LLM context.  
  - The UI exposes all supporting passages under “View Source Citations” so users can manually verify answers.

- **Basic observability**  
  - Pipeline logs retrieval size, context length, and end‑to‑end latency in milliseconds.
  - Streamlit footer displays `Latency` and `Evidence Count` for each query.

---

## Architecture

```
PDF upload
   ↓
Page-wise parsing (pdf_loader.load_pdf_pages)
   ↓
Recursive chunking with metadata (RecursiveChunker)
   ↓
Embeddings (Embedder → MiniLM)
   ↓
Vector store (ChromaRetriever → Chroma PersistentClient)
   ↓
Retrieval (top-k by cosine, metadata preserved)
   ↓
Prompt construction (SYSTEM_PROMPT + formatted context)
   ↓
Local LLM (LLMClient → Llama 3 8B via HTTP)
   ↓
Streamlit UI with answer + citations
```

### Key modules

- `src/ingestion/pdf_loader.py`  
  - `load_pdf_pages(path)`: returns a list of `{text, page_number, file_name}` dicts for downstream chunking.

- `src/preprocessing/chunker.py`  
  - `RecursiveChunker`: splits long pages into overlapping chunks while carrying over document metadata (`doc_id`, `page_number`, `file_name`).

- `src/embeddings/embedder.py`  
  - Wraps `sentence-transformers` to embed lists of texts for both documents and queries.
- `src/retrieval/retriever.py`  
  - `ChromaRetriever`: persistent Chroma client with `upsert` and metadata‑aware `query()`.
  - Can format results into a context string with built‑in source annotations.

- `src/llm/prompts.py`  
  - `SYSTEM_PROMPT`: instructs the model to answer *only* from the provided context and to stay concise.  
  - `build_user_prompt(question, context)`: builds the user message sent to the LLM.

- `src/llm/client.py`  
  - `LLMClient`: HTTP client for a local LLM endpoint (e.g. LM Studio), returning structured responses used by the pipeline.

- `src/rag_pipeline/pipeline.py`  
  - `RAGConfig`: configuration for embedding model, LLM model id, vector store path, `top_k`, and temperature.
  - `RAGPipeline.index_chunks(chunks)`: embeds chunks and upserts them into Chroma with full metadata. 
  - `RAGPipeline.answer(question)`:  
    - embeds question → retrieves top‑k chunks → formats traceable context → calls LLM → returns answer, citations, latency, retrieval count.

- `src/ui/app.py`  
  - Streamlit front‑end: upload PDF, trigger indexing, ask questions, inspect sources and metrics.

---

## Getting Started

### 1. Install dependencies

Create and activate a virtual environment, then install requirements:

```
pip install -r requirements.txt
```

Requirements include:

- `streamlit`
- `sentence-transformers`
- `chromadb`
- `requests` (or `httpx` depending on your LLM client)
- `pymupdf` (for PDF parsing)

### 2. Start your local LLM server

Run Llama 3 (or another instruct model) in a tool like **LM Studio** and expose an HTTP endpoint matching the configuration in `src/llm/client.py` and `config/model.yaml`.

Update `config/model.yaml` if needed:

```
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
llm_model: "llama-3-8b-local"
max_context_tokens: 3000
temperature: 0.1
top_k: 4
```

### 3. Launch the Streamlit app

From the project root:

```
export PYTHONPATH=.
streamlit run src/ui/app.py
```

On Windows (PowerShell):

```
$env:PYTHONPATH="."
streamlit run src/ui/app.py
```

Open the printed URL (typically `http://localhost:8501`).

---

## Usage

1. **Upload a PDF**  
   - Use the file uploader in the main page to upload a research paper or technical report.  
   - The app parses it page‑wise, chunks text with metadata, and indexes all chunks into Chroma.
2. **Ask a question**  
   - Enter a question such as “What is the main contribution of this paper?” or “How do they evaluate their method?”.  
   - Click **Generate Insight** to run retrieval + generation.

3. **Inspect answer and sources**  
   - The **Analysis** section shows the synthesized answer from the LLM.
   - Open **View Source Citations** to see every supporting passage with `[Source: file.pdf, Page: n]` and a snippet, allowing manual verification. 
   - At the bottom, check metrics: `Latency` and `Evidence Count`.

4. **Maintenance**  
   - Use the sidebar button **Clear Vector Database** to wipe the Chroma index if you want to start fresh with different documents.

---

## Design Choices & Limitations

- **Local‑first**: The system is designed to work with a locally hosted LLM, which improves data privacy but shifts responsibility for hardware/latency to the user.
- **Semantic‑only retrieval (for now)**: Retrieval uses cosine similarity over MiniLM embeddings; hybrid retrieval (BM25 + semantic) can be added as a future extension.  
- **Page‑level, not sentence‑level citations**: Metadata tracks pages and file names; within‑page highlighting is not yet implemented.  
- **Model quality**: Answer quality depends heavily on the local model; small models can hallucinate, especially on very technical content.

These constraints are documented deliberately to encourage **trustworthy use** and make the project easy to extend in future iterations.

---

## Future Work

- Add keyword/BM25 search and re‑ranking to improve retrieval on long, noisy documents.  
- Implement confidence scoring (e.g. based on similarity scores) and display it alongside answers.  
- Add a small evaluation notebook (RAGAS / manual labels) to quantify faithfulness vs. different chunking and retrieval strategies.  
- Support multiple document collections (e.g. “papers”, “ESG reports”, “internal policies”) with simple filters in the UI.

