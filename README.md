# Paper LLM Assistant

RAG-based **assistant** for research papers: upload PDFs, index them, and ask questions like “What is the main contribution?” or “How does this compare to prior work?”.

## Features

- PDF ingestion and chunking for AI/CS papers.
- Vector-store backed retrieval and LLM answering.
- HTTP API for upload and question answering (FastAPI).
- Report and slides that explain use cases for students, researchers, and companies.

## Quickstart

pip install -r requirements.txt # or use pyproject.toml
export OPENAI_API_KEY=...
uvicorn src.ui.app:app --reload

- Open `http://localhost:8000/docs` and:
  - Use `/upload` to upload a paper.
  - Use `/ask` with a question, e.g., “What is the main contribution?”.

## Architecture

- `ingestion/`: PDF parsing with PyMuPDF.
- `preprocessing/`: text cleaning and chunking.
- `embeddings/` and `retrieval/`: vector store (Chroma) for semantic search.
- `llm/`: LLM client and prompts for grounded answers.
- `rag_pipeline/`: end-to-end RAG orchestration.
- `ui/`: FastAPI service for demo and integration.

## Limitations

- Dependent on PDF quality and extraction.
- No citation-level grounding yet.
- Uses external LLM APIs; cost and latency apply.
 set PYTHONPATH=.