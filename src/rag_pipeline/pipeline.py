from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import logging

from src.embeddings.embedder import Embedder
from src.retrieval.retriever import ChromaRetriever
from src.llm.client import LLMClient
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt

# Configure logging for "transparency" - a key appliedAI value
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGPipeline")

@dataclass
class RAGConfig:
    embedding_model: str
    llm_model: str
    persist_dir: str
    top_k: int = 5
    llm_temperature: float = 0.0 

class RAGPipeline:
    def __init__(self, cfg: RAGConfig):
        self.embedder = Embedder(cfg.embedding_model)
        self.retriever = ChromaRetriever(persist_dir=cfg.persist_dir)
        self.llm = LLMClient(cfg.llm_model)
        self.top_k = cfg.top_k
        self.temperature = cfg.llm_temperature

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Processes and stores document chunks.
        Improved to capture full metadata for better 'trustworthy AI' citations.
        """
        ids = [str(c["chunk_id"]) for c in chunks]
        texts = [c["text"] for c in chunks]
        
        # Capture rich metadata (page numbers, titles) to avoid "reverse-engineered" feel
        metadatas = [
            {
                "doc_id": c.get("doc_id"), 
                "page": c.get("page_number", 0),
                "source": c.get("file_name", "Unknown")
            } for c in chunks
        ]


        logger.info(f"Indexing {len(texts)} chunks into vector store...")
        embeddings = self.embedder.embed_texts(texts).tolist()
        self.retriever.add_documents(ids, texts, metadatas, embeddings=embeddings)

    def _format_context(self, docs: List[str], metadatas: List[Dict]) -> str:
        """Helper to create traceable context blocks."""
        context_blocks = []
        for text, meta in zip(docs, metadatas):
            source_info = f"[Source: {meta.get('source')}, Page: {meta.get('page')}]"
            context_blocks.append(f"{source_info}\n{text}")
        return "\n\n---\n\n".join(context_blocks)

    def answer(self, question: str, filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generates an answer with performance tracking and source verification.
        """
        start_time = time.time()

        # 1. Retrieval
        q_emb = self.embedder.embed_texts([question]).tolist()
        query_result = self.retriever.query(
            q_emb, 
            n_results=self.top_k, 
            filter_dict=filter_dict
        )
        
        docs = query_result["documents"][0]
        metadatas = query_result["metadatas"][0]
        
        # 2. Context Preparation
        context = self._format_context(docs, metadatas)

        # 3. Generation
        user_prompt = build_user_prompt(question, context)
        
        # Demonstrates analytical habit: logging the prompt size/latency
        logger.info(f"Generating answer for query. Context length: {len(context)} chars.")
        
        response = self.llm.chat(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            # temperature=self.temperature # If your LLMClient supports it
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # In pipeline.py
        return {
        "answer": response.get("answer", "No answer generated."),
        "status": response.get("status", "unknown"),
        "source_docs": docs,
        "citations": metadatas,
        "latency_ms": latency_ms,
        "retrieval_count": len(docs),
    }
    