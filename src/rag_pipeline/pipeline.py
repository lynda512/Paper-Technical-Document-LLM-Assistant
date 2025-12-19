from dataclasses import dataclass
from typing import List, Dict
import time
import numpy as np

from src.embeddings.embedder import Embedder
from src.retrieval.retriever import ChromaRetriever
from src.llm.client import LLMClient
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt

@dataclass
class RAGConfig:
    embedding_model: str
    llm_model: str
    persist_dir: str
    top_k: int = 5

class RAGPipeline:
    def __init__(self, cfg: RAGConfig):
        self.embedder = Embedder(cfg.embedding_model)
        self.retriever = ChromaRetriever(persist_dir=cfg.persist_dir)
        self.llm = LLMClient(cfg.llm_model)
        self.top_k = cfg.top_k

    def index_chunks(self, chunks: List[Dict]):
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"doc_id": c["doc_id"]} for c in chunks]

        # Compute embeddings once here
        embeddings = self.embedder.embed_texts(texts).tolist()
        self.retriever.add_documents(ids, texts, metadatas, embeddings=embeddings)

    def answer(self, question: str) -> Dict:
        start = time.time()

        q_emb = self.embedder.embed_texts([question]).tolist()
        query_result = self.retriever.query(q_emb, n_results=self.top_k)
        docs = query_result["documents"][0]
        metadatas = query_result["metadatas"][0]

        context_blocks = []
        for text, meta in zip(docs, metadatas):
            context_blocks.append(f"[doc_id={meta.get('doc_id')}] {text}")
        context = "\n\n".join(context_blocks)

        user_prompt = build_user_prompt(question, context)
        answer = self.llm.chat(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency_ms = int((time.time() - start) * 1000)

        return {
            "answer": answer,
            "source_docs": docs,
            "metadata": metadatas,
            "latency_ms": latency_ms,
        }
