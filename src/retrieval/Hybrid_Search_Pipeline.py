from typing import List, Optional
from sentence_transformers import CrossEncoder

class HybridSearchPipeline:
    def __init__(
        self,
        documents: List[dict],
        collection_name: str,
        qdrant_url: str = "http://localhost:6333",
        rrf_k: int = 60,
        reranker_model: Optional[str] = None
    ):
        """
        documents: list of {"id": str, "text": str} used for BM25
        collection_name: Qdrant collection (pre-populated with dense vectors)
        reranker_model: Optional cross-encoder model name for reranking
        """
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(collection_name, qdrant_url)
        self.rrf_k = rrf_k
        self.reranker = CrossEncoder(reranker_model) if reranker_model else None

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = 50
    ) -> List[SearchResult]:
        """
        Run hybrid retrieval: BM25 + Dense → RRF Fusion → optional Reranking.

        retrieval_k: how many results to pull from each retriever before fusion.
                     Should be larger than top_k to give RRF enough candidates.
        """
        # Step 1: Parallel retrieval (in production, run these concurrently)
        sparse_results = self.bm25.search(query, top_k=retrieval_k)
        dense_results = self.dense.search(query, top_k=retrieval_k)

        # Step 2: RRF Fusion
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k
        )

        # Step 3: Optional cross-encoder reranking
        if self.reranker and len(fused) > 0:
            candidates = fused[:min(top_k * 3, len(fused))]
            pairs = [[query, r.text] for r in candidates]
            rerank_scores = self.reranker.predict(pairs)

            reranked = sorted(
                zip(candidates, rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )
            return [r for r, _ in reranked[:top_k]]

        return fused[:top_k]


# Usage
"""pipeline = HybridSearchPipeline(
    documents=my_docs,
    collection_name="my_rag_collection",
    rrf_k=60,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional
)"""

results = pipeline.hybrid_search("ERR_CONN_RESET_4XX retry semantics", top_k=5)
