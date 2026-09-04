from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float

class BM25Retriever:
    def __init__(self, documents: List[dict]):
        """
        documents: list of {"id": str, "text": str}
        """
        self.docs = documents
        tokenized = [doc["text"].lower().split() for doc in documents]
        self.index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        tokens = query.lower().split()
        raw_scores = self.index.get_scores(tokens)

        # Get top-k indices
        top_indices = np.argsort(raw_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if raw_scores[idx] > 0:
                results.append(SearchResult(
                    doc_id=self.docs[idx]["id"],
                    text=self.docs[idx]["text"],
                    score=float(raw_scores[idx])
                ))
        return results

    def search_normalized(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Return results with scores normalized to [0, 1]."""
        results = self.search(query, top_k)
        if not results:
            return []
        max_score = max(r.score for r in results)
        if max_score > 0:
            for r in results:
                r.score = r.score / max_score
        return results
