from typing import List, Dict

def reciprocal_rank_fusion(
    ranked_lists: List[List[SearchResult]],
    k: int = 60
) -> List[SearchResult]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists (each already sorted by score desc)
        k: RRF constant. Default 60 works well for large corpora.
           Use 10-20 for small corpora (<200 docs).

    Returns:
        Merged list sorted by RRF score descending.
    """
    rrf_scores: Dict[str, float] = {}
    doc_store: Dict[str, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            doc_id = result.doc_id
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_store[doc_id] = result
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Build final sorted result list
    fused = []
    for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        result = doc_store[doc_id]
        fused.append(SearchResult(
            doc_id=doc_id,
            text=result.text,
            score=score
        ))

    return fused
