from typing import List, Dict

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    doc_id: str | None = None,
) -> List[Dict]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{start}",
                "text": chunk_text,
            }
        )
        if end == len(words):
            break
        start = max(0, end - chunk_overlap)
    return chunks
