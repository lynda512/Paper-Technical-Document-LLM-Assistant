from typing import List, Dict
import logging
import uuid

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """
    Industry-standard chunker that splits text based on a hierarchy 
    to preserve semantic meaning and metadata for citations.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", ". ", " ", ""],
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def chunk_text(
        self,
        text: str,
        doc_metadata: Dict,
    ) -> List[Dict]:
        """
        Splits text and attaches metadata compatible with RAGPipeline.index_chunks.
        """
        # Safely extract metadata for citation traceability
        file_name = doc_metadata.get("source", "unknown")
        page_num = doc_metadata.get("page", 0)
        doc_id = doc_metadata.get("doc_id", file_name)

        split_texts = self._recursive_split(text, self.separators)

        chunks = []
        current_char = 0  # track position to estimate line

        for chunk_content in split_texts:
            # rough line estimate: count newlines up to this chunk
            prefix = text[:current_char]
            approx_line = prefix.count("\n") + 1

            chunks.append(
                {
                    "chunk_id": f"{doc_id}_p{page_num}_{uuid.uuid4().hex[:8]}",
                    "text": chunk_content,
                    # flat metadata so RAGPipeline.index_chunks can read it
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page_number": page_num,
                    "line": approx_line,
                }
            )

            current_char += len(chunk_content)

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Handles the sliding-window splitting logic.
        """
        final_chunks = []
        current_start = 0

        while current_start < len(text):
            end = min(current_start + self.chunk_size, len(text))
            chunk = text[current_start:end]
            final_chunks.append(chunk)

            if end == len(text):
                break

            current_start += (self.chunk_size - self.chunk_overlap)

        return final_chunks
