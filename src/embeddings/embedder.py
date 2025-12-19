import torch
import numpy as np
import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Embedder:
    """
    Optimized Embedder for industrial NLP use cases.
    Supports hardware acceleration and batching for scalability.
    """
    def __init__(self, model_name: str, device: str = None):
        # Auto-detect GPU for "Deep Learning" know-how
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading embedding model '{model_name}' on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embeds texts with optimized batching and normalization.
        
        Args:
            texts: List of strings to encode.
            batch_size: Number of texts to process at once (saves memory).
            normalize: If True, uses Cosine Similarity (standard for RAG).
        """
        if not texts:
            return np.array([])

        logger.info(f"Encoding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100, # Only show for large jobs
            convert_to_numpy=True,
            normalize_embeddings=normalize 
        )
        
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Helper for initializing vector databases correctly."""
        return self.model.get_sentence_embedding_dimension()