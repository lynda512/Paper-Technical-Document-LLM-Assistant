from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from openai import OpenAI
from typing import List, Optional

class DenseRetriever:
    def __init__(
        self,
        collection_name: str,
        qdrant_url: str = "http://localhost:6333",
        openai_model: str = "text-embedding-3-small"
    ):
        self.collection = collection_name
        self.qdrant = QdrantClient(url=qdrant_url)
        self.openai = OpenAI()
        self.model = openai_model

    def embed(self, text: str) -> List[float]:
        response = self.openai.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter: Optional[Filter] = None
    ) -> List[SearchResult]:
        query_vector = self.embed(query)

        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter,
            with_payload=True
        )

        return [
            SearchResult(
                doc_id=str(hit.id),
                text=hit.payload.get("text", ""),
                score=hit.score  # cosine similarity, already in [-1, 1]
            )
            for hit in hits
        ]
