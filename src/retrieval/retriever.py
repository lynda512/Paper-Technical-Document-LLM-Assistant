from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

class ChromaRetriever:
    def __init__(self, persist_dir: str, collection_name: str = "papers"):
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_dir,
                is_persistent=True,
            )
        )
        # Turn OFF Chroma's own embedding function by using None
        self.collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=None,
        )

    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict],
        embeddings: Optional[List[List[float]]] = None,
    ):
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_embeddings, n_results: int = 5) -> Dict:
        # query_embeddings: List[List[float]]
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
