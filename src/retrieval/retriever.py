from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings

class ChromaRetriever:
    """
    Enhanced Retriever for industrial LLM applications.
    Implements metadata filtering and structured retrieval for higher reliability.
    """
    def __init__(self, persist_dir: str, collection_name: str = "papers"):
        # Using PersistentClient is the modern standard for local storage
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Explicitly defining distance metric for transparency
        )

   
    def add_documents(
    self,
    ids: List[str],
    texts: List[str],
    metadatas: List[Dict],
    embeddings: Optional[List[List[float]]] = None,):
        self.collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    def query(
        self, 
        query_embeddings: List[List[float]], 
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        query_text: Optional[str] = None
    ) -> Dict:
        """
        Enhanced query with metadata filtering.
        AppliedAI values 'trustworthy' results; filtering by source/date helps.
        """
        # If query_text is provided, we can combine it with embeddings for 
        # a more robust 'Hybrid-like' retrieval in the application layer.
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=filter_dict, # Allows filtering by 'page_number' or 'document_type'
        )

    def get_context_with_sources(self, results: Dict) -> str:
        """
        Formats results for the LLM. 
        Critical for the mission of 'transferring know-how' and documentation.
        """
        context_parts = []
        for i in range(len(results['documents'][0])):
            text = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            # Engineering clean citations
            source = meta.get('source', 'Unknown')
            page = meta.get('page', 'N/A')
            context_parts.append(f"[Source: {source}, Page: {page}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)