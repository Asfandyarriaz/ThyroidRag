from typing import Any, Optional, List, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class QdrantVectorStore:
    def __init__(self, client: Any, collection_name: str, embedder: Any = None):
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns a list of retrieved items with text + metadata for citations.
        """
        if self.embedder is None:
            raise ValueError("Embedder is not initialized. Cannot encode query text.")

        # SentenceTransformers usually uses .encode()
        query_vector = self.embedder.encode(query_text)

        # Normalize output shape
        if isinstance(query_vector, np.ndarray) and query_vector.ndim == 2:
            query_vector = query_vector[0]

        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        response: Any = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        results: List[Dict[str, Any]] = []
        for hit in response.points:
            payload: Dict[str, Any] = hit.payload or {}

            results.append({
                "text": payload.get("text", ""),
                "title": payload.get("title", ""),
                "pmid": payload.get("pmid", None),
                "year": payload.get("year", None),
                "evidence_level": payload.get("evidence_level", None),
                "score": getattr(hit, "score", None),
            })

        return results
