from typing import Any, List, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class QdrantVectorStore:
    def __init__(self, client: Any, collection_name: str, embedder: Any = None):
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.embedder is None:
            raise ValueError("Embedder is not initialized.")

        # Use your Embedder.encode_query (returns numpy vector)
        query_vec = self.embedder.encode_query(query_text)

        if isinstance(query_vec, np.ndarray) and query_vec.ndim == 2:
            query_vec = query_vec[0]

        if isinstance(query_vec, np.ndarray):
            query_vec = query_vec.tolist()

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=k,
                with_payload=True,
                with_vectors=False,
            )

            results: List[Dict[str, Any]] = []
            for hit in response.points:
                payload = hit.payload or {}
                results.append({
                    "text": payload.get("text", ""),
                    "title": payload.get("title", ""),
                    "pmid": payload.get("pmid", None),
                    "year": payload.get("year", None),
                    "evidence_level": payload.get("evidence_level", None),
                    "score": getattr(hit, "score", None),
                })
            return results

        except Exception as e:
            logging.error(f"Error during vector search: {e}")
            return []
