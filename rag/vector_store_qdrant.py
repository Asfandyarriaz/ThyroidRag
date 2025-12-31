# rag/vector_store_qdrant.py
from typing import Any, List, Dict, Optional
import logging
import numpy as np

from qdrant_client.http import models as rest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, client: Any, collection_name: str, embedder: Any = None):
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    def _build_level_filter(self, levels: Optional[List[int]]) -> Optional[rest.Filter]:
        """
        Evidence level filter (INT ONLY).

        Why int-only?
        - If you indexed payload evidence_level as integer in Qdrant,
          sending a string match (e.g. "1") can cause server-side filter errors.
        """
        if not levels:
            return None

        levels_int = [int(x) for x in levels]

        # OR across levels: evidence_level == any of levels_int
        should = [
            rest.FieldCondition(
                key="evidence_level",
                match=rest.MatchValue(value=lvl),
            )
            for lvl in levels_int
        ]

        return rest.Filter(should=should)

    def search(
        self,
        query_text: str,
        k: int = 5,
        levels: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        if self.embedder is None:
            raise ValueError("Embedder is not initialized.")

        query_vec = self.embedder.encode_query(query_text)

        if isinstance(query_vec, np.ndarray) and query_vec.ndim == 2:
            query_vec = query_vec[0]
        if isinstance(query_vec, np.ndarray):
            query_vec = query_vec.tolist()

        qfilter = self._build_level_filter(levels)

        try:
            kwargs = dict(
                collection_name=self.collection_name,
                query=query_vec,
                limit=k,
                with_payload=True,
                with_vectors=False,
            )

            if qfilter is not None:
                # qdrant-client versions vary: query_filter vs filter
                kwargs["query_filter"] = qfilter

            try:
                response = self.client.query_points(**kwargs)
            except TypeError:
                # fallback for older clients
                if "query_filter" in kwargs:
                    kwargs["filter"] = kwargs.pop("query_filter")
                response = self.client.query_points(**kwargs)

            results: List[Dict[str, Any]] = []
            for hit in response.points:
                payload = hit.payload or {}
                results.append({
                    "text": payload.get("text", "") or "",
                    "title": payload.get("title", "") or "",
                    "pmid": payload.get("pmid", None),
                    "doi": payload.get("doi", None),
                    "year": payload.get("year", None),
                    "evidence_level": payload.get("evidence_level", None),
                    "score": getattr(hit, "score", None),
                })

            logger.info(
                "Qdrant search ok | k=%s | levels=%s | returned=%s",
                k, levels, len(results)
            )
            return results

        except Exception as e:
            logger.exception(f"Error during vector search: {e}")
            return []
