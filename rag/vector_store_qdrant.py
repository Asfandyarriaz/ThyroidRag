# rag/vector_store_qdrant.py
from typing import Any, Dict, List, Optional
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
        Builds a Qdrant filter like:
          evidence_level IN {1,2,3}

        Robustness:
        - Qdrant MatchValue exact match is type-sensitive, so we match:
            int, "int", and sometimes "int.0"
        - Some datasets use Evidence_Level instead of evidence_level.
        """
        if not levels:
            return None

        keys = ["evidence_level", "Evidence_Level"]
        should: List[rest.FieldCondition] = []

        for lvl in levels:
            lvl_int = int(lvl)
            lvl_str = str(lvl_int)
            lvl_str_dot0 = f"{lvl_int}.0"  # handles accidental string "7.0"

            for key in keys:
                # exact match: int
                should.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=lvl_int)))
                # exact match: "int"
                should.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=lvl_str)))
                # exact match: "int.0"
                should.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=lvl_str_dot0)))

        # should = OR
        return rest.Filter(should=should)

    def search(
        self,
        query_text: str,
        k: int = 5,
        levels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        if self.embedder is None:
            raise ValueError("Embedder is not initialized.")

        # Create query embedding
        query_vec = self.embedder.encode_query(query_text)

        # Normalize to Python list
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

            # Newer clients commonly accept query_filter=
            if qfilter is not None:
                kwargs["query_filter"] = qfilter

            try:
                response = self.client.query_points(**kwargs)
            except TypeError:
                # Older clients may expect filter= instead
                if "query_filter" in kwargs:
                    kwargs["filter"] = kwargs.pop("query_filter")
                response = self.client.query_points(**kwargs)

            results: List[Dict[str, Any]] = []
            for hit in response.points:
                payload = hit.payload or {}
                results.append(
                    {
                        "text": payload.get("text", "") or "",
                        "title": payload.get("title", "") or "",
                        "pmid": payload.get("pmid", None),
                        "doi": payload.get("doi", None),
                        "year": payload.get("year", None),
                        "evidence_level": payload.get("evidence_level", payload.get("Evidence_Level", None)),
                        "score": getattr(hit, "score", None),
                    }
                )

            return results

        except Exception as e:
            logger.exception(f"Error during vector search: {e}")
            return []
