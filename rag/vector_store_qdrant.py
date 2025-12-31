# rag/vector_store_qdrant.py
from typing import Any, List, Dict, Optional
import numpy as np
import logging

from qdrant_client.http import models as rest

logging.basicConfig(level=logging.INFO)


class QdrantVectorStore:
    def __init__(self, client: Any, collection_name: str, embedder: Any = None):
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    def _build_level_filter(self, levels: Optional[List[int]]) -> Optional[rest.Filter]:
        """
        Robust filter:
        - no minimum_should_match (not supported in qdrant-client 1.16.x)
        - matches evidence_level stored as int OR string OR float-like
        - supports both keys: 'evidence_level' and 'Evidence_Level'
        """
        if not levels:
            return None

        should: List[rest.FieldCondition] = []

        keys = ["evidence_level", "Evidence_Level"]
        for lvl in levels:
            lvl_int = int(lvl)
            lvl_str = str(lvl_int)
            lvl_float = float(lvl_int)

            for key in keys:
                # match int
                should.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=lvl_int)))
                # match string
                should.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=lvl_str)))
                # match float-ish (only helps if you accidentally stored 7.0)
                should.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=lvl_float)))

        return rest.Filter(should=should)

    def search(self, query_text: str, k: int = 5, levels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
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
                kwargs["query_filter"] = qfilter  # some versions
            try:
                response = self.client.query_points(**kwargs)
            except TypeError:
                # other versions
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
                    "evidence_level": payload.get("evidence_level", payload.get("Evidence_Level", None)),
                    "score": getattr(hit, "score", None),
                })
            return results

        except Exception as e:
            logging.error(f"Error during vector search: {e}")
            return []
