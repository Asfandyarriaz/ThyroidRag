# config.py
import os
from typing import Any, Optional


def _get_secret(name: str, default: Optional[Any] = None) -> Optional[Any]:
    """
    Prefer Streamlit secrets if available, otherwise fall back to environment variables.
    Works locally and on Streamlit Cloud.
    """
    try:
        import streamlit as st  # local import to avoid hard dependency in non-streamlit contexts
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets.get(name)
    except Exception:
        pass

    return os.getenv(name, default)


def _require_secret(name: str) -> str:
    value = _get_secret(name)
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"Missing required secret: {name}. "
            f"Set it in Streamlit Cloud → App → Settings → Secrets "
            f"(or as an environment variable locally)."
        )
    return str(value)


class Config:
    """
    Central configuration.
    Use cfg = Config() and read instance attributes.

    Primary names used everywhere:
      - QDRANT_COLLECTION_NAME
      - EMBEDDING_MODEL
      - OPENAI_MODEL
      - QDRANT_URL / QDRANT_API_KEY / OPENAI_API_KEY
    """

    def __init__(self):
        # -------------------
        # Non-secret settings
        # -------------------
        self.QDRANT_COLLECTION_NAME = str(
            _get_secret("QDRANT_COLLECTION_NAME", "thyroid_cancer_rag")
        )
        self.EMBEDDING_MODEL = str(
            _get_secret("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        )

        self.VECTOR_SIZE = int(_get_secret("VECTOR_SIZE", "384"))
        self.BATCH_SIZE = int(_get_secret("BATCH_SIZE", "32"))
        self.TOP_K_DEFAULT = int(_get_secret("TOP_K_DEFAULT", "5"))

        self.OPENAI_MODEL = str(_get_secret("OPENAI_MODEL", "gpt-5-nano"))

        # -------------------
        # Secret settings
        # -------------------
        self.QDRANT_URL = _require_secret("QDRANT_URL")
        self.QDRANT_API_KEY = _require_secret("QDRANT_API_KEY")
        self.OPENAI_API_KEY = _require_secret("OPENAI_API_KEY")

        # -------------------
        # Backward-compatible aliases (optional, but prevents breakage)
        # -------------------
        self.COLLECTION_NAME = self.QDRANT_COLLECTION_NAME
        self.EMBEDDING_MODEL_NAME = self.EMBEDDING_MODEL
