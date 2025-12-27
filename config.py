# config.py
import os


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Set it in Streamlit Secrets / env vars."
        )
    return value


class Config:
    # Collection + models
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "thyroid_cancer_rag")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    # Retrieval
    TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

    # Secrets
    QDRANT_URL = _require_env("QDRANT_URL")
    QDRANT_API_KEY = _require_env("QDRANT_API_KEY")
    OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
