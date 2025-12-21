# config.py
import os


def _require_env(name: str) -> str:
    """Read an environment variable and fail with a clear error if missing."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Set it in Streamlit Secrets / Render env vars (or your local shell)."
        )
    return value


class Config:
    # -------------------
    # Non-secret settings
    # -------------------
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "thyroid_cancer_rag")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # MiniLM-L6-v2 vector size
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

    # Batch size for embedding + upserts
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

    # Default retrieval size
    TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

    # LLM model name (you can change without code edits)
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    # -------------------
    # Secret settings
    # -------------------
    QDRANT_URL = _require_env("QDRANT_URL")
    QDRANT_API_KEY = _require_env("QDRANT_API_KEY")
    OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
