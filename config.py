# config.py
import os

class Config:
    # ---- Core RAG settings ----
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "thyroid_cancer_rag")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

    # ---- Qdrant ----
    # Use the base cluster URL (no :6333 for most Qdrant Cloud setups)
    QDRANT_URL = os.getenv("https://1fbcde6b-6683-4da1-b68d-bb08d8acfad7.europe-west3-0.gcp.cloud.qdrant.io")
    QDRANT_API_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sifxaTpo19EUbjz2idlo4QfGJk--eIyPSYD_XtEiZ5A")

    # ---- OpenAI ----
    OPENAI_API_KEY = os.getenv("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")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    # ---- Retrieval ----
    TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

    # ---- Validation ----
    missing = []
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if missing:
        raise ValueError(
            "Missing required environment variables: " + ", ".join(missing)
        )
