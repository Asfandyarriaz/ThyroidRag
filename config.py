import os
from dotenv import load_dotenv

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANR_COLLECTION_NAME = os.getenv("QDRANR_COLLECTION_NAME")

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_NAME="gemini-2.5-flash"

   # Validate required keys
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env!")
    if not QDRANT_API_KEY or not QDRANT_URL:
        raise ValueError("QDRANT_API_KEY or QDRANT_URL not found in .env!")