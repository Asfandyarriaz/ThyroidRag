# core/pipeline_loader.py
import streamlit as st
from qdrant_client import QdrantClient

from config import Config
from rag.embedder import Embedder
from rag.qa_pipeline import QAPipeline
from rag.llm_client import LLMClient
from rag.vector_store_qdrant import QdrantVectorStore


@st.cache_resource(show_spinner=False)
def init_pipeline():
    cfg = Config()

    qdrant_client = QdrantClient(
        url=cfg.QDRANT_URL,
        api_key=cfg.QDRANT_API_KEY,
    )

    embedder = Embedder(cfg.EMBEDDING_MODEL)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=cfg.QDRANT_COLLECTION_NAME,
        embedder=embedder,
    )

    llm_client = LLMClient(
        api_key=cfg.OPENAI_API_KEY,
        model=cfg.OPENAI_MODEL,
    )

    return QAPipeline(embedder, vector_store, llm_client)
