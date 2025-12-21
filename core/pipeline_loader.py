# core/pipeline_loader.py
import streamlit as st
from qdrant_client import QdrantClient

from rag.embedder import Embedder
from rag.qa_pipeline import QAPipeline
from rag.llm_client import LLMClient
from rag.vector_store_qdrant import QdrantVectorStore


@st.cache_resource(show_spinner=False)
def init_pipeline(Config):
    # 1) Qdrant client
    qdrant_client = QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY
    )

    # 2) Embedder
    embedder = Embedder(Config.EMBEDDING_MODEL_NAME)

    # 3) Vector store (Qdrant)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=Config.COLLECTION_NAME,
        embedder=embedder
    )

    # 4) LLM client (OpenAI)
    llm_client = LLMClient(
        api_key=Config.OPENAI_API_KEY,
        model=Config.OPENAI_MODEL
    )

    # 5) QA pipeline
    return QAPipeline(embedder, vector_store, llm_client)
