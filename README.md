# Thyroid Cancer RAG Assistant

Thyroid Cancer RAG Assistant is a lightweight chat-style web app for querying a curated thyroid cancer literature dataset. It uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant excerpts from a Qdrant vector database and generate evidence-grounded answers with a large language model.

⚠️ **Disclaimer:** This project is for research and educational purposes only and does **not** provide medical advice. Always consult clinical guidelines and qualified healthcare professionals for clinical decisions.

## Features
- Simple chat UI (Streamlit): type a question and receive an answer
- Retrieval from a Qdrant vector database (semantic search over indexed excerpts)
- Evidence-grounded responses: the model is instructed to answer using only retrieved excerpts
- Source-aware output (metadata such as title/PMID/year can be included in retrieved excerpts)

## Tech Stack
- **UI:** Streamlit
- **Vector Database:** Qdrant Cloud
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM:** OpenAI (Responses API)

## How it works (high level)
1. The user submits a question in the chat UI.
2. The question is embedded using a sentence-transformer embedding model.
3. Qdrant retrieves the top-k most relevant document chunks.
4. The app constructs a prompt containing the retrieved excerpts and strict instructions.
5. The LLM generates an answer grounded in the retrieved excerpts.

## Deployment
This app is designed to be deployed on **Streamlit Community Cloud**
## Acknowledgements
This project is part of an academic workflow to explore RAG-based question answering over thyroid cancer literature.
