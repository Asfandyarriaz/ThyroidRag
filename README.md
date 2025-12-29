# Thyroid Cancer RAG Assistant

Thyroid Cancer RAG Assistant is a lightweight chat-style web app for querying a curated **thyroid cancer literature** dataset. It uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant excerpts from a **Qdrant** vector database and generate **evidence-grounded** answers with an LLM.

🌐 **Live App:** https://thyroidraggit-mvsu5jmnjtkvxvve86emk2.streamlit.app/

⚠️ **Disclaimer:** This tool is for research and educational purposes only and does **not** provide medical advice. Always consult clinical guidelines and qualified healthcare professionals for clinical decisions.

---

## Features
- **Chat UI (Streamlit):** ask thyroid-cancer-related questions in a simple chat interface  
- **RAG pipeline:** semantic retrieval over indexed paper excerpts using Qdrant  
- **Evidence-grounded answers:** the model is instructed to use *only* retrieved excerpts  
- **Source-aware citations:** responses cite sources as *(Title, Year)* and optionally include *(PMID)* for quotes  
- **Evidence-level filtering (Level 1–7):** query results can be filtered by medical evidence hierarchy  
- **Confidence rating:** automatically estimates confidence based on retrieved evidence levels  
- **Credibility Check mode:** paste a third-party claim to verify whether the indexed papers support it

---

## Evidence hierarchy (Level 1–7)
Your dataset is categorized by evidence strength (highest → lowest):

1. Clinical practice guidelines / consensus  
2. Systematic reviews / meta-analyses  
3. Randomized controlled trials  
4. Clinical trials (non-randomized)  
5. Cohort studies  
6. Case-control studies  
7. Case reports / case series  

---

## Tech Stack
- **UI:** Streamlit  
- **Vector Database:** Qdrant Cloud  
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)  
- **LLM:** OpenAI (Responses API)

---

## How it works
1. The user submits a question in the chat UI.  
2. The query is embedded using a sentence-transformer model.  
3. Qdrant retrieves the most relevant document chunks (optionally filtered by evidence level).  
4. Retrieved excerpts are packed into a prompt with strict “use only sources” instructions.  
5. The LLM generates an answer grounded in the retrieved excerpts (with citations).

---

## Deployment
Designed for deployment on **Streamlit Community Cloud**.

---

## Acknowledgements
This project is part of an academic workflow exploring RAG-based question answering over thyroid cancer literature.
