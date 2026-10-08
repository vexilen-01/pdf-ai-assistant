# 📚 PDF AI Assistant

A full-stack, interactive Document Summarization and Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, and **Google Gemini**. 

This assistant automatically generates structured executive summaries upon PDF upload and enables context-aware natural language Q&A across document contents.

---

## ✨ Features & Sub-Features

### 1. Instant Document Summarization
* **Automatic Executive Overviews:** Generates a structured summary highlighting core takeaways immediately upon PDF upload.
* **Context Sampler:** Samples initial document sections to create concise high-level digests without breaching model context windows.

### 2. Context-Aware Q&A (RAG Pipeline)
* **LangChain Expression Language (LCEL):** Implements a modern, declarative LCEL pipeline for flexible document retrieval and answer synthesis.
* **FAISS Vector Indexing:** Fast, in-memory similarity searching over document text chunks using `faiss-cpu`.
* **Grounded Responses:** Prompts enforce factual, context-bound answers to minimize hallucinations.

### 3. Rate-Limit & Performance Optimization
* **Batched Embedding Construction:** Splits text chunks into batches of 15 with deliberate delays (`time.sleep`) to respect Google Free Tier limits (100 requests/minute).
* **Dynamic Chunk Sizing:** Uses optimized text splitter parameters (`chunk_size=2000`, `chunk_overlap=200`) to maximize contextual depth while minimizing API calls.

### 4. Resilient Multi-Key API Management
* **Multi-Key Pooling:** Automatically detects multi-key configurations in environment files or cloud secrets to distribute request load.
* **Fallback Hierarchy:** Resolves API keys through a 3-tier priority structure:
  1. Streamlit Cloud Secrets multi-key list (`GEMINI_KEYS`)
  2. Local environment multi-key string (`GEMINI_KEYS`)
  3. Local single key variable (`GEMINI_API_KEY`)

### 5. Interactive Stateful User Experience
* **Streamlit Session State:** Preserves vector stores and document summaries across query turns to eliminate redundant re-embedding.
* **Clean UI Layout:** Organizes controls into a dedicated sidebar for document processing and a main conversational chat timeline.

---

## 🛠️ Tech Stack & API Endpoints

* **Frontend / Web Framework:** Streamlit
* **LLM Orchestration:** LangChain / LCEL
* **Chat & Summarization Model:** Google Gemini (`gemini-3.8-flash`)
* **Text Embedding Model:** Google Gemini Embeddings (`gemini-embedding-2-preview`)
* **Vector Database:** FAISS (`faiss-cpu`)
* **PDF Parser:** PyPDFLoader (`pypdf`)

---

## 📄 File Structure

```text
.
├── app.py              # Main Streamlit application and RAG pipeline script
├── requirements.txt    # Production Python dependencies
├── .env                # Local environment variables (Git-ignored)
├── .gitignore          # Prevents tracking secret keys and virtual environments
└── README.md           # Project documentation
