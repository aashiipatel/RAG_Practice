# RAG Practice

My first Retrieval-Augmented Generation (RAG) project built using LangChain, ChromaDB/FAISS, Sentence Transformers, and Groq LLMs.

## Project Overview

This project demonstrates how to build a complete RAG (Retrieval-Augmented Generation) pipeline that:

1. Loads documents from multiple file formats
2. Splits documents into chunks
3. Generates embeddings
4. Stores embeddings in a vector database
5. Retrieves relevant context based on user queries
6. Uses an LLM to generate accurate answers from retrieved documents

---

## Features

### Document Loading

Supports:

- PDF (`.pdf`)
- Text Files (`.txt`)
- CSV (`.csv`)
- Excel (`.xlsx`)
- Word Documents (`.docx`)
- JSON (`.json`)

### Text Chunking

Documents are split into smaller chunks using:

- RecursiveCharacterTextSplitter
- Configurable chunk size
- Configurable chunk overlap

### Embeddings

Uses:

- Sentence Transformers
- Model: `all-MiniLM-L6-v2`

Converts text chunks into vector embeddings for semantic search.

### Vector Database

Supports:

- ChromaDB
- FAISS

Stores embeddings for efficient similarity search.

### Retrieval

Performs semantic search to find the most relevant document chunks for a query.

### Generation

Uses Groq LLMs such as:

- Llama 3.1 8B Instant
- Other Groq-supported models

to generate answers based on retrieved context.

---

## Project Structure

```text
RAG_Practice/
│
├── data/
│   ├── pdf/
│   ├── text_files/
│   └── vector_store/
│
├── notebook/
│   ├── document.ipynb
│   ├── pdf_loader.ipynb
│   └── test.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── search.py
│
├── main.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Workflow

### Step 1: Load Documents

```python
docs = load_all_documents("data")
```

Loads all supported documents from the data directory.

### Step 2: Split Documents

```python
chunks = emb_pipe.chunk_documents(docs)
```

Breaks large documents into smaller chunks.

### Step 3: Generate Embeddings

```python
embeddings = emb_pipe.embed_chunks(chunks)
```

Creates vector representations for semantic search.

### Step 4: Store Embeddings

```python
store.build_from_documents(docs)
```

Stores embeddings in ChromaDB or FAISS.

### Step 5: Retrieve Context

```python
results = store.query("What is ABC ID?")
```

Finds the most relevant chunks.

### Step 6: Generate Answer

```python
answer = rag.search_and_summarize(query)
```

Uses the retrieved context and LLM to answer the question.

---

## Example Query

```text
Question:
How to Create ABC ID?
```

The system:

1. Converts the query into embeddings
2. Searches the vector store
3. Retrieves relevant chunks
4. Sends them to the LLM
5. Generates a final response

---

## Technologies Used

- Python
- LangChain
- ChromaDB
- FAISS
- Sentence Transformers
- HuggingFace Embeddings
- Groq API
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aashiipatel/RAG_Practice.git
```

Move into the project:

```bash
cd RAG_Practice
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Project

Run:

```bash
python main.py
```

Or explore the notebooks:

```text
notebook/document.ipynb
notebook/pdf_loader.ipynb
```

---

## Learning Outcomes

Through this project I learned:

- Document preprocessing
- Text chunking strategies
- Embedding generation
- Vector databases
- Semantic search
- Retrieval-Augmented Generation (RAG)
- LangChain integration
- Groq LLM integration
- End-to-end AI application development

---

## Future Improvements

- LangGraph integration
- Agentic RAG workflows
- Hybrid Search
- Re-ranking
- Multi-document reasoning
- Web search integration
- Conversational memory
- Streamlit UI

---

## Author

**Aashi Patel**

Built as part of my journey in Generative AI, RAG Systems, and LLM Application Development.
