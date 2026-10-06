# SemanticSEARCH1

A simple semantic search project that demonstrates how to embed text, index it with FAISS, and store/query it with ChromaDB for similarity search.

## Overview

This project uses:

- SentenceTransformers to generate text embeddings
- FAISS for fast vector similarity search
- ChromaDB for persistent vector storage
- A small sample document set to demonstrate semantic retrieval

The script loads a pre-trained embedding model, converts a set of documents into vectors, searches for the most relevant documents for a user query, and compares the results from both FAISS and ChromaDB.

## Features

- Semantic search using sentence embeddings
- Similarity search with FAISS
- Persistent vector storage in ChromaDB
- Sample document corpus for demo/testing
- Query-based retrieval of the top matching results

## Project Structure

- `semantic_search.py` — main script for embedding, indexing, querying, and comparison
- `chroma_db/` — created automatically when the script runs to persist ChromaDB data

## Requirements

Python 3.9+ recommended

Install the required libraries:

```bash
pip install sentence-transformers faiss-cpu chromadb
```

## Run the Project

From the project directory, run:

```bash
python semantic_search.py
```

When prompted, enter a search query such as:

```text
What is machine learning?
```

The script will:

1. Load the embedding model
2. Convert all sample documents into vector embeddings
3. Create a FAISS index
4. Search FAISS for the closest matches
5. Save documents to a persistent ChromaDB collection
6. Query ChromaDB for similar results
7. Print both result sets for comparison

## How It Works

### 1. Document Embedding
The sample documents are converted into dense vector representations using the SentenceTransformer model `all-MiniLM-L6-v2`.

### 2. FAISS Search
The embedding vectors are added to an FAISS index and queried using the input text embedding. The most similar documents are returned based on vector distance.

### 3. ChromaDB Storage
The same embeddings are inserted into a persistent ChromaDB collection so documents can be stored and searched later.

## Example Output

The script prints:

- number of documents
- model loading status
- embedding shape
- FAISS search results
- ChromaDB search results
- final comparison of both search engines

## Notes

- The document list is a sample dataset and can be replaced with your own real content.
- The script is intended as a learning/demo project for semantic retrieval and vector search.
- The ChromaDB database is stored in the local `chroma_db` folder, which is generated automatically.

## Possible Extensions

- Replace the sample documents with PDFs, text files, or database content
- Add a web interface or REST API
- Use a larger embedding model for better quality
- Integrate with LLM-based RAG pipelines

## License

This project is provided for educational and demonstration purposes.
