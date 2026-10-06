# Install these libraries in the VS Code terminal before running:
# pip install sentence-transformers faiss-cpu chromadb

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import chromadb


# ---------------------------------------------------------
# 1. Sample documents
# ---------------------------------------------------------
# These documents act as our small knowledge base.
# In a real project, these could come from PDFs, websites,
# text files, or a database.

documents = [
    "Machine learning allows computers to learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural Language Processing enables computers to understand human language.",
    "Transformers are neural network architectures widely used in NLP.",
    "Embeddings represent text as numerical vectors.",
    "Vector databases store and search high-dimensional vectors.",
    "FAISS is a library for efficient similarity search.",
    "ChromaDB is a vector database used for storing embeddings.",
    "RAG combines document retrieval with large language models.",
    "Cosine similarity measures how similar two vectors are."
]

print("Number of documents:", len(documents))


# ---------------------------------------------------------
# 2. Load the embedding model
# ---------------------------------------------------------
# The Sentence Transformer converts text into numerical
# vectors called embeddings.
#
# Similar meanings produce vectors that are close to each
# other in the vector space.

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# ---------------------------------------------------------
# 3. Create embeddings for all documents
# ---------------------------------------------------------
# Each document is converted from text into a numerical vector.
# FAISS and ChromaDB will use these vectors for similarity search.

print("\nCreating document embeddings...")

embeddings = model.encode(documents)

# FAISS expects vectors in float32 format.
embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)


# ---------------------------------------------------------
# 4. Create a FAISS vector index
# ---------------------------------------------------------
# The dimension represents the number of values in each
# embedding vector.
#
# IndexFlatL2 performs similarity search using L2 distance.

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Store all document embeddings inside the FAISS index.
index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# ---------------------------------------------------------
# 5. Get the user's search query
# ---------------------------------------------------------
# The user can enter a question or phrase.
# The system will search for documents with similar meaning.

query = input("\nEnter your search query: ")

print("\nYour query:", query)


# ---------------------------------------------------------
# 6. Convert the query into an embedding
# ---------------------------------------------------------
# The query must be converted into the same type of vector
# as the document embeddings before searching.

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")


# ---------------------------------------------------------
# 7. Search the FAISS index
# ---------------------------------------------------------
# k=3 means we want the three most similar documents.

distances, indices = index.search(query_embedding, k=3)

print("\n========== FAISS SEARCH ==========")

for i, idx in enumerate(indices[0]):
    print(f"\nResult {i + 1}")
    print("Document:", documents[idx])
    print("Distance:", distances[0][i])


# ---------------------------------------------------------
# 8. Create a persistent ChromaDB database
# ---------------------------------------------------------
# ChromaDB stores the documents and their embeddings.
# PersistentClient saves the database inside the
# "chroma_db" folder so it can be reused later.

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="semantic_documents"
)

print("\nChromaDB collection created.")


# ---------------------------------------------------------
# 9. Store documents and embeddings in ChromaDB
# ---------------------------------------------------------
# Each document gets a unique ID.
# We also provide the embeddings created earlier.

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings.tolist()
)

print("Documents stored in ChromaDB.")


# ---------------------------------------------------------
# 10. Search ChromaDB
# ---------------------------------------------------------
# We send the query embedding to ChromaDB and ask for
# the three most similar documents.

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

print("\n========== CHROMADB SEARCH ==========")

for i, doc in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}")
    print("Document:", doc)


# ---------------------------------------------------------
# 11. Final comparison
# ---------------------------------------------------------
# Display the documents returned by both vector databases
# so we can compare their semantic search results.

print("\n\n====================================")
print("FINAL SEARCH RESULTS")
print("====================================")

print("\nQuery:", query)

print("\nFAISS Results:")
for idx in indices[0]:
    print("-", documents[idx])

print("\nChromaDB Results:")
for doc in results["documents"][0]:
    print("-", doc)