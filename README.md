# Week 3 -> Day 2 -> Retrieval-Augmented Generation (RAG)
---

## Table of Contents

1. [Vector Stores & Retrievers](#vector-stores--retrievers)
    *   [1. Vector Stores](#1-vector-stores)
    *   [2. Retrievers](#2-retrievers)
2. [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    *   [1. What is RAG?](#1-what-is-rag)
    *   [2. Why RAG? - The Problem It Solves](#2-why-rag---the-problem-it-solves)
    *   [3. RAG Components](#3-rag-components)
    *   [4. Flow of RAG - Step by Step](#4-flow-of-rag---step-by-step)
    *   [5. Use Cases of RAG](#5-use-cases-of-rag)
    *   [6. Types of RAG](#6-types-of-rag)
3. [Full RAG Pipeline with LCEL](#full-rag-pipeline-with-lcel)
4. [Assignment: Code Examples for all types of RAG](#assignment)

# Vector Stores & Retrievers

---

## 1. Vector Stores

### 1.1 What is a Vector Store?

A **Vector Store** (also called a Vector Database) is a specialized database designed to store and search **vector embeddings** — numerical representations of data (text, images, audio, etc.).

When you convert a piece of text into a vector embedding, it becomes a list of floating-point numbers (e.g., 384 or 1536 dimensions). These numbers capture the **semantic meaning** of the text. Similar texts will have vectors that are numerically "close" to each other.

```
"I love dogs"   →  [0.12, -0.45, 0.78, 0.33, ...]  (384 numbers)
"I adore pets"  →  [0.11, -0.43, 0.76, 0.31, ...]  (384 numbers) ← very close!
"Stock market"  →  [-0.88, 0.22, -0.10, 0.95, ...] (384 numbers) ← far away
```

A vector store indexes these embeddings so you can quickly find the most **semantically similar** entries to any query.

---

### 1.2 Why Use a Vector Store?

Traditional databases search by **exact keyword matching**. Vector stores search by **meaning**.

| Feature | Traditional DB | Vector Store |
|---|---|---|
| Search type | Exact / keyword | Semantic / meaning |
| Query: "car" finds "automobile"? | ❌ No | ✅ Yes |
| Handles unstructured text? | Limited | ✅ Yes |
| Use case | Structured data | AI / LLM apps |

#### Real-world use cases:
- **RAG (Retrieval-Augmented Generation):** Let an LLM answer questions from your own documents
- **Semantic search:** Search a knowledge base by meaning, not keywords
- **Chatbots with memory:** Store and retrieve past conversations
- **Recommendation systems:** Find similar products, articles, or users
- **Document Q&A:** Ask questions over PDFs, wikis, or internal docs

---

### 1.3 Vector Store Options in LangChain

LangChain integrates with a wide variety of vector stores. Here are the most popular ones:

| Vector Store | Type | Best For |
|---|---|---|
| **FAISS** | Local / In-memory | Prototyping, offline use, no server needed |
| **Chroma** | Local / Server | Local development, open-source projects |
| **Pinecone** | Cloud (managed) | Production, scalable cloud deployments |
| **Weaviate** | Cloud / Self-hosted | Advanced filtering + semantic search |
| **Qdrant** | Cloud / Self-hosted | High-performance production use |
| **Milvus** | Self-hosted | Large-scale enterprise deployments |
| **PGVector** | PostgreSQL extension | If you already use PostgreSQL |
| **Redis** | In-memory | Ultra-low latency retrieval |
| **Elasticsearch** | Self-hosted / Cloud | Hybrid keyword + vector search |
| **OpenSearch** | Self-hosted / Cloud | AWS-native similarity search |

#### For Beginners — Recommended Starting Points:
- **FAISS** → Best for learning. No server, no API key, fully local.
- **Chroma** → Slightly more feature-rich, also local-first.

---

### 1.4 Installation

```bash
# Core LangChain packages
pip install langchain langchain-community

# HuggingFace embeddings
pip install sentence-transformers langchain-huggingface

# FAISS (choose one based on your system)
pip install faiss-cpu       # For CPU (most users)
pip install faiss-gpu       # For GPU (if you have CUDA)
```

---

### 1.5 Simple Example — HuggingFace Embeddings + FAISS + Save Locally

This example:
1. Takes some sample text documents
2. Converts them to vector embeddings using a HuggingFace model
3. Stores those embeddings in a FAISS vector store
4. Saves the FAISS index to disk for later use
5. Loads it back and performs a similarity search

```python
# ============================================================
# STEP 1: Import Required Libraries
# ============================================================
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ============================================================
# STEP 2: Prepare Your Documents
# ============================================================
# These are the texts you want to store and search over.
# In a real app, these could come from PDFs, websites, databases, etc.

documents = [
    Document(
        page_content="LangChain is a framework for building LLM-powered applications.",
        metadata={"source": "langchain_docs", "topic": "intro"}
    ),
    Document(
        page_content="FAISS stands for Facebook AI Similarity Search. It is a library for efficient similarity search.",
        metadata={"source": "faiss_docs", "topic": "faiss"}
    ),
    Document(
        page_content="Vector embeddings are numerical representations of text that capture semantic meaning.",
        metadata={"source": "ml_concepts", "topic": "embeddings"}
    ),
    Document(
        page_content="HuggingFace provides thousands of pre-trained models for NLP tasks.",
        metadata={"source": "huggingface_docs", "topic": "huggingface"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines document retrieval with language model generation.",
        metadata={"source": "rag_paper", "topic": "rag"}
    ),
    Document(
        page_content="Python is a popular programming language widely used in data science and AI.",
        metadata={"source": "python_docs", "topic": "python"}
    ),
]

# ============================================================
# STEP 3: Load HuggingFace Embedding Model
# ============================================================
# 'all-MiniLM-L6-v2' is a lightweight, fast, and high-quality model.
# It converts text into 384-dimensional vectors.
# The model is downloaded automatically on first use (~90MB).

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},   # Use "cuda" if you have a GPU
    encode_kwargs={"normalize_embeddings": True}  # Normalize for cosine similarity
)

print("Embedding model loaded!")

# ============================================================
# STEP 4: Create FAISS Vector Store
# ============================================================
# This step:
# - Converts each document's text into a vector embedding
# - Stores all vectors in a FAISS index for fast similarity search

print("Creating FAISS vector store...")

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

print(f"Vector store created with {len(documents)} documents!")

# ============================================================
# STEP 5: Save FAISS Vector Store to Local Disk
# ============================================================
# FAISS stores two files:
#   - index.faiss  → the actual vector index
#   - index.pkl    → document metadata and text

save_path = "./my_faiss_index"

vector_store.save_local(save_path)
print(f"FAISS index saved to: '{save_path}/'")

# ============================================================
# STEP 6: Load FAISS Vector Store from Disk
# ============================================================
# You can now load this index in a new session without re-creating it!
# Note: allow_dangerous_deserialization=True is required for loading pkl files.

print("Loading FAISS index from disk...")

loaded_vector_store = FAISS.load_local(
    folder_path=save_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True  # Required for loading saved FAISS stores
)

print("FAISS index loaded successfully!")

# ============================================================
# STEP 7: Perform a Similarity Search
# ============================================================
# Search for the top 3 most semantically similar documents to the query.

query = "How does vector search work?"

print(f"\nQuery: '{query}'")
print("=" * 50)

results = loaded_vector_store.similarity_search(
    query=query,
    k=3  # Return top 3 most similar documents
)

# Display results
for i, doc in enumerate(results, 1):
    print(f"\nResult #{i}")
    print(f"  Content : {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")

# ============================================================
# STEP 8: Similarity Search with Scores
# ============================================================
# You can also get the similarity score (lower = more similar in FAISS)

print("\n" + "=" * 50)
print("Search with similarity scores:")

results_with_scores = loaded_vector_store.similarity_search_with_score(
    query=query,
    k=3
)

for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\nResult #{i} | Score: {score:.4f} (lower = more similar)")
    print(f"  Content: {doc.page_content}")
```

#### Expected Output:
```
Loading embedding model...
Embedding model loaded!
Creating FAISS vector store...
Vector store created with 6 documents!
FAISS index saved to: './my_faiss_index/'
Loading FAISS index from disk...
FAISS index loaded successfully!

Query: 'How does vector search work?'
==================================================

Result #1
  Content : Vector embeddings are numerical representations of text that capture semantic meaning.
  Metadata: {'source': 'ml_concepts', 'topic': 'embeddings'}

Result #2
  Content : FAISS stands for Facebook AI Similarity Search. It is a library for efficient similarity search.
  Metadata: {'source': 'faiss_docs', 'topic': 'faiss'}

Result #3
  Content : LangChain is a framework for building LLM-powered applications.
  Metadata: {'source': 'langchain_docs', 'topic': 'intro'}
```

#### Files saved on disk:
```
my_faiss_index/
├── index.faiss     ← Vector index (binary)
└── index.pkl       ← Document texts + metadata (pickle)
```

---

## 2. Retrievers

### 2.1 What is a Retriever?

A **Retriever** is a LangChain abstraction that wraps a vector store and exposes a clean, standardized interface for fetching relevant documents.

**Why use a Retriever instead of calling the vector store directly?**

- ✅ Standardized interface — all retrievers have the same `.invoke()` method
- ✅ Pluggable into LangChain chains (RAG pipelines, agents, etc.)
- ✅ Supports advanced retrieval strategies (MMR, threshold filtering, etc.)
- ✅ Easier to swap out the underlying store without changing your pipeline logic

```
Your Query
    ↓
[ Retriever ]
    ↓
[ Vector Store ] ← FAISS / Chroma / Pinecone / etc.
    ↓
Relevant Documents  →  fed into LLM for answering
```

---

### 2.2 Retriever Types in LangChain

| Retriever Type | Description |
|---|---|
| **VectorStoreRetriever** | Basic similarity search (most common) |
| **MMR Retriever** | Maximal Marginal Relevance — balances relevance & diversity |
| **SelfQueryRetriever** | Uses an LLM to auto-generate filters |
| **MultiQueryRetriever** | Generates multiple query variants for better recall |
| **ContextualCompressionRetriever** | Compresses retrieved docs to only relevant parts |
| **ParentDocumentRetriever** | Retrieves small chunks, returns parent documents |

---

### 2.3 Complete Example — HuggingFace + FAISS + Save + Retriever

This example builds on Section 1 and shows how to use a **Retriever** with `invoke()` (the modern API) and the legacy `get_relevant_documents()`.

```python
# ============================================================
# STEP 1: Imports
# ============================================================
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ============================================================
# STEP 2: Prepare Documents
# ============================================================
documents = [
    Document(
        page_content="LangChain is a framework for building LLM-powered applications.",
        metadata={"source": "langchain_docs", "topic": "intro"}
    ),
    Document(
        page_content="FAISS stands for Facebook AI Similarity Search. It enables fast vector search at scale.",
        metadata={"source": "faiss_docs", "topic": "faiss"}
    ),
    Document(
        page_content="Vector embeddings are numerical representations of text that capture semantic meaning.",
        metadata={"source": "ml_concepts", "topic": "embeddings"}
    ),
    Document(
        page_content="HuggingFace provides thousands of pre-trained transformer models for NLP.",
        metadata={"source": "huggingface_docs", "topic": "huggingface"}
    ),
    Document(
        page_content="RAG stands for Retrieval-Augmented Generation. It grounds LLM answers in real documents.",
        metadata={"source": "rag_paper", "topic": "rag"}
    ),
    Document(
        page_content="A retriever in LangChain fetches relevant documents based on a query.",
        metadata={"source": "langchain_docs", "topic": "retriever"}
    ),
    Document(
        page_content="Cosine similarity measures the angle between two vectors to determine how similar they are.",
        metadata={"source": "math_concepts", "topic": "similarity"}
    ),
]

# ============================================================
# STEP 3: Load Embedding Model
# ============================================================
print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Model loaded!")

# ============================================================
# STEP 4: Build and Save FAISS Vector Store
# ============================================================
print("Building FAISS vector store...")

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

save_path = "./retriever_faiss_index"
vector_store.save_local(save_path)
print(f"Vector store saved to '{save_path}/'")

# ============================================================
# STEP 5: Load FAISS from Disk
# ============================================================
print("Loading FAISS from disk...")

loaded_vector_store = FAISS.load_local(
    folder_path=save_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

print("Loaded successfully!")

# ============================================================
# STEP 6: Create a Retriever from the Vector Store
# ============================================================
# .as_retriever() wraps the vector store in the standard Retriever interface.
#
# search_type options:
#   "similarity"       → plain cosine/L2 similarity (default)
#   "mmr"              → Maximal Marginal Relevance (diverse results)
#   "similarity_score_threshold" → filter by minimum score
#
# search_kwargs:
#   k                  → number of documents to return
#   score_threshold    → minimum score (used with similarity_score_threshold)
#   fetch_k            → candidates to consider for MMR (used with mmr)

retriever = loaded_vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}     # Return top 3 relevant documents
)

# ============================================================
# STEP 7: Retrieve Documents Using .invoke() [Modern API]
# ============================================================
# .invoke() is the current recommended method in LangChain v0.2+

query_1 = "What is RAG and how does it help LLMs?"

print(f"\n{'='*60}")
print(f"Query: '{query_1}'")
print(f"{'='*60}")

results_1 = retriever.invoke(query_1)

for i, doc in enumerate(results_1, 1):
    print(f"\n  Result #{i}")
    print(f"  Content : {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")

# ============================================================
# STEP 8: Using get_relevant_documents() [Legacy API]
# ============================================================
# This is the older method — still works but shows a deprecation warning.
# Internally, it calls .invoke() in newer LangChain versions.

query_2 = "How do embeddings represent text meaning?"

print(f"\n{'='*60}")
print(f"Query (legacy): '{query_2}'")
print(f"{'='*60}")

# Note: In LangChain >= 0.2, use .invoke() instead.
# get_relevant_documents() still works but may show DeprecationWarning.
results_2 = retriever.get_relevant_documents(query_2)

for i, doc in enumerate(results_2, 1):
    print(f"\n  Result #{i}")
    print(f"  Content : {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")

# ============================================================
# STEP 9: MMR Retriever (More Diverse Results)
# ============================================================
# Maximal Marginal Relevance avoids returning redundant/very similar documents.
# Great when your documents have lots of near-duplicates.

mmr_retriever = loaded_vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,        # Final number of documents to return
        "fetch_k": 10, # Candidates to evaluate before selecting diverse ones
        "lambda_mult": 0.5  # 0 = max diversity, 1 = max relevance
    }
)

query_3 = "Tell me about similarity search and vector databases."

print(f"\n{'='*60}")
print(f"MMR Query: '{query_3}'")
print(f"{'='*60}")

results_3 = mmr_retriever.invoke(query_3)

for i, doc in enumerate(results_3, 1):
    print(f"\n  Result #{i}")
    print(f"  Content : {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")

# ============================================================
# STEP 10: Score Threshold Retriever
# ============================================================
# Only return documents with similarity score above a threshold.
# Useful to filter out low-quality matches.

threshold_retriever = loaded_vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.4,  # Only return docs with score >= 0.4
        "k": 5
    }
)

query_4 = "Python programming language"

print(f"\n{'='*60}")
print(f"Threshold Query: '{query_4}'")
print(f"{'='*60}")

results_4 = threshold_retriever.invoke(query_4)

if results_4:
    for i, doc in enumerate(results_4, 1):
        print(f"\n  Result #{i}")
        print(f"  Content : {doc.page_content}")
else:
    print("  No documents met the similarity threshold.")
```

---

### 2.4 `.invoke()` vs `get_relevant_documents()` — Quick Reference

| Method | Status | Notes |
|---|---|---|
| `.invoke(query)` | ✅ Current (recommended) | Use this in LangChain >= 0.2 |
| `.get_relevant_documents(query)` | ⚠️ Deprecated | Still works, shows warning |
| `.aget_relevant_documents(query)` | ⚠️ Deprecated async | Use `.ainvoke()` instead |
| `.ainvoke(query)` | ✅ Current async | For async applications |

```python
# ✅ Recommended
docs = retriever.invoke("your query here")

# ⚠️ Still works but deprecated
docs = retriever.get_relevant_documents("your query here")

# ✅ Async version
docs = await retriever.ainvoke("your query here")
```

---

### 2.5 Adding More Documents to an Existing FAISS Store

```python
# Load existing store
vector_store = FAISS.load_local(
    folder_path="./my_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Add new documents
new_docs = [
    Document(page_content="New document about transformers.", metadata={"topic": "transformers"}),
    Document(page_content="Attention mechanism is the core of transformers.", metadata={"topic": "attention"}),
]

vector_store.add_documents(new_docs)

# Save the updated store
vector_store.save_local("./my_faiss_index")
print("Updated store saved!")
```

---

## 3. Key Concepts Summary

```
Raw Text Documents
      ↓
[ Embedding Model ]  ← HuggingFace all-MiniLM-L6-v2
      ↓
  Vector Embeddings   [0.12, -0.45, 0.78, ...]
      ↓
[ FAISS Vector Store ]  ← Stores + indexes all vectors
      ↓   (saved to disk as index.faiss + index.pkl)
[ Retriever ]          ← .as_retriever(search_type, k)
      ↓
 Relevant Documents    ← based on semantic similarity
      ↓
  [ LLM / Your App ]
```

| Concept | What It Does |
|---|---|
| **Embedding Model** | Converts text → vector numbers |
| **FAISS** | Stores and searches vectors efficiently |
| **`save_local()`** | Saves FAISS index to disk |
| **`load_local()`** | Loads FAISS index from disk |
| **`similarity_search()`** | Direct search on vector store |
| **`as_retriever()`** | Wraps store in standard Retriever API |
| **`.invoke()`** | Fetches relevant docs via retriever |
| **`search_type="mmr"`** | Returns diverse, non-redundant results |

---

## 4. Quick Reference — Common HuggingFace Embedding Models

| Model | Dimensions | Speed | Quality | Best For |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ⚡ Fast | Good | Beginners, prototyping |
| `all-mpnet-base-v2` | 768 | Medium | Better | Production use |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | ⚡ Fast | Good for Q&A | RAG apps |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | ⚡ Fast | Good | Multilingual apps |

---

# Retrieval-Augmented Generation (RAG)

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances a Large Language Model (LLM) by giving it the ability to **look up relevant information from an external knowledge base** before generating a response.

Think of it like this:

> Without RAG → LLM answers from **memory** (training data only, frozen in time)
> With RAG → LLM answers from **memory + a live reference library** (your documents)

### The Core Idea in Plain English

Imagine you ask a doctor a question. A doctor without RAG answers purely from what they studied in medical school years ago. A doctor **with RAG** quickly looks up the latest research paper, reads the relevant section, and then gives you an informed answer grounded in that fresh information.

```
Without RAG:
  User: "What is our company's refund policy?"
  LLM:  "I don't know — this wasn't in my training data." ❌

With RAG:
  User: "What is our company's refund policy?"
  RAG:  [retrieves 'refund_policy.pdf' from your knowledge base]
  LLM:  "According to your policy document, refunds are accepted within 30 days..." ✅
```

---

## 2. Why RAG? - The Problem It Solves

LLMs on their own have several critical limitations:

| Problem | Description | RAG Solution |
|---|---|---|
| **Knowledge Cutoff** | LLMs don't know events after their training date | Retrieve from live/updated documents |
| **Hallucination** | LLMs confidently make up false facts | Ground answers in real retrieved context |
| **No Private Data** | LLMs don't know your internal docs, codebase, policies | Index and retrieve your own data |
| **Context Window Limits** | You can't dump all your documents into one prompt | Retrieve only the *relevant* chunks |
| **Costly Fine-tuning** | Updating an LLM with new knowledge is expensive | Just update the knowledge base |
| **No Source Attribution** | LLMs can't cite where they got information | Retrieved docs provide traceable sources |

### RAG vs. Fine-tuning

| Aspect | RAG | Fine-tuning |
|---|---|---|
| Update knowledge | ✅ Just update the document store | ❌ Retrain the model |
| Cost | ✅ Low | ❌ High (GPU, time, data) |
| Source citation | ✅ Built-in | ❌ Not native |
| Best for | Dynamic, frequently changing data | Changing model behavior/style/tone |
| Transparency | ✅ You see what was retrieved | ❌ Black box |

---

## 3. RAG Components

RAG has three core components that work together:

```
┌─────────────────────────────────────────────────────────┐
│                   RAG Architecture                       │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐   ┌────────────┐ │
│  │ Knowledge    │    │  Retriever   │   │ Generator  │ │
│  │    Base      │───▶│              │──▶│   (LLM)    │ │
│  └──────────────┘    └──────────────┘   └────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 3.1 Knowledge Base

The **Knowledge Base** is the external data source that the RAG system draws from. It is the "library" your LLM will reference at query time.

**What it contains:**
- Your own documents: PDFs, Word files, Markdown files
- Databases, spreadsheets, CSV files
- Web pages, scraped content, RSS feeds
- Code repositories, API documentation
- Chat histories, emails, support tickets

**How it is structured (the indexing pipeline):**

```
Raw Documents
     ↓
[ Document Loader ]     ← Load PDFs, web pages, txt files, etc.
     ↓
[ Text Splitter ]       ← Break large documents into smaller chunks
     ↓
[ Embedding Model ]     ← Convert each chunk to a vector
     ↓
[ Vector Store ]        ← Store & index vectors for fast retrieval
                          (FAISS, Chroma, Pinecone, etc.)
```

**Why split into chunks?**
- LLMs have a limited context window (e.g., 4K, 16K, 128K tokens)
- Searching smaller chunks gives more precise retrieval
- Typical chunk size: 256–1024 tokens, with ~50–100 token overlap to preserve context across boundaries

---

### 3.2 Retriever

The **Retriever** is responsible for finding the most relevant chunks from the knowledge base in response to the user's query.

**How it works:**
1. The user's query is converted into a vector using the same embedding model used during indexing
2. The retriever computes similarity between the query vector and all stored vectors
3. The top-K most similar chunks are returned as context

**Types of retrieval strategies:**

| Strategy | How It Works | Best For |
|---|---|---|
| **Dense Retrieval** | Pure vector similarity (cosine, dot product) | Semantic questions |
| **Sparse Retrieval** | Keyword-based (BM25, TF-IDF) | Exact term matching |
| **Hybrid Retrieval** | Combines dense + sparse | Best of both worlds |
| **MMR** | Maximizes relevance + diversity | Avoiding redundant results |
| **Self-Query** | LLM generates metadata filters automatically | Structured data + semantic |
| **Contextual Compression** | Extracts only the relevant part of each chunk | Reducing noise in context |

**Key retriever parameters:**
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",    # or "mmr", "similarity_score_threshold"
    search_kwargs={
        "k": 4,                  # number of chunks to retrieve
        "score_threshold": 0.6   # minimum relevance score
    }
)
```

---

### 3.3 Generator (LLM)

The **Generator** is the LLM that takes the retrieved context + the user's question and generates a final, grounded answer.

**What it does:**
- Receives a **prompt** that includes: `[System Instructions] + [Retrieved Context] + [User Question]`
- Reads and synthesizes the retrieved chunks
- Produces a coherent, contextually accurate answer
- (Optionally) cites the sources it used

**Example prompt structure sent to the LLM:**

```
System: You are a helpful assistant. Answer the user's question
        using ONLY the provided context. If the answer is not
        in the context, say "I don't know."

Context:
  [Chunk 1]: "Our return policy allows returns within 30 days of purchase..."
  [Chunk 2]: "Items must be in original packaging to qualify for a refund..."
  [Chunk 3]: "Digital products are non-refundable once downloaded..."

User Question: "Can I return a digital product?"
```

**Popular Generator choices:**

| Model | Type | Notes |
|---|---|---|
| GPT-4o / GPT-4 | OpenAI (cloud) | Best quality, paid |
| Claude 3.5 Sonnet | Anthropic (cloud) | Excellent reasoning |
| Gemini 1.5 Pro | Google (cloud) | Large context window |
| Llama 3 | Meta (open-source) | Free, run locally |
| Mistral 7B | Mistral (open-source) | Lightweight, fast |
| Phi-3 | Microsoft (open-source) | Very small, efficient |

---

## 4. Flow of RAG - Step by Step

### 4.1 High-Level Flow

```
User Question
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE                         │
│                                                            │
│  Question ──▶ [Embed Question] ──▶ Query Vector           │
│                                         │                  │
│  Knowledge Base ◀── [Vector Store] ◀───┘                  │
│  (indexed docs)       similarity search                    │
│                            │                               │
│                    Top-K Relevant Chunks                   │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│                   GENERATION PHASE                         │
│                                                            │
│  [System Prompt]                                           │
│       +                                                    │
│  [Retrieved Chunks as Context]  ──▶  [LLM Generator]      │
│       +                                      │             │
│  [User Question]                             ▼             │
│                                       Final Answer         │
└────────────────────────────────────────────────────────────┘
     │
     ▼
Final Answer (grounded in your documents)
```

---

### 4.2 Detailed Step-by-Step Flow

#### 🔵 OFFLINE PHASE — Indexing (done once, or periodically updated)

```
Step 1: LOAD
  Raw documents (PDF, TXT, HTML, CSV, etc.)
        ↓
  DocumentLoader → List of Document objects

Step 2: SPLIT
  Large documents → Small, overlapping chunks
  (e.g., 500 tokens per chunk, 50 token overlap)

Step 3: EMBED
  Each chunk → embedding model → dense vector
  e.g., "Our return policy..." → [0.12, -0.45, 0.78, ...]

Step 4: STORE
  All vectors + their text → stored in Vector Store (FAISS, Chroma, etc.)
  Saved to disk for reuse
```

#### 🟢 ONLINE PHASE — Query Time (happens on every user request)

```
Step 5: RECEIVE QUERY
  User asks: "What is the return policy for digital goods?"

Step 6: EMBED QUERY
  Query → same embedding model → query vector
  "What is the return policy..." → [0.11, -0.43, 0.76, ...]

Step 7: RETRIEVE
  Compare query vector against all stored vectors
  Return top-K most similar document chunks (e.g., k=4)

Step 8: AUGMENT PROMPT
  Construct a prompt:
  ┌─────────────────────────────────────────────┐
  │ System: Answer using the context below...   │
  │ Context: [chunk1] [chunk2] [chunk3] [chunk4]│
  │ Question: What is the return policy for...? │
  └─────────────────────────────────────────────┘

Step 9: GENERATE
  LLM reads the prompt → produces grounded answer

Step 10: RETURN ANSWER
  "Digital products are non-refundable once downloaded,
   as stated in our return policy document."
```

---

### 4.3 RAG Flow in LangChain Code (Minimal Example)

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI   # or any other LLM

# ── 1. Load the saved vector store (from previous session) ──
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "./my_faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# ── 2. Create a Retriever ────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── 3. Define the RAG Prompt Template ────────────────────────
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the
context provided below. If the answer is not in the context,
say "I don't have that information."

Context:
{context}

Question: {question}

Answer:
""")

# ── 4. Initialize the LLM (Generator) ────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── 5. Build the RAG Chain ────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,   # retrieve → format
        "question": RunnablePassthrough()     # pass question through unchanged
    }
    | prompt        # fill the template
    | llm           # generate the answer
    | StrOutputParser()  # parse output to plain string
)

# ── 6. Ask a Question ─────────────────────────────────────────
question = "What is RAG and why is it useful?"
answer = rag_chain.invoke(question)

print(f"Question: {question}")
print(f"Answer:   {answer}")
```

---

## 5. Use Cases of RAG

RAG is the dominant architecture for any application where an LLM needs to answer questions grounded in specific, real, or private data.

### 5.1 Enterprise & Business

| Use Case | Description |
|---|---|
| **Internal Knowledge Base Q&A** | Employees ask questions answered from company wikis, HR docs, SOPs |
| **Customer Support Chatbots** | Bot answers from product manuals, FAQs, support tickets |
| **Legal Document Analysis** | Lawyers query contracts, case law, and regulatory filings |
| **HR Policy Assistant** | Employees ask about leave policies, benefits, onboarding steps |
| **Sales Enablement** | Reps query product specs, competitor comparisons, pricing sheets |
| **IT Helpdesk Automation** | Auto-resolve tickets by searching internal runbooks |

### 5.2 Healthcare & Research

| Use Case | Description |
|---|---|
| **Medical Literature Q&A** | Doctors query PubMed papers, clinical guidelines, drug interactions |
| **Clinical Trial Search** | Find relevant trials based on patient criteria |
| **Research Paper Summarization** | Ask questions across hundreds of research papers |
| **Drug Information System** | Pharmacists query dosage, side effects, contraindications |

### 5.3 Education & Learning

| Use Case | Description |
|---|---|
| **Textbook Q&A** | Students ask questions answered from course materials |
| **Personalized Tutoring** | Tutor answers from the student's own notes and syllabus |
| **E-learning Platforms** | Interactive Q&A grounded in course content |
| **Code Documentation Assistant** | Developers ask questions about a codebase or SDK docs |

### 5.4 Software & Development

| Use Case | Description |
|---|---|
| **Codebase Q&A** | "Where is the payment logic handled?" → searches the repo |
| **API Documentation Assistant** | Answer "how do I use endpoint X?" from API docs |
| **Incident Post-mortem Analysis** | Search historical incidents, logs, runbooks |
| **Software Review Summarization** | Aggregate and query user reviews from app stores |

### 5.5 Finance & Government

| Use Case | Description |
|---|---|
| **Financial Report Analysis** | Query earnings reports, 10-Ks, analyst notes |
| **Regulatory Compliance** | Answer "does this action violate regulation X?" from legal text |
| **Government Policy Q&A** | Citizens query legislation and public policies |
| **Tax Document Assistant** | Answer tax questions grounded in official IRS/HMRC guidelines |

---

## 6. Types of RAG

RAG has evolved significantly. Here are the major variants:

---

### 6.1 Naive RAG (Basic RAG)

The simplest form. Described in this document so far.

```
Query → Retrieve top-K chunks → Stuff into prompt → LLM generates answer
```

**Strengths:** Simple, fast to implement
**Weaknesses:** Retrieval quality directly bottlenecks answer quality. No mechanism to verify or improve retrieval.

---

### 6.2 Advanced RAG

Enhances naive RAG with better pre-retrieval, retrieval, and post-retrieval steps.

```
Pre-retrieval improvements:
  - Better chunking strategies (semantic chunking, hierarchical chunking)
  - Query rewriting / expansion before retrieval
  - HyDE: Generate a hypothetical answer, then retrieve using THAT

Post-retrieval improvements:
  - Re-ranking: Use a cross-encoder to re-score retrieved chunks
  - Contextual compression: Trim irrelevant parts of retrieved chunks
  - Reciprocal Rank Fusion: Merge results from multiple retrievers
```

**Key techniques in Advanced RAG:**

| Technique | What It Does |
|---|---|
| **Query Rewriting** | Rephrase the query to improve retrieval |
| **HyDE** | Generate a hypothetical ideal answer, use it to retrieve |
| **Step-back Prompting** | Ask a broader question to retrieve more context |
| **Re-ranking** | Use a cross-encoder to re-order retrieved chunks by relevance |
| **Hybrid Search** | Combine vector search + BM25 keyword search |
| **Contextual Compression** | Strip irrelevant sentences from retrieved chunks |

---

### 6.3 Modular RAG

Treats each RAG component as an independent, swappable module. You can mix and match components (retrievers, rerankers, generators) freely.

```
Query
  ↓
[Query Transformer Module]    ← rewrite / expand / decompose
  ↓
[Router Module]               ← decide WHICH knowledge base to search
  ↓
[Retriever Module(s)]         ← one or many retrievers in parallel
  ↓
[Reranker Module]             ← score and sort retrieved results
  ↓
[Generator Module]            ← LLM generates the final answer
  ↓
[Response Validator Module]   ← check answer quality
```

---

### 6.4 Self-RAG

**Self-RAG** teaches the LLM to **decide for itself** whether retrieval is even necessary, and to **critique its own output** after generation.

The model generates special reflection tokens:

| Token | Meaning |
|---|---|
| `[Retrieve]` | Should I retrieve documents? YES or NO |
| `[ISREL]` | Is the retrieved document relevant? |
| `[ISSUP]` | Is the generated text supported by the retrieved doc? |
| `[ISUSE]` | Is the response useful to the user? |

**Flow:**
```
User Query
    ↓
LLM decides: [Retrieve = YES / NO]
    ↓ (if YES)
Retriever fetches documents
    ↓
LLM generates answer segments
    ↓
LLM critiques: [ISREL] [ISSUP] [ISUSE]
    ↓
Best segment selected as final answer
```

**Why it's powerful:** The LLM avoids unnecessary retrieval for simple questions, and self-corrects when retrieved documents are irrelevant or unsupported.

---

### 6.5 Corrective RAG (CRAG)

**Corrective RAG** adds an automatic **correction mechanism** when retrieval quality is poor.

```
User Query
    ↓
Retrieve documents
    ↓
[Retrieval Evaluator]   ← LLM grades each retrieved doc
    ↓
Three possible outcomes:
  ✅ CORRECT   → confidence is HIGH → use documents as-is
  ⚠️ AMBIGUOUS → confidence is MEDIUM → refine + also web search
  ❌ INCORRECT → confidence is LOW → discard, do a web search instead
    ↓
[Knowledge Refinement]  ← strip irrelevant sentences from docs
    ↓
LLM generates final answer
```

**Key innovation:** It doesn't blindly trust the retrieved documents. If retrieval fails, it falls back to web search automatically.

---

### 6.6 Fusion RAG (RAG-Fusion)

**Fusion RAG** generates **multiple queries** from the original question, retrieves documents for each, then fuses the results using **Reciprocal Rank Fusion (RRF)**.

```
Original Query: "How does attention work in transformers?"
    ↓
[LLM Query Generator]  ←  generates N alternative queries:
  - "Explain self-attention mechanism"
  - "Transformer architecture attention head"
  - "Scaled dot-product attention formula"
    ↓
Run retrieval for EACH query in parallel
    ↓
[Reciprocal Rank Fusion]  ← merge + re-rank all results
    ↓
Top documents passed to LLM for final answer
```

**Why it's powerful:**
- Overcomes vocabulary mismatch (one phrasing might retrieve better)
- Produces more diverse and comprehensive retrieval coverage
- RRF consistently outperforms any single query retrieval

---

### 6.7 Speculative RAG

**Speculative RAG** uses a **small specialist LLM** to draft answers from retrieved documents, then a **large generalist LLM** to verify and finalize the answer.

```
User Query
    ↓
Retrieve documents
    ↓
[Small Specialist LLM]  ← generates a draft answer from retrieved context
       (faster, cheaper)
    ↓
[Large Generalist LLM]  ← verifies, corrects, and polishes the draft
       (slower, more capable)
    ↓
Final high-quality answer
```

**Why it's powerful:**
- Reduces expensive LLM calls (specialist does the heavy lifting)
- The large LLM only needs to verify, not read all documents
- Balances quality and cost efficiently

---

### 6.8 Agentic RAG

**Agentic RAG** gives the retrieval system **tool-use and reasoning capabilities**. The LLM acts as an agent that decides when, what, and how to retrieve — iteratively.

```
User Query
    ↓
[LLM Agent]
    ↓ (decides which tool to use)
    ├── search_vector_store("query")
    ├── search_web("query")
    ├── query_sql_database("SELECT ...")
    ├── call_api("https://...")
    └── read_file("report.pdf")
    ↓
Agent reviews results, decides if more retrieval is needed
    ↓
Iterates until it has enough information
    ↓
Synthesizes final answer
```

**Why it's powerful:**
- Handles complex multi-hop questions ("Who founded the company that made the tool used in X?")
- Can query multiple data sources in sequence
- Self-determines when it has enough context to answer

---

### 6.9 Graph RAG

**Graph RAG** stores knowledge as a **knowledge graph** (entities + relationships) rather than plain text chunks, enabling complex relational reasoning.

```
Plain RAG:
  "Apple acquired Beats Electronics in 2014"
  → stored as a text chunk

Graph RAG:
  Node: Apple (Company)
  Node: Beats Electronics (Company)
  Edge: ACQUIRED (year=2014)
  → stored as structured graph relationships
```

**Why it's powerful:**
- Answers multi-hop questions ("What companies did Apple acquire before 2015?")
- Understands entity relationships, not just text similarity
- Combines graph traversal + semantic search

---

## 7. Comparison of RAG Types

| RAG Type | Key Innovation | Complexity | Best For |
|---|---|---|---|
| **Naive RAG** | Basic retrieve + generate | ⭐ Low | Quick prototypes |
| **Advanced RAG** | Better chunking, reranking, HyDE | ⭐⭐ Medium | Production apps |
| **Modular RAG** | Swappable component architecture | ⭐⭐⭐ High | Large flexible systems |
| **Self-RAG** | LLM self-critiques retrieval + output | ⭐⭐⭐ High | High accuracy needs |
| **Corrective RAG** | Auto-corrects poor retrieval | ⭐⭐ Medium | Unreliable knowledge bases |
| **Fusion RAG** | Multi-query + RRF fusion | ⭐⭐ Medium | Diverse, comprehensive retrieval |
| **Speculative RAG** | Small LLM drafts, large LLM verifies | ⭐⭐⭐ High | Cost-efficient high quality |
| **Agentic RAG** | LLM agent with multiple tools | ⭐⭐⭐⭐ Very High | Multi-source complex queries |
| **Graph RAG** | Knowledge graph + semantic search | ⭐⭐⭐⭐ Very High | Relational multi-hop questions |

---

## 8. Summary — The Big Picture

```
                     ┌─────────────────────────────────────────┐
                     │         THE RAG ECOSYSTEM               │
                     │                                         │
  Your Documents ───▶│  Knowledge Base (Vector Store / Graph)  │
                     │                                         │
  User Question ────▶│  Retriever (finds relevant context)     │
                     │         +                               │
                     │  Generator (LLM synthesizes answer)     │
                     │                                         │
                     └──────────────────┬──────────────────────┘
                                        │
                                        ▼
                               Grounded, Accurate Answer
                               with Source Attribution

  Evolution:
  Naive RAG → Advanced RAG → Modular RAG
       ↓
  Self-RAG / Corrective RAG  (smarter retrieval decisions)
       ↓
  Fusion RAG / Speculative RAG  (better coverage & efficiency)
       ↓
  Agentic RAG / Graph RAG  (multi-source, relational reasoning)
```

### Golden Rules of RAG

1. **Garbage in, garbage out** — your retrieval quality limits your answer quality
2. **Chunk wisely** — too small loses context, too large adds noise
3. **Always cite sources** — makes the system trustworthy and debuggable
4. **Evaluate retrieval separately from generation** — they fail in different ways
5. **Start simple** — Naive RAG first, then add complexity only where needed

---

*Next in your LangChain journey → Building a full RAG pipeline end-to-end with LangChain LCEL (LangChain Expression Language) 🦜🔗*


# Full RAG Pipeline with LCEL
## (LangChain Expression Language)

---

## 1. What is LCEL?

**LCEL (LangChain Expression Language)** is LangChain's modern, declarative way to **compose chains** by connecting components together using the **pipe operator `|`**.

Think of it like Unix pipes — the output of one component flows directly into the input of the next.

```bash
# Unix pipes — output of one command feeds the next
cat file.txt | grep "error" | sort | uniq

# LCEL — output of one LangChain component feeds the next
retriever | format_docs | prompt | llm | output_parser
```

### Before LCEL (Old Way — LangChain v0.1)

```python
# Verbose, rigid, hard to customize
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
result = chain.run("What is RAG?")
```

### After LCEL (New Way — LangChain v0.2+)

```python
# Clean, readable, fully composable
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = chain.invoke("What is RAG?")
```

---

## 2. Why LCEL?

| Feature | Old Chains | LCEL |
|---|---|---|
| Readability | ❌ Buried in class internals | ✅ Visual, readable left-to-right |
| Streaming | ❌ Complex to add | ✅ Built-in `.stream()` |
| Async support | ❌ Separate implementation | ✅ Built-in `.ainvoke()` |
| Batch processing | ❌ Manual loops | ✅ Built-in `.batch()` |
| Debugging | ❌ Hard to inspect intermediate steps | ✅ Easy with `RunnableLambda` |
| Composability | ❌ Fixed chain structures | ✅ Freely mix and compose |
| Parallel execution | ❌ Not native | ✅ `RunnableParallel` |

---

## 3. Core LCEL Primitives

Before building the full pipeline, understand these building blocks:

### 3.1 The Pipe Operator `|`

```python
chain = component_A | component_B | component_C

# Equivalent to:
output_A = component_A.invoke(input)
output_B = component_B.invoke(output_A)
output_C = component_C.invoke(output_B)
```

### 3.2 Key Runnable Classes

| Class | Purpose | Example |
|---|---|---|
| `RunnablePassthrough` | Pass input through unchanged | Forward the question as-is |
| `RunnableParallel` | Run multiple chains in parallel | Retrieve + pass question simultaneously |
| `RunnableLambda` | Wrap any Python function | Custom formatting, parsing |
| `RunnableBranch` | Conditional routing | Route to different chains based on input |
| `RunnableMap` | Dict of runnables run in parallel | Same as `RunnableParallel` |

### 3.3 LCEL Invocation Methods

```python
chain = prompt | llm | parser

# Single input
result   = chain.invoke({"question": "What is RAG?"})

# Async single input
result   = await chain.ainvoke({"question": "What is RAG?"})

# Batch (multiple inputs at once)
results  = chain.batch([{"question": "Q1"}, {"question": "Q2"}])

# Streaming (token by token)
for chunk in chain.stream({"question": "What is RAG?"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"question": "What is RAG?"}):
    print(chunk, end="", flush=True)
```

---

## 4. Full RAG Pipeline — Setup & Installation

```bash
# Install all required packages
pip install langchain langchain-community langchain-core
pip install langchain-huggingface sentence-transformers
pip install langchain-openai          # for OpenAI LLM (or use any other)
pip install faiss-cpu                 # vector store
pip install pypdf                     # for loading PDF files
pip install tiktoken                  # token counting for text splitting
```

### Project File Structure

```
rag_project/
│
├── documents/                  ← Your source documents go here
│   ├── company_handbook.pdf
│   ├── product_manual.pdf
│   └── faq.txt
│
├── faiss_index/                ← Saved vector store (auto-created)
│   ├── index.faiss
│   └── index.pkl
│
├── 01_indexing.py              ← Step 1: Build the knowledge base
├── 02_rag_chain.py             ← Step 2: Build and run the RAG chain
└── 03_advanced_rag.py          ← Step 3: Advanced LCEL techniques
```

---

## 5. Step 1 — Indexing Pipeline (Build the Knowledge Base)

```python
# 01_indexing.py
# ================================================================
# PURPOSE: Load documents → Split → Embed → Store in FAISS
# Run this ONCE (or whenever your documents change)
# ================================================================

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── STEP 1A: Load Documents ──────────────────────────────────────

print("📄 Loading documents...")

# Option A: Load a single PDF
pdf_loader = PyPDFLoader("documents/company_handbook.pdf")
pdf_docs = pdf_loader.load()

# Option B: Load a plain text file
txt_loader = TextLoader("documents/faq.txt", encoding="utf-8")
txt_docs = txt_loader.load()

# Option C: Load ALL files from a directory
dir_loader = DirectoryLoader(
    path="documents/",
    glob="**/*.pdf",        # load all PDFs recursively
    loader_cls=PyPDFLoader
)
all_docs = dir_loader.load()

# Option D: Load from a web URL
web_loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
web_docs = web_loader.load()

# Combine all documents into one list
raw_documents = pdf_docs + txt_docs
print(f"   Loaded {len(raw_documents)} raw document pages/chunks.")

# ── STEP 1B: Split Documents into Chunks ─────────────────────────
#
# WHY split?
#   - LLMs have limited context windows
#   - Smaller chunks = more precise retrieval
#   - chunk_size: tokens/chars per chunk (sweet spot: 500–1000)
#   - chunk_overlap: overlap between chunks to preserve boundary context

print("✂️  Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # each chunk is ~500 characters
    chunk_overlap=100,       # 100-char overlap between consecutive chunks
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]   # try splitting at these, in order
)

chunks = text_splitter.split_documents(raw_documents)

print(f"   Created {len(chunks)} chunks from {len(raw_documents)} documents.")
print(f"   Sample chunk:\n   '{chunks[0].page_content[:200]}...'")

# ── STEP 1C: Load Embedding Model ───────────────────────────────

print("🤖 Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("   Embedding model ready!")

# ── STEP 1D: Create & Save FAISS Vector Store ───────────────────
#
# This converts each chunk → vector, then stores them in FAISS.
# This can take a few minutes for large document sets.

print("💾 Building FAISS vector store...")

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

vectorstore.save_local("faiss_index")
print(f"   ✅ FAISS index saved! ({len(chunks)} vectors indexed)")
print("   Files created: faiss_index/index.faiss + faiss_index/index.pkl")
```

---

## 6. Step 2 — RAG Chain with LCEL (Core Pipeline)

```python
# 02_rag_chain.py
# ================================================================
# PURPOSE: Load the index → Build LCEL RAG chain → Ask questions
# ================================================================

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# ── STEP 2A: Load Saved Vector Store ────────────────────────────

print("📦 Loading FAISS index from disk...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

print("   ✅ Vector store loaded!")

# ── STEP 2B: Create Retriever ────────────────────────────────────

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}    # return top 4 most relevant chunks
)

# ── STEP 2C: Define the RAG Prompt Template ──────────────────────
#
# {context} will be filled with retrieved document chunks
# {question} will be filled with the user's actual question

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful, knowledgeable assistant. Your job is to answer
the user's question based ONLY on the provided context below.

Rules:
- If the answer is clearly present in the context, answer it directly.
- If the context is partially helpful, use what's there and say what's missing.
- If the context contains NO relevant information, say:
  "I couldn't find that information in the provided documents."
- Never make up facts that aren't in the context.
- Keep your answer clear and concise.

─────────────────────────────────────────────────
CONTEXT:
{context}
─────────────────────────────────────────────────

QUESTION: {question}

ANSWER:
""")

# ── STEP 2D: Initialize the LLM (Generator) ─────────────────────

# Using OpenAI — swap for any LangChain-compatible LLM:
# from langchain_anthropic import ChatAnthropic       → Claude
# from langchain_google_genai import ChatGoogleGenerativeAI → Gemini
# from langchain_community.llms import Ollama         → Local Llama3

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,        # 0 = deterministic/factual (best for RAG)
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── STEP 2E: Helper Function — Format Retrieved Docs ────────────

def format_docs(docs):
    """
    Convert a list of Document objects into a single formatted string.
    Each document chunk is numbered and separated for clarity.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "")
        header = f"[Source {i}: {source}" + (f", page {page}]" if page else "]")
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)

# ── STEP 2F: Build the LCEL RAG Chain ───────────────────────────
#
# CHAIN ANATOMY:
#
#   Input: "What is our refund policy?"
#      ↓
#   RunnableParallel runs TWO things simultaneously:
#     ├── "context":  retriever → format_docs  (fetches relevant chunks)
#     └── "question": RunnablePassthrough()    (passes question unchanged)
#      ↓
#   rag_prompt   → fills {context} and {question} into the template
#      ↓
#   llm          → generates the answer
#      ↓
#   StrOutputParser → extracts plain text from LLM response object

rag_chain = (
    RunnableParallel({
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ── STEP 2G: Run the Chain ───────────────────────────────────────

def ask(question: str) -> str:
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    print(f"{'='*60}")
    answer = rag_chain.invoke(question)
    print(f"💬 Answer:\n{answer}")
    return answer

# Ask questions!
ask("What is the company's work-from-home policy?")
ask("How many days of annual leave do employees get?")
ask("What are the steps for the performance review process?")
```

---

## 7. Step 3 — Advanced LCEL Techniques

### 7.1 Streaming Responses (Token by Token)

```python
# 03_advanced_rag.py — Part 1: Streaming

print("🔴 Streaming answer (token by token):\n")

question = "Explain the onboarding process for new employees."

# .stream() yields tokens as they are generated — great for chat UIs
for token in rag_chain.stream(question):
    print(token, end="", flush=True)

print("\n")  # newline after streaming completes
```

### 7.2 Returning Sources Alongside the Answer

```python
# Part 2: Return answer + source documents used

from langchain_core.runnables import RunnableParallel

# Build a chain that returns BOTH the answer AND the source docs
rag_chain_with_sources = RunnableParallel({
    "answer":   rag_chain,
    "sources":  retriever                # also return the raw docs
})

result = rag_chain_with_sources.invoke("What is the sick leave policy?")

print(f"Answer:\n{result['answer']}\n")
print("Sources used:")
for i, doc in enumerate(result["sources"], 1):
    source = doc.metadata.get("source", "Unknown")
    page   = doc.metadata.get("page", "N/A")
    print(f"  [{i}] {source} — page {page}")
    print(f"       Excerpt: {doc.page_content[:120]}...")
```

### 7.3 Batch Processing Multiple Questions

```python
# Part 3: Batch — ask multiple questions at once (runs in parallel)

questions = [
    "What is the company's refund policy?",
    "How do I submit an expense report?",
    "What are the working hours?",
    "Who do I contact for IT support?",
]

print("⚡ Batch processing questions...\n")

# batch() runs all inputs simultaneously for efficiency
answers = rag_chain.batch(questions)

for q, a in zip(questions, answers):
    print(f"Q: {q}")
    print(f"A: {a}\n")
    print("-" * 50)
```

### 7.4 Adding Chat History — Conversational RAG

```python
# Part 4: Conversational RAG with memory

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# ── Step A: Contextualise the question using chat history ────────
#
# If user says "tell me more about that" — the LLM reformulates it
# as a standalone question ("tell me more about the refund policy")
# so it can be retrieved properly.

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and the latest user question,
    reformulate the question as a standalone question that can be understood
    without the chat history. Do NOT answer — just reformulate if needed,
    otherwise return as-is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# This chain reformulates follow-up questions
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# ── Step B: RAG prompt that includes chat history ────────────────

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using
    ONLY the context provided. If you don't know, say so.

    Context:
    {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ── Step C: Retriever that considers chat history ────────────────

def contextualized_retriever(input_dict):
    """
    If there's chat history, reformulate the question first.
    Otherwise, retrieve using the raw question.
    """
    if input_dict.get("chat_history"):
        # Reformulate the question in context of chat history
        standalone_q = contextualize_q_chain.invoke(input_dict)
        return retriever.invoke(standalone_q)
    else:
        return retriever.invoke(input_dict["input"])

from langchain_core.runnables import RunnableLambda

# ── Step D: Full conversational RAG chain ────────────────────────

conversational_rag_chain = (
    RunnableParallel({
        "context":      RunnableLambda(contextualized_retriever) | format_docs,
        "input":        RunnablePassthrough(),
        "chat_history": lambda x: x.get("chat_history", [])
    })
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ── Step E: Simulate a multi-turn conversation ───────────────────

chat_history = []

def chat(question: str):
    global chat_history

    print(f"\n👤 User: {question}")

    response = conversational_rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    print(f"🤖 Assistant: {response}")

    # Append to history for next turn
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response

# Multi-turn conversation example:
chat("What are the annual leave policies?")
chat("How many days are for sick leave specifically?")   # follow-up — uses context
chat("Can I carry over unused days to next year?")       # another follow-up
```

### 7.5 Custom Retrieval with RunnableLambda

```python
# Part 5: Inject custom logic anywhere in the chain

from langchain_core.runnables import RunnableLambda

# Preprocess the question before retrieval
def preprocess_question(question: str) -> str:
    """Clean and normalize the question."""
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    return question

# Log which documents were retrieved
def log_and_format(docs):
    print(f"\n📚 Retrieved {len(docs)} chunks:")
    for i, doc in enumerate(docs, 1):
        print(f"   [{i}] {doc.metadata.get('source', '?')} — {doc.page_content[:80]}...")
    return format_docs(docs)

# Custom chain with preprocessing and logging
custom_rag_chain = (
    RunnableParallel({
        "context":  (
            RunnableLambda(lambda x: x["question"])   # extract question
            | RunnableLambda(preprocess_question)      # clean it
            | retriever                                # retrieve
            | RunnableLambda(log_and_format)           # log + format
        ),
        "question": RunnableLambda(lambda x: x["question"])
    })
    | rag_prompt
    | llm
    | StrOutputParser()
)

result = custom_rag_chain.invoke({"question": "What is the maternity leave policy"})
print(f"\nAnswer: {result}")
```

### 7.6 Async RAG Chain (for Web Servers / FastAPI)

```python
# Part 6: Async — for use in FastAPI, Django async views, etc.

import asyncio

async def ask_async(question: str) -> str:
    print(f"\n⚡ [Async] Question: {question}")
    answer = await rag_chain.ainvoke(question)
    print(f"💬 [Async] Answer: {answer}")
    return answer

async def stream_async(question: str):
    print(f"\n🔴 [Async Stream] Question: {question}\n")
    async for token in rag_chain.astream(question):
        print(token, end="", flush=True)
    print()

# Run async examples
asyncio.run(ask_async("What is the company mission statement?"))
asyncio.run(stream_async("Summarize the employee code of conduct."))
```

---

## 8. Complete End-to-End Pipeline — All Together

```
OFFLINE (run once):

  Your PDFs / TXTs / Web pages
         ↓
  [ DirectoryLoader / PyPDFLoader / WebBaseLoader ]
         ↓
  [ RecursiveCharacterTextSplitter ]   chunk_size=500, overlap=100
         ↓
  [ HuggingFaceEmbeddings ]            all-MiniLM-L6-v2
         ↓
  [ FAISS.from_documents() ]
         ↓
  [ vectorstore.save_local("faiss_index") ]


ONLINE (every user query):

  User Question: "What is the leave policy?"
         │
         ▼
  ┌──────────────────────────────────────────────────────┐
  │            LCEL RAG CHAIN                            │
  │                                                      │
  │  RunnableParallel({                                  │
  │    "context":  retriever | format_docs,              │
  │    "question": RunnablePassthrough()                 │
  │  })                                                  │
  │       ↓                                              │
  │  ChatPromptTemplate  ← fills {context} + {question} │
  │       ↓                                              │
  │  ChatOpenAI (or any LLM)                             │
  │       ↓                                              │
  │  StrOutputParser                                     │
  └──────────────────────────────────────────────────────┘
         │
         ▼
  "Employees are entitled to 20 days of annual leave per year,
   as stated in Section 4 of the Employee Handbook."
```

---

## 9. LCEL Chain Debugging

```python
# Use .with_config() to add verbose tracing
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    tags=["rag-debug"],
    metadata={"user_id": "user_123"}
)

# Add intermediate step inspection with RunnableLambda
from langchain_core.runnables import RunnableLambda

def debug_step(name):
    def _debug(input_val):
        print(f"\n🔍 [{name}] Input type: {type(input_val).__name__}")
        if isinstance(input_val, str):
            print(f"   Value: {input_val[:100]}")
        elif isinstance(input_val, dict):
            print(f"   Keys: {list(input_val.keys())}")
        elif isinstance(input_val, list):
            print(f"   Length: {len(input_val)}")
        return input_val  # pass through unchanged
    return RunnableLambda(_debug)

# Insert debug steps anywhere in the chain
debug_chain = (
    RunnableParallel({
        "context":  retriever | debug_step("after retriever") | format_docs,
        "question": RunnablePassthrough()
    })
    | debug_step("after parallel")
    | rag_prompt
    | debug_step("after prompt")
    | llm
    | StrOutputParser()
)

debug_chain.invoke("What is the remote work policy?")
```

---

## 10. Swapping LLMs — Model Flexibility

One of LCEL's greatest strengths is that you can **swap the LLM** without changing any other part of the chain:

```python
# ── OpenAI ──────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Anthropic Claude ─────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

# ── Google Gemini ────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# ── Local Llama3 via Ollama (completely free, runs offline) ──────
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")   # run: `ollama pull llama3` first

# ── HuggingFace Inference API ────────────────────────────────────
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token="your_hf_token"
)

# The rest of the RAG chain stays EXACTLY the same ✅
rag_chain = (
    RunnableParallel({
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | rag_prompt
    | llm             # ← just swap this line
    | StrOutputParser()
)
```

---

## 11. Quick Reference — LCEL Cheat Sheet

```python
# ── Imports ─────────────────────────────────────────────────────
from langchain_core.runnables import (
    RunnablePassthrough,      # pass input unchanged
    RunnableParallel,         # run multiple chains simultaneously
    RunnableLambda,           # wrap any Python function
    RunnableBranch,           # conditional routing
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Chain patterns ───────────────────────────────────────────────

# 1. Basic sequential chain
chain = prompt | llm | parser

# 2. RAG chain
chain = (
    RunnableParallel({"context": retriever | fmt, "question": RunnablePassthrough()})
    | prompt | llm | parser
)

# 3. Custom function in chain
chain = RunnableLambda(my_func) | llm | parser

# 4. Conditional routing
chain = RunnableBranch(
    (lambda x: "urgent" in x["question"], urgent_chain),
    default_chain
)

# ── Invocation ───────────────────────────────────────────────────
chain.invoke(input)                        # single, synchronous
chain.batch([input1, input2, input3])      # multiple, parallel
chain.stream(input)                        # streaming, sync
await chain.ainvoke(input)                 # single, async
await chain.abatch([input1, input2])       # multiple, async
async for chunk in chain.astream(input):   # streaming, async
    print(chunk, end="")
```

---

## 12. Summary

| Phase | Tools Used | Purpose |
|---|---|---|
| **Loading** | `PyPDFLoader`, `TextLoader`, `WebBaseLoader` | Ingest raw documents |
| **Splitting** | `RecursiveCharacterTextSplitter` | Break into retrievable chunks |
| **Embedding** | `HuggingFaceEmbeddings` | Convert text → vectors |
| **Storing** | `FAISS.from_documents()` + `save_local()` | Index and persist vectors |
| **Retrieving** | `vectorstore.as_retriever()` | Find relevant chunks at query time |
| **Prompting** | `ChatPromptTemplate` | Format context + question for the LLM |
| **Generating** | `ChatOpenAI`, `Ollama`, `Claude`, etc. | Produce the grounded answer |
| **Parsing** | `StrOutputParser` | Extract clean text from LLM response |
| **Composing** | LCEL `\|` operator + `RunnableParallel` | Wire all components together |
| **Serving** | `.stream()`, `.ainvoke()`, `.batch()` | Flexible deployment patterns |

---


# Assignment

## Project Setup with `uv` 

> `uv` is a blazing-fast Python package manager written in Rust — it replaces `pip`, `venv`, and `pip-tools` in one tool.

### Step 1 — Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
# uv 0.5.x (or later)
```

---

### Step 2 — Create Project Folder

```bash
# Create and navigate into your project folder
mkdir langchain-rag-types
cd langchain-rag-types
```

---

### Step 3 — Initialise the `uv` Project

```bash
# Initialise a new uv Python project
# This creates: pyproject.toml, .python-version, .venv/, hello.py
uv init

# Your folder structure now looks like:
# langchain-rag-types/
# ├── .venv/              ← virtual environment (auto-created)
# ├── .python-version     ← pins the Python version
# ├── pyproject.toml      ← project metadata + dependencies
# └── hello.py            ← starter file (you can delete this)
```

---

### Step 4 — Pin Python Version (Recommended)

```bash
# Use Python 3.11 (recommended for LangChain)
uv python pin 3.11

# Verify
uv python list
```

---

### Step 5 — Add All Required Dependencies

```bash
# Create a virtual environment
uv venv

# Activate it
.\.venv\Scripts\activate           # Windows

# Add all LangChain + RAG dependencies in one command
uv add langchain langchain-community langchain-core langchain-openai langchain-huggingface faiss-cpu sentence-transformers networkx duckduckgo-search python-dotenv

# uv add \
#   langchain \
#   langchain-community \
#   langchain-core \
#   langchain-openai \
#   langchain-huggingface \
#   faiss-cpu \
#   sentence-transformers \
#   networkx \
#   duckduckgo-search \
#   python-dotenv

# uv automatically:
#   ✅ Resolves dependency versions
#   ✅ Creates/updates pyproject.toml
#   ✅ Creates uv.lock (reproducible installs)
#   ✅ Installs into .venv/
```

---

### Step 6 — Set Your API Key

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."

# Or create a .env file in your project root:
echo 'OPENAI_API_KEY="sk-..."' > .env
```

---

### Step 7 — Create Project Files

```bash
# Create all the Python files for this guide
New-Item -Path . -Name shared_setup.py -ItemType File
New-Item -Path . -Name 01_naive_rag.py -ItemType File
New-Item -Path . -Name 02_advanced_rag.py -ItemType File
New-Item -Path . -Name 03_modular_rag.py -ItemType File
New-Item -Path . -Name 04_self_rag.py -ItemType File
New-Item -Path . -Name 05_corrective_rag.py -ItemType File
New-Item -Path . -Name 06_fusion_rag.py -ItemType File
New-Item -Path . -Name 07_speculative_rag.py -ItemType File
New-Item -Path . -Name 08_agentic_rag.py -ItemType File
New-Item -Path . -Name 09_graph_rag.py -ItemType File

# touch shared_setup.py
# touch 01_naive_rag.py
# touch 02_advanced_rag.py
# touch 03_modular_rag.py
# touch 04_self_rag.py
# touch 05_corrective_rag.py
# touch 06_fusion_rag.py
# touch 07_speculative_rag.py
# touch 08_agentic_rag.py
# touch 09_graph_rag.py
```

Your final project structure:

```
langchain-rag-types/
├── .venv/                  ← virtual environment (managed by uv)
├── .python-version         ← Python 3.11
├── pyproject.toml          ← dependencies declared here
├── uv.lock                 ← locked, reproducible dependency tree
├── shared_setup.py         ← common setup used by all examples
├── 01_naive_rag.py
├── 02_advanced_rag.py
├── 03_modular_rag.py
├── 04_self_rag.py
├── 05_corrective_rag.py
├── 06_fusion_rag.py
├── 07_speculative_rag.py
├── 08_agentic_rag.py
└── 09_graph_rag.py
```

---

### Step 8 — Run Any File

```bash
# uv run automatically uses the project's .venv — no manual activation needed
uv run shared_setup.py
uv run 01_naive_rag.py
uv run 02_advanced_rag.py
# ... and so on
```

### Quick `uv` Command Reference

```bash
uv init                        # initialise a new project
uv add <package>               # add a dependency
uv add <package> --dev         # add a dev-only dependency
uv remove <package>            # remove a dependency
uv sync                        # install all deps from uv.lock
uv run <script.py>             # run a script inside the .venv
uv python pin 3.11             # pin Python version
uv lock                        # regenerate the lock file
uv tree                        # show dependency tree
```

---

## Shared Setup (Used by All Examples)

```python
# shared_setup.py — run this first, reused across all RAG type examples

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv
import os


load_dotenv()
# ── Sample documents (shared across all examples) ───────────────
docs = [
    Document(page_content="LangChain is a framework for building LLM applications.", metadata={"source": "langchain_docs"}),
    Document(page_content="FAISS is a library for fast similarity search developed by Facebook AI.", metadata={"source": "faiss_docs"}),
    Document(page_content="RAG combines retrieval from a knowledge base with LLM generation.", metadata={"source": "rag_paper"}),
    Document(page_content="Vector embeddings capture the semantic meaning of text as numbers.", metadata={"source": "ml_concepts"}),
    Document(page_content="HuggingFace provides open-source pre-trained transformer models.", metadata={"source": "hf_docs"}),
    Document(page_content="Retrieval-Augmented Generation reduces hallucinations in LLMs.", metadata={"source": "rag_paper"}),
    Document(page_content="Self-RAG allows the LLM to decide when retrieval is necessary.", metadata={"source": "self_rag_paper"}),
    Document(page_content="Corrective RAG evaluates retrieved documents and falls back to web search.", metadata={"source": "crag_paper"}),
]

# ── Embedding model ──────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ── Vector store ─────────────────────────────────────────────────
vectorstore = FAISS.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── LLM ──────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Utility: format retrieved docs ───────────────────────────────
def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] ({d.metadata.get('source','?')}): {d.page_content}"
        for i, d in enumerate(docs)
    )

print("✅ Shared setup complete!")
```

Run it:
```bash
uv run shared_setup.py
# ✅ Shared setup complete!
```

---

## 1. Naive RAG ⭐

> **Pattern:** Query → Retrieve top-K chunks → Stuff into prompt → Generate answer.
> No query transformation. No post-processing. Simplest possible form.

```python
# 01_naive_rag.py

from shared_setup import retriever, llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If not in context, say "I don't know."

Context: {context}
Question: {question}
Answer:
""")

# The entire chain in one line — classic Naive RAG
naive_rag_chain = (
    RunnableParallel({
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# ── Run ──────────────────────────────────────────────────────────
question = "What is RAG and why is it useful?"
answer   = naive_rag_chain.invoke(question)

print(f"Q: {question}")
print(f"A: {answer}")

# Output:
# A: RAG stands for Retrieval-Augmented Generation. It combines retrieval
#    from a knowledge base with LLM generation to reduce hallucinations.
```

Run it:
```bash
uv run 01_naive_rag.py
```

**What's happening:**
```
"What is RAG?"
      ↓
retriever.invoke("What is RAG?")   → top-3 docs
      ↓
prompt.format(context=..., question=...)
      ↓
llm.invoke(filled_prompt)
      ↓
"RAG reduces hallucinations by grounding LLMs in real documents."
```

---

## 2. Advanced RAG ⭐⭐

> **Pattern:** Improves Naive RAG with **query rewriting**, **HyDE**, and **re-ranking**.
> Better retrieval quality before and after the vector search step.

```python
# 02_advanced_rag.py

from shared_setup import retriever, llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Technique A: Query Rewriting ─────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_template("""
You are an expert at reformulating search queries.
Rewrite the question below to improve document retrieval.
Return ONLY the rewritten question, nothing else.

Original question: {question}
Rewritten question:
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# ── Technique B: HyDE (Hypothetical Document Embedding) ──────────
# Generate a hypothetical ideal answer, then retrieve using THAT
# (hypothetical doc is closer in embedding space to real docs)

hyde_prompt = ChatPromptTemplate.from_template("""
Write a short, ideal reference document (2-3 sentences) that would
perfectly answer the following question. Write it as if it's a factual
reference document, not a direct answer.

Question: {question}
Ideal reference document:
""")

hyde_chain = hyde_prompt | llm | StrOutputParser()

# ── Technique C: Reranking ────────────────────────────────────────
rerank_prompt = ChatPromptTemplate.from_template("""
On a scale of 1-10, how relevant is this document to the question?
Return ONLY the number.

Question: {question}
Document: {document}
Score:
""")

def rerank_docs(input_dict):
    """Score each retrieved doc and return sorted by relevance."""
    question = input_dict["question"]
    docs     = input_dict["docs"]
    scored   = []
    for doc in docs:
        score_str = (rerank_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "document": doc.page_content
        })
        try:
            score = float(score_str.strip())
        except ValueError:
            score = 0.0
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]

# ── Full Advanced RAG Chain ───────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using the context below.

Context: {context}
Question: {question}
Answer:
""")

def advanced_rag(question: str) -> str:
    rewritten_q      = rewrite_chain.invoke({"question": question})
    print(f"   Rewritten query: '{rewritten_q}'")

    hypothetical_doc = hyde_chain.invoke({"question": question})
    print(f"   HyDE doc: '{hypothetical_doc[:80]}...'")

    retrieved_docs   = retriever.invoke(hypothetical_doc)
    reranked_docs    = rerank_docs({"question": question, "docs": retrieved_docs})

    context = format_docs(reranked_docs)
    return (rag_prompt | llm | StrOutputParser()).invoke({
        "context": context, "question": question
    })

# ── Run ──────────────────────────────────────────────────────────
question = "How do vector databases help AI systems?"
print(f"Q: {question}\n")
print(f"A: {advanced_rag(question)}")
```

Run it:
```bash
uv run 02_advanced_rag.py
```

---

## 3. Modular RAG ⭐⭐⭐

> **Pattern:** Each component (retriever, prompt, generator) is a **swappable module**.
> A router classifies each query and selects the right module combination.

```python
# 03_modular_rag.py

from shared_setup import vectorstore, llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Module 1: Query Router ────────────────────────────────────────
router_prompt = ChatPromptTemplate.from_template("""
Classify the question into one of these categories:
  - "factual"    : asking for a specific fact or definition
  - "conceptual" : asking to explain a concept or idea
  - "comparison" : asking to compare two or more things

Return ONLY the category word.

Question: {question}
Category:
""")

router_chain = router_prompt | llm | StrOutputParser()

# ── Module 2: Retrievers (different strategies per query type) ────
factual_retriever    = vectorstore.as_retriever(search_kwargs={"k": 2})
conceptual_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
comparison_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.3}
)

# ── Module 3: Generators (different prompts per query type) ───────
factual_prompt    = ChatPromptTemplate.from_template(
    "Give a direct, factual answer.\nContext: {context}\nQ: {question}\nA:"
)
conceptual_prompt = ChatPromptTemplate.from_template(
    "Explain clearly with examples.\nContext: {context}\nQ: {question}\nExplanation:"
)
comparison_prompt = ChatPromptTemplate.from_template(
    "Compare and contrast with a table if helpful.\nContext: {context}\nQ: {question}\nComparison:"
)

# ── Full Modular RAG ──────────────────────────────────────────────
def modular_rag(question: str) -> str:
    category = router_chain.invoke({"question": question}).strip().lower()
    print(f"   Query category: '{category}'")

    active_retriever = {"factual": factual_retriever, "conceptual": conceptual_retriever,
                        "comparison": comparison_retriever}.get(category, conceptual_retriever)
    active_prompt    = {"factual": factual_prompt, "conceptual": conceptual_prompt,
                        "comparison": comparison_prompt}.get(category, conceptual_prompt)

    context = format_docs(active_retriever.invoke(question))
    return (active_prompt | llm | StrOutputParser()).invoke({
        "context": context, "question": question
    })

# ── Run ──────────────────────────────────────────────────────────
print(modular_rag("What is FAISS?"))                          # → factual
print(modular_rag("Explain how RAG reduces hallucinations"))  # → conceptual
print(modular_rag("Compare FAISS and vector embeddings"))     # → comparison
```

Run it:
```bash
uv run 03_modular_rag.py
```

---

## 4. Self-RAG ⭐⭐⭐

> **Pattern:** The LLM **decides if retrieval is needed**, retrieves only when necessary,
> then **self-critiques** the retrieved docs and generated answer before returning it.

```python
# 04_self_rag.py

from shared_setup import retriever, llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Token 1: Should I retrieve? ──────────────────────────────────
retrieve_decision_prompt = ChatPromptTemplate.from_template("""
Given the question, decide if external document retrieval is needed.
Answer ONLY "YES" or "NO".

Retrieval IS needed for: specific facts, domain knowledge, recent info.
Retrieval is NOT needed for: greetings, general knowledge, simple math.

Question: {question}
Retrieve?
""")

# ── Token 2: Is retrieved doc relevant? ──────────────────────────
relevance_prompt = ChatPromptTemplate.from_template("""
Is this document relevant to answering the question?
Answer ONLY "RELEVANT" or "IRRELEVANT".

Question: {question}
Document: {document}
Relevance:
""")

# ── Token 3: Is the answer supported by the document? ────────────
support_prompt = ChatPromptTemplate.from_template("""
Is this answer fully supported by the provided document?
Answer ONLY "SUPPORTED" or "UNSUPPORTED".

Document: {document}
Answer:   {answer}
Support:
""")

answer_prompt = ChatPromptTemplate.from_template(
    "Answer using the context.\nContext: {context}\nQ: {question}\nA:"
)
direct_prompt = ChatPromptTemplate.from_template(
    "Answer this from your own knowledge.\nQ: {question}\nA:"
)

def self_rag(question: str) -> str:
    # Step 1: Decide if retrieval is needed
    should_retrieve = (retrieve_decision_prompt | llm | StrOutputParser()).invoke(
        {"question": question}
    ).strip().upper()
    print(f"   [RETRIEVE token]: {should_retrieve}")

    if should_retrieve == "NO":
        print("   → Answering without retrieval")
        return (direct_prompt | llm | StrOutputParser()).invoke({"question": question})

    # Step 2: Retrieve + filter relevant docs
    relevant_docs = []
    for doc in retriever.invoke(question):
        relevance = (relevance_prompt | llm | StrOutputParser()).invoke({
            "question": question, "document": doc.page_content
        }).strip().upper()
        print(f"   [ISREL token]: {relevance} — '{doc.page_content[:60]}...'")
        if relevance == "RELEVANT":
            relevant_docs.append(doc)

    if not relevant_docs:
        print("   → No relevant docs found, answering from knowledge")
        return (direct_prompt | llm | StrOutputParser()).invoke({"question": question})

    # Step 3: Generate answer
    context = format_docs(relevant_docs)
    answer  = (answer_prompt | llm | StrOutputParser()).invoke({
        "context": context, "question": question
    })

    # Step 4: Verify support
    support = (support_prompt | llm | StrOutputParser()).invoke({
        "document": context, "answer": answer
    }).strip().upper()
    print(f"   [ISSUP token]: {support}")

    if support == "UNSUPPORTED":
        print("   → Regenerating conservatively")
        conservative = ChatPromptTemplate.from_template(
            "Answer ONLY with facts from the context.\nContext: {context}\nQ: {question}\nA:"
        )
        answer = (conservative | llm | StrOutputParser()).invoke({
            "context": context, "question": question
        })

    return answer

# ── Run ──────────────────────────────────────────────────────────
print(self_rag("What is 2 + 2?"))           # → NO retrieval
print(self_rag("How does Self-RAG work?"))  # → YES retrieval
```

Run it:
```bash
uv run 04_self_rag.py
```

---

## 5. Corrective RAG (CRAG) ⭐⭐

> **Pattern:** After retrieval, an **evaluator grades each document**.
> Low-confidence results trigger a **web search fallback** automatically.

```python
# 05_corrective_rag.py

from shared_setup import retriever, llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

web_search = DuckDuckGoSearchRun()

eval_prompt = ChatPromptTemplate.from_template("""
Score how well this document answers the question.
Return ONLY one word: "HIGH", "MEDIUM", or "LOW"

Question: {question}
Document: {document}
Confidence:
""")

refine_prompt = ChatPromptTemplate.from_template("""
From the document below, extract ONLY the sentences directly relevant
to the question. Remove everything else.

Question: {question}
Document: {document}
Relevant sentences:
""")

answer_prompt = ChatPromptTemplate.from_template(
    "Answer accurately.\nContext: {context}\nQ: {question}\nA:"
)

def corrective_rag(question: str) -> str:
    retrieved_docs = retriever.invoke(question)

    scores = []
    for doc in retrieved_docs:
        score = (eval_prompt | llm | StrOutputParser()).invoke({
            "question": question, "document": doc.page_content
        }).strip().upper()
        scores.append(score)
        print(f"   [EVAL]: {score} — '{doc.page_content[:60]}...'")

    high_count = scores.count("HIGH")
    low_count  = scores.count("LOW")

    if high_count >= 1:
        # ✅ CORRECT — refine and use good docs
        print("   → Action: CORRECT")
        good_docs = [d for d, s in zip(retrieved_docs, scores) if s in ("HIGH", "MEDIUM")]
        refined   = [(refine_prompt | llm | StrOutputParser()).invoke({
            "question": question, "document": doc.page_content
        }) for doc in good_docs]
        context = "\n\n".join(refined)

    elif low_count == len(scores):
        # ❌ INCORRECT — fall back to web search
        print("   → Action: INCORRECT (web search fallback)")
        context = f"[Web Search Result]:\n{web_search.run(question)}"

    else:
        # ⚠️ AMBIGUOUS — use docs + supplement with web
        print("   → Action: AMBIGUOUS (docs + web)")
        context = f"{format_docs(retrieved_docs)}\n\n[Web Supplement]:\n{web_search.run(question)}"

    return (answer_prompt | llm | StrOutputParser()).invoke({
        "context": context, "question": question
    })

# ── Run ──────────────────────────────────────────────────────────
print(corrective_rag("What is corrective RAG?"))
print(corrective_rag("Who won the 2024 US presidential election?"))  # → web fallback
```

Run it:
```bash
uv run 05_corrective_rag.py
```

---

## 6. Fusion RAG ⭐⭐

> **Pattern:** Generate **N alternative queries** from the original question,
> retrieve for each, then merge results using **Reciprocal Rank Fusion (RRF)**.

```python
# 06_fusion_rag.py

from shared_setup import retriever, llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections import defaultdict

multi_query_prompt = ChatPromptTemplate.from_template("""
Generate {n} different search queries to retrieve documents
that answer the question from different angles.
Return ONLY the queries, one per line, no numbering.

Question: {question}
Queries:
""")

def generate_queries(question: str, n: int = 3) -> list[str]:
    raw = (multi_query_prompt | llm | StrOutputParser()).invoke({
        "question": question, "n": n
    })
    queries = [q.strip() for q in raw.strip().split("\n") if q.strip()]
    print(f"   Generated {len(queries)} queries:")
    for q in queries:
        print(f"     - {q}")
    return queries[:n]

def reciprocal_rank_fusion(results_lists: list, k: int = 60) -> list:
    """Fuse multiple ranked lists into one using RRF scoring."""
    scores  = defaultdict(float)
    doc_map = {}
    for results in results_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id           = doc.page_content
            scores[doc_id]  += 1.0 / (k + rank)
            doc_map[doc_id]  = doc
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    print(f"   RRF: {sum(len(r) for r in results_lists)} total docs → {len(sorted_ids)} unique")
    return [doc_map[did] for did in sorted_ids]

answer_prompt = ChatPromptTemplate.from_template(
    "Answer comprehensively.\nContext: {context}\nQ: {question}\nA:"
)

def fusion_rag(question: str) -> str:
    queries     = generate_queries(question, n=3)
    all_results = [retriever.invoke(q) for q in queries]
    fused_docs  = reciprocal_rank_fusion(all_results)
    context     = format_docs(fused_docs[:4])
    return (answer_prompt | llm | StrOutputParser()).invoke({
        "context": context, "question": question
    })

# ── Run ──────────────────────────────────────────────────────────
print(fusion_rag("How does RAG improve language model responses?"))
```

Run it:
```bash
uv run 06_fusion_rag.py
```

**Why RRF works:**
```
Query 1 results:  [doc_A #1, doc_B #2, doc_C #3]
Query 2 results:  [doc_B #1, doc_A #3, doc_D #2]
Query 3 results:  [doc_A #1, doc_C #2, doc_B #3]

RRF scores:
  doc_A: 1/(60+1) + 1/(60+3) + 1/(60+1) = 0.0328  ← rank #1 ✅
  doc_B: 1/(60+2) + 1/(60+1) + 1/(60+3) = 0.0324  ← rank #2
  doc_D: 1/(60+2)                         = 0.0161  ← rank #3
  doc_C: 1/(60+3) + 1/(60+2)             = 0.0159  ← rank #4
```

---

## 7. Speculative RAG ⭐⭐⭐

> **Pattern:** A **small, fast specialist LLM** drafts multiple candidate answers from retrieved docs.
> A **large, powerful generalist LLM** then picks and refines the best one.
> Saves cost — the large LLM verifies instead of reading all documents itself.

```python
# 07_speculative_rag.py

from shared_setup import retriever, format_docs
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

small_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # fast specialist drafter
large_llm = ChatOpenAI(model="gpt-4o",      temperature=0)    # powerful verifier

draft_prompt = ChatPromptTemplate.from_template("""
You are a specialist AI. Based ONLY on the document below,
write a concise 2-sentence answer to the question.

Document: {document}
Question: {question}
Draft answer:
""")

verify_prompt = ChatPromptTemplate.from_template("""
You are a senior AI reviewer. Below are draft answers to a question,
each written from a different source document.

Tasks:
1. Identify which draft(s) are most accurate and well-supported
2. Synthesise the best final answer, combining insights where appropriate
3. Correct any errors or hallucinations from the drafts

Question: {question}

Draft Answers:
{drafts}

Final verified answer:
""")

def speculative_rag(question: str) -> str:
    retrieved_docs = retriever.invoke(question)
    print(f"   Retrieved {len(retrieved_docs)} docs for drafting")

    drafts = []
    for i, doc in enumerate(retrieved_docs, 1):
        draft = (draft_prompt | small_llm | StrOutputParser()).invoke({
            "document": doc.page_content, "question": question
        })
        drafts.append(f"Draft {i} (from '{doc.metadata.get('source', '?')}'):\n{draft}")
        print(f"   Draft {i}: '{draft[:80]}...'")

    return (verify_prompt | large_llm | StrOutputParser()).invoke({
        "question": question, "drafts": "\n\n".join(drafts)
    })

# ── Run ──────────────────────────────────────────────────────────
print(speculative_rag("What is Self-RAG and how does it improve retrieval?"))
```

Run it:
```bash
uv run 07_speculative_rag.py
```

**Cost comparison:**
```
Naive RAG:
  LLM reads all 4 chunks (~1000 tokens)             → 1 expensive call

Speculative RAG:
  small_llm: 1 chunk × 4 calls (~250 tokens each)   → cheap
  large_llm: reads only 4 short drafts (~400 tokens) → 1 cheaper call
  Result: lower total cost, same or better quality ✅
```

---

## 8. Agentic RAG ⭐⭐⭐⭐

> **Pattern:** The LLM acts as an **autonomous agent** with tools. It decides WHICH tool to call,
> calls it, reviews results, and iterates — until it has enough information to answer.

```python
# 08_agentic_rag.py

from shared_setup import retriever, llm, format_docs
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate

# ── Tool 1: Vector Store Search ───────────────────────────────────
def search_knowledge_base(query: str) -> str:
    results = retriever.invoke(query)
    return format_docs(results) if results else "No relevant documents found."

# ── Tool 2: Web Search ────────────────────────────────────────────
web_search = DuckDuckGoSearchRun()

# ── Tool 3: Calculator ────────────────────────────────────────────
def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(
        name="KnowledgeBase",
        func=search_knowledge_base,
        description=(
            "Search the internal knowledge base for LangChain, RAG, FAISS, "
            "and AI concepts. Always use this FIRST before web search."
        )
    ),
    Tool(
        name="WebSearch",
        func=web_search.run,
        description="Search the web for current events or info not in the knowledge base."
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Evaluate math expressions. Input: a Python math expression like '4 * 512'."
    ),
]

react_prompt = PromptTemplate.from_template("""
You are a helpful research assistant with access to tools.
Always check KnowledgeBase first, then WebSearch if needed.

Tools available:
{tools}

Format:
Thought: (think about what to do)
Action: (tool name from [{tool_names}])
Action Input: (input to the tool)
Observation: (result from tool)
... (repeat as needed)
Thought: I now have enough information.
Final Answer: (your complete answer)

Question: {input}
{agent_scratchpad}
""")

agent          = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools,
    verbose=True, max_iterations=5, handle_parsing_errors=True
)

# ── Run ──────────────────────────────────────────────────────────
result = agent_executor.invoke({
    "input": "What is RAG and if a RAG system retrieves 4 documents "
             "each with 512 tokens, what is the total context size?"
})

print(f"\nFinal Answer: {result['output']}")

# Agent trace (verbose=True):
# Thought: Check knowledge base for RAG definition.
# Action: KnowledgeBase | Action Input: "What is RAG?"
# Observation: [docs...]
# Thought: Now calculate 4 * 512.
# Action: Calculator | Action Input: "4 * 512"
# Observation: 2048
# Final Answer: RAG is... total context = 2048 tokens.
```

Run it:
```bash
uv run 08_agentic_rag.py
```

---

## 9. Graph RAG ⭐⭐⭐⭐

> **Pattern:** Documents are parsed into a **knowledge graph** (entities + relationships).
> Queries traverse the graph to answer **multi-hop relational questions** that pure vector search cannot handle.

```python
# 09_graph_rag.py

from shared_setup import retriever, llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import networkx as nx

# ── Step 1: Build Knowledge Graph ────────────────────────────────
# In production, an LLM extracts entities + relations from documents.
# Here we define the graph manually for clarity.

G = nx.DiGraph()

G.add_nodes_from([
    "LangChain", "FAISS", "RAG", "HuggingFace",
    "OpenAI", "GPT-4", "Embeddings", "Vector Store",
    "Facebook AI", "LLM"
])

relations = [
    ("LangChain",   "RAG",          "enables"),
    ("LangChain",   "FAISS",        "integrates_with"),
    ("LangChain",   "HuggingFace",  "integrates_with"),
    ("LangChain",   "OpenAI",       "integrates_with"),
    ("RAG",         "LLM",          "uses"),
    ("RAG",         "Vector Store", "uses"),
    ("FAISS",       "Vector Store", "is_a"),
    ("FAISS",       "Facebook AI",  "developed_by"),
    ("FAISS",       "Embeddings",   "stores"),
    ("HuggingFace", "Embeddings",   "provides"),
    ("OpenAI",      "GPT-4",        "developed"),
    ("GPT-4",       "LLM",          "is_a"),
    ("Embeddings",  "Vector Store", "stored_in"),
]
G.add_edges_from([(s, t, {"relation": r}) for s, t, r in relations])
print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Step 2: Entity Extractor ──────────────────────────────────────
entity_prompt = ChatPromptTemplate.from_template("""
Extract all named entities (tools, companies, concepts, products) from the question.
Return them as a comma-separated list. Return ONLY the list.

Question: {question}
Entities:
""")

def extract_entities(question: str) -> list[str]:
    raw = (entity_prompt | llm | StrOutputParser()).invoke({"question": question})
    return [e.strip() for e in raw.split(",") if e.strip()]

# ── Step 3: BFS Graph Traversal ───────────────────────────────────
def traverse_graph(entities: list[str], hops: int = 2) -> str:
    graph_context = []
    for entity in entities:
        matched = [n for n in G.nodes if entity.lower() in n.lower()]
        if not matched:
            continue
        start_node = matched[0]
        print(f"   Traversal from: '{start_node}'")
        visited, queue = set(), [(start_node, 0)]
        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > hops:
                continue
            visited.add(node)
            for _, nbr, data in G.out_edges(node, data=True):
                graph_context.append(f"{node} --[{data.get('relation','?')}]--> {nbr}")
                queue.append((nbr, depth + 1))
            for pre, _, data in G.in_edges(node, data=True):
                graph_context.append(f"{pre} --[{data.get('relation','?')}]--> {node}")
    return "\n".join(set(graph_context))

# ── Step 4: Hybrid Graph + Vector RAG ────────────────────────────
answer_prompt = ChatPromptTemplate.from_template("""
Answer the question using the knowledge graph context AND document context.

Knowledge Graph (entity relationships):
{graph_context}

Document Context:
{doc_context}

Question: {question}
Answer:
""")

def graph_rag(question: str) -> str:
    entities      = extract_entities(question)
    print(f"   Entities: {entities}")
    graph_context = traverse_graph(entities, hops=2)
    doc_context   = format_docs(retriever.invoke(question))
    return (answer_prompt | llm | StrOutputParser()).invoke({
        "graph_context": graph_context or "No graph data found.",
        "doc_context":   doc_context,
        "question":      question
    })

# ── Run ──────────────────────────────────────────────────────────
print(graph_rag("What did the company that developed FAISS contribute to AI?"))
print(graph_rag("How are HuggingFace and LangChain related to RAG?"))
```

Run it:
```bash
uv run 09_graph_rag.py
```

**Why Graph RAG handles multi-hop questions:**
```
Question: "What company developed the library LangChain uses for fast search?"

Pure vector search → searches semantically, may miss the link ❌

Graph RAG traversal:
  LangChain → integrates_with → FAISS
  FAISS     → developed_by   → Facebook AI
  Answer: "Facebook AI"  ✅  (2-hop traversal)
```

---

## Summary Cheat Sheet

```
┌──────────────────┬────────────────────────────────────────────────────────────┐
│ RAG Type         │ Core Pattern                                                │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Naive            │ query → retrieve(k) → prompt(context+q) → llm → answer    │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Advanced         │ rewrite(q) → HyDE → retrieve → rerank → generate          │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Modular          │ route(q) → select_retriever → select_prompt → generate     │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Self-RAG         │ [RETRIEVE?] → retrieve → [ISREL?] → generate → [ISSUP?]   │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Corrective       │ retrieve → eval(HIGH/MED/LOW) → refine OR web_search       │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Fusion           │ gen_queries(n) → retrieve_each → RRF_merge → generate      │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Speculative      │ small_llm(draft×n) → large_llm(verify+finalize)            │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Agentic          │ agent → [tool?] → iterate until done → final_answer        │
├──────────────────┼────────────────────────────────────────────────────────────┤
│ Graph            │ extract_entities → graph_traverse → vector_search → merge  │
└──────────────────┴────────────────────────────────────────────────────────────┘
```

### Full `uv` Workflow Recap

```bash
# 1. Create and enter project folder
mkdir langchain-rag-types && cd langchain-rag-types

# 2. Initialise project
uv init
uv python pin 3.11

# 3. Add all dependencies (replaces pip install)
uv add langchain langchain-community langchain-core \
       langchain-openai langchain-huggingface \
       faiss-cpu sentence-transformers networkx duckduckgo-search

# 4. Set API key
export OPENAI_API_KEY="sk-..."

# 5. Run any example (no manual venv activation needed)
uv run shared_setup.py
uv run 01_naive_rag.py
uv run 09_graph_rag.py
```

---
