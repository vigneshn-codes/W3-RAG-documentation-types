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
#print("API KEY:", os.getenv("OPENAI_API_KEY"))

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