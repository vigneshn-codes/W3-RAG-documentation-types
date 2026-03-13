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