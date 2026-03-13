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