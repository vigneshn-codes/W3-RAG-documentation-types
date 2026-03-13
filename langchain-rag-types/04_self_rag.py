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