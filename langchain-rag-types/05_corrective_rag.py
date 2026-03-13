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