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