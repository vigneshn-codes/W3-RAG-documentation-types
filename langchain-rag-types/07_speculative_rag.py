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