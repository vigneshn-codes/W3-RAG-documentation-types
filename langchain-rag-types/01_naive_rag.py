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