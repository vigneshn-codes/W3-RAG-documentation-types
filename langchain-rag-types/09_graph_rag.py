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