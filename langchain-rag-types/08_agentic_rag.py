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