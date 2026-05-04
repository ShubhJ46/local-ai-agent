import json
import re

from app.embed import get_embedding
from app.vector_store import search
from app.llm import query_llm
from app.tools import TOOLS, rerank, rewrite_query

from app.memory import ShortTermMemory, retrieve_memory, store_memory, init_memory_collection


# ---------------------------
# Helpers
# ---------------------------

def extract_json(text):
    """Safely extract first JSON object from LLM output."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except:
        return None


def format_context(results):
    """Convert retrieved docs into structured context for LLM."""
    context_blocks = []

    for r in results:
        if r.get("type") == "endpoint":
            block = f"""Endpoint: {r.get('http_method')} {r.get('endpoint')}
Method: {r.get('name')}
File: {r.get('file')}"""
        else:
            block = f"""{r.get('type', '').upper()}: {r.get('name')}
File: {r.get('file')}"""

        context_blocks.append(block)

    return "\n\n".join(context_blocks)


# ---------------------------
# Main Agent (RAG + Tools)
# ---------------------------

memory = ShortTermMemory()


def agent_query(query, max_steps=3, top_k=5):
    """
    Hybrid RAG + Agent:
    1. Retrieve context
    2. Let LLM decide (use context or tools)
    3. Loop for multi-step reasoning
    """

    rewritten_query = rewrite_query(query)

    print("Original query:", query)
    print("Rewritten query:", rewritten_query)


    # ---------------------------
    # Step 1: Retrieve context
    # ---------------------------
    query_embedding = get_embedding(rewritten_query)

    doc_results = search(
        query_embedding,
        top_k=top_k,
        filter={"type": "endpoint"}  # focus on endpoints first
    )

    if not doc_results:
        return TOOLS["find_endpoint"](query)

    if not doc_results:
        doc_results = search(
            query_embedding,
            top_k=top_k  # no filter
        )
    doc_results = rerank(doc_results, query)
    doc_context = format_context(doc_results)

    # Optional debug
    print("\n===== DEBUG: RETRIEVED CONTEXT =====")
    print(doc_context if doc_context else "No context found")
    print("====================================\n")

    # ---------------------------
    # Step 2: Memory 
    # ---------------------------
    init_memory_collection(len(query_embedding))

    past_memory = retrieve_memory(query)
    memory_context = "\n".join(past_memory)

    chat_context = memory.get_context()

    # ---------------------------
    # Step 3: Initial Prompt
    # ---------------------------
    current_prompt = f"""
You are an AI code assistant.

You MUST use the provided context to answer.
Do NOT guess or invent information.

---

CONTEXT:
{doc_context}

---

PAST MEMORY:
{memory_context}

---

CHAT HISTORY:
{chat_context}

---

TOOLS:
search_documents(input: string)
read_file(input: string)
find_endpoint(input: string)

---

RULES:
- If answer is already in context or tool result → return FINAL_ANSWER
- DO NOT call tools again if you already have enough information
- Only call another tool if critical information is missing

FORMAT:
{{ "tool": "<tool_name>", "input": "<string>" }}

User Question:
{query}
"""

    # ---------------------------
    # Step 4: Agent Loop
    # ---------------------------
    for step in range(max_steps):
        response = query_llm(current_prompt)

        # Case 1: Final Answer
        if "FINAL_ANSWER:" in response:
            final_answer = response.split("FINAL_ANSWER:")[-1].strip()

            # store memory
            memory.add("User", query)
            memory.add("AI", final_answer)
            # store_memory(f"User: {query}\nAI: {final_answer}")

            return final_answer

        # Case 2: Tool call
        tool_call = extract_json(response)

        if tool_call:
            tool_name = tool_call.get("tool")
            tool_input = tool_call.get("input")

            if tool_name in TOOLS and tool_input:
                tool_result = TOOLS[tool_name](tool_input)

                current_prompt = f"""
You are an AI code assistant.

User Question:
{query}

Existing Context:
{doc_context}

You used tool: {tool_name}

Tool result:
{tool_result}

---

RULES:
- Prefer existing context if sufficient
- If more info needed → call another tool
- If enough → return FINAL_ANSWER
"""

                continue

        # Fallback (bad format / no tool / hallucination)
        return response.strip()

    return "Max steps reached without final answer."