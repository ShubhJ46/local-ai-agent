from app.embed import get_embedding
from app.vector_store import search
from app.llm import query_llm
from app.memory import ShortTermMemory, retrieve_memory
from app.memory import store_memory
from app.memory import init_memory_collection

import json
from app.llm import query_llm
from app.tools import TOOLS

def agent_query(query):
    prompt = f"""
You are an AI code assistant.

You help users understand codebases.
Mention file names when possible.

Available tools:
- search_documents: search codebase
- read_file: read full file

Use:
- search_documents → for finding logic
- read_file → for full file understanding

If a tool is needed, respond ONLY in JSON.

User: {query}
"""

    response = query_llm(prompt)

    # Try parsing tool call
    try:
        tool_call = json.loads(response)
        tool_name = tool_call["tool"]
        tool_input = tool_call["input"]

        if tool_name in TOOLS:
            tool_result = TOOLS[tool_name](tool_input)

            # Second pass to LLM with tool result
            final_prompt = f"""
You used tool: {tool_name}

Tool result:
{tool_result}

Now answer the user question clearly.

Question:
{query}

Answer:
"""
            return query_llm(final_prompt)

    except:
        pass

    return response

memory = ShortTermMemory()

def answer_query(query, top_k=5):
    query_embedding = get_embedding(query)

    # initialize memory collection if needed
    init_memory_collection(len(query_embedding))
    # 1. Retrieve document context
    query_embedding = get_embedding(query)
    doc_results = search(query_embedding, top_k=top_k)
    doc_context = "\n\n".join([r["text"] for r in doc_results])

    # 2. Retrieve long-term memory
    past_memory = retrieve_memory(query)
    memory_context = "\n".join(past_memory)

    # 3. Short-term memory
    chat_context = memory.get_context()

    # 4. Build prompt
    prompt = f"""
You are an AI assistant with memory.

Conversation so far:
{chat_context}

Relevant past memory:
{memory_context}

Relevant documents:
{doc_context}

Answer clearly and concisely.
Avoid repeating the same information.
Ignore any unrelated context.

If multiple documents are present, focus only on relevant parts.

Question:
{query}

Answer:
"""

    # 5. Get response
    response = query_llm(prompt)

    # 6. Store in short-term memory
    memory.add("User", query)
    memory.add("AI", response)

    # 7. Store in long-term memory
    store_memory(f"User: {query}\nAI: {response}")

    return response