"""A bounded tool-use loop over the local model.

The model is given a small set of tools and asked for one action at a time. Each
observation is fed back, so a question that needs two lookups ("which endpoint
creates an owner, and what does it validate?") can take two steps instead of
being answered from a single retrieval.

The loop is deliberately bounded and degrades to a direct answer: a small local
model will sometimes ignore the protocol, and an agent that loops forever or
crashes on malformed output is worse than one that answers in a single pass.
"""

import re

from app.citations import cited_files, verify
from app.llm import query_llm
from app.memory import ShortTermMemory, retrieve_memory, store_memory
from app.rag import agent_query, format_context
from app.retrieval import hybrid_search
from app.tools import TOOL_DESCRIPTIONS, TOOLS

MAX_STEPS = 4

TOOL_CALL = re.compile(r"TOOL\s*:\s*(\w+)\s*\|\s*(.+)", re.IGNORECASE)
FINAL_ANSWER = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

PROMPT = """You are a code assistant answering questions about an indexed codebase.

Available tools:
{tools}

Reply with exactly one line, in one of these two forms:
TOOL: <tool_name> | <argument>
FINAL_ANSWER: <answer in at most 120 words>

Cite only file names that appear in the material below. If the answer is not
there, use a tool to find it, or say it is not in the index. Never invent a
file name, path, or behaviour.
{memory}
RETRIEVED SOURCE:
{context}

QUESTION:
{question}
{transcript}"""


def _memory_block(short_term: ShortTermMemory, question: str) -> str:
    """Assemble both memory kinds into the prompt, omitting whichever is empty."""
    sections = []

    conversation = short_term.get_context()
    if conversation:
        sections.append(f"CONVERSATION SO FAR:\n{conversation}")

    recalled = retrieve_memory(question)
    if recalled:
        sections.append("RELEVANT EARLIER EXCHANGES:\n" + "\n".join(recalled))

    return "\n\n" + "\n\n".join(sections) + "\n" if sections else "\n"


def run_agent(
    question: str,
    memory: ShortTermMemory | None = None,
    max_steps: int = MAX_STEPS,
    remember: bool = True,
) -> str:
    """Answer a question, using tools for up to max_steps before answering."""
    short_term = memory if memory is not None else _session_memory
    transcript: list[str] = []

    # Built once: recall depends on the question, which does not change during
    # the loop. Rebuilding it per step re-embedded the question every iteration,
    # and each embedding call evicts the chat model when both do not fit in VRAM.
    memory_block = _memory_block(short_term, question)

    # Retrieve before the first turn rather than waiting to be asked. Left to
    # itself the model answers straight from parametric knowledge and invents
    # plausible file names, so grounding is not optional.
    results = hybrid_search(question, top_k=3)
    if not results:
        return "No indexed documents were found. Run `load <path>` first."
    context = format_context(results)

    # Everything the model has legitimately been shown. Tool observations add to
    # it, so a file found mid-loop is not later flagged as invented.
    seen_files = {result.get("file") for result in results if result.get("file")}

    for _step in range(max_steps):
        prompt = PROMPT.format(
            tools="\n".join(TOOL_DESCRIPTIONS.values()),
            memory=memory_block,
            context=context,
            question=question,
            transcript=("\n\nSTEPS SO FAR:\n" + "\n".join(transcript)) if transcript else "",
        )
        reply = query_llm(prompt).strip()

        final = FINAL_ANSWER.search(reply)
        call = TOOL_CALL.search(reply)

        # A model that emits both is answering while narrating; prefer the answer
        # unless the tool call came first.
        if final and (not call or final.start() < call.start()):
            answer = verify(final.group(1).strip(), seen_files)
            return _finish(question, answer, short_term, remember)

        if call:
            name, argument = call.group(1).strip(), call.group(2).strip()
            tool = TOOLS.get(name)
            if tool is None:
                transcript.append(f"TOOL {name} -> no such tool. Available: {', '.join(TOOLS)}")
                continue
            try:
                observation = str(tool(argument))
            except Exception as error:  # a failing tool must not end the session
                observation = f"tool raised {type(error).__name__}: {error}"
            seen_files |= cited_files(observation)
            transcript.append(f"TOOL {name}({argument}) ->\n{observation[:1200]}")
            continue

        # Unparseable reply. Nothing is gained by asking again with the same
        # prompt, so fall back to answering directly from retrieval.
        break

    answer = verify(_direct_answer(question, transcript, context), seen_files)
    return _finish(question, answer, short_term, remember)


def _direct_answer(question: str, transcript: list[str], context: str = "") -> str:
    """Answer in a single pass, using whatever the loop already gathered."""
    if not transcript:
        return agent_query(question)

    context = context or format_context(hybrid_search(question, top_k=3))
    prompt = f"""Answer using only the material below. At most 120 words. Cite file names.
If the answer is not present, say so rather than inventing it.

RETRIEVED SOURCE:
{context}

TOOL OBSERVATIONS:
{chr(10).join(transcript)}

QUESTION:
{question}
"""
    return query_llm(prompt).removeprefix("FINAL_ANSWER:").strip()


def _finish(question: str, answer: str, short_term: ShortTermMemory, remember: bool) -> str:
    short_term.add("User", question)
    short_term.add("AI", answer)
    if remember:
        store_memory(f"Q: {question}\nA: {answer}")
    return answer


_session_memory = ShortTermMemory()
