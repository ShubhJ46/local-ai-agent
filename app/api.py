"""HTTP interface to the agent.

The CLI is the primary interface, but an endpoint is far easier to sample than
an interactive terminal, and it lets the response carry structure the terminal
cannot: /query returns the sources the answer was grounded on, with line ranges,
so a caller can check the answer against the code rather than trusting prose.

    uvicorn app.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import run_agent
from app.errors import LocalAgentError
from app.ingest.ingest import ingest_codebase
from app.memory import ShortTermMemory
from app.retrieval import hybrid_search
from app.vector_store import iter_points

app = FastAPI(
    title="Local AI Code Agent",
    description="Ask questions about an indexed codebase, answered from retrieved source.",
    version="0.2.0",
)

# One conversation per process. Sessions are deliberately not modelled: this is
# a local single-user tool, and inventing session handling would be scope no
# caller has asked for.
_memory = ShortTermMemory()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Which endpoint creates a new owner?"])
    top_k: int = Field(default=3, ge=1, le=20)


class Source(BaseModel):
    file: str | None
    lines: str | None
    type: str | None
    name: str | None
    endpoint: str | None = None
    http_method: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IndexRequest(BaseModel):
    path: str = Field(min_length=1, examples=["/absolute/path/to/a/codebase"])
    full: bool = Field(default=False, description="Re-embed everything, ignoring content hashes.")


class IndexResponse(BaseModel):
    indexed: int
    files_changed: int
    files_removed: int
    files_total: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


def to_source(result: dict) -> Source:
    start, end = result.get("start_line"), result.get("end_line")
    return Source(
        file=result.get("file"),
        lines=f"{start}-{end}" if start and end else None,
        type=result.get("type"),
        name=result.get("name"),
        endpoint=result.get("endpoint"),
        http_method=result.get("http_method"),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether anything is indexed. Cheap enough to poll."""
    try:
        return HealthResponse(status="ok", indexed_chunks=len(iter_points()))
    except Exception:
        # An absent collection is the normal state before the first index.
        return HealthResponse(status="empty", indexed_chunks=0)


@app.post("/index", response_model=IndexResponse)
def index(request: IndexRequest) -> IndexResponse:
    try:
        return IndexResponse(**ingest_codebase(request.path, full=request.full))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except LocalAgentError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        # Retrieve separately so the caller sees the same source the model was
        # given, rather than a second, unrelated search.
        sources = hybrid_search(request.question, top_k=request.top_k)
        answer = run_agent(request.question, memory=_memory)
    except LocalAgentError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return QueryResponse(answer=answer, sources=[to_source(result) for result in sources])
