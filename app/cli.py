import shutil
import textwrap

from app.agent import run_agent
from app.errors import LocalAgentError
from app.ingest.ingest import ingest_codebase
from app.vector_store import close_client

MAX_WIDTH = 100


def _report_progress(done: int, total: int) -> None:
    """Overwrite a single line: indexing a large repository is not instant."""
    print(f"\r  embedding {done}/{total} chunks...", end="", flush=True)


def wrap(text: str, indent: str = "") -> str:
    """Wrap on word boundaries.

    Left to the terminal, an answer breaks mid-token and splits exactly the
    thing worth reading -- a path like OwnerController.java:72-75.
    """
    width = min(shutil.get_terminal_size(fallback=(MAX_WIDTH, 24)).columns, MAX_WIDTH)
    paragraphs = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            paragraphs.append("")
            continue
        paragraphs.append(
            textwrap.fill(
                paragraph,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
                # Keep identifiers and paths intact rather than hyphenating or
                # splitting them across lines.
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(paragraphs)


def main():
    print("🧠 Local AI Code Agent")
    print("Commands:")
    print("  load <path>  → index codebase")
    print("  exit         → quit\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue

        if query.lower() == "exit":
            break

        if query.startswith("load "):
            path = query.replace("load ", "").strip()
            try:
                summary = ingest_codebase(path, progress=_report_progress)
                print(f"\r{' ' * 40}\r", end="")
                if summary["indexed"]:
                    print(
                        f"\n✅ Indexed {summary['indexed']} chunks from "
                        f"{summary['files_changed']} changed file(s); "
                        f"{summary['files_total']} file(s) known.\n"
                    )
                else:
                    print(f"\n✅ Already up to date ({summary['files_total']} files).\n")
            except (LocalAgentError, ValueError) as error:
                print(f"\nError: {error}\n")
            continue

        try:
            response = run_agent(query)
            print("\nAgent:")
            print(wrap(response, indent="  "), "\n")
        except LocalAgentError as error:
            print(f"\nError: {error}\n")
    close_client()


if __name__ == "__main__":
    main()
