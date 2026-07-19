from app.errors import LocalAgentError
from app.ingest.ingest import ingest_codebase
from app.rag import agent_query
from app.vector_store import close_client


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
                count = ingest_codebase(path)
                print(f"\n✅ Indexed {count} chunks from {path}\n")
            except (LocalAgentError, ValueError) as error:
                print(f"\nError: {error}\n")
            continue

        try:
            response = agent_query(query)
            print("\nAgent:", response, "\n")
        except LocalAgentError as error:
            print(f"\nError: {error}\n")
    close_client()


if __name__ == "__main__":
    main()
