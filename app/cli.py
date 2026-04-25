from app.rag import agent_query
from app.ingest.ingest import ingest_codebase

def main():
    print("🧠 Local AI Code Agent")
    print("Commands:")
    print("  load <path>  → index codebase")
    print("  exit         → quit\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        if query.startswith("load "):
            path = query.replace("load ", "").strip()
            count = ingest_codebase(path)
            print(f"\n✅ Indexed {count} chunks from {path}\n")
            continue

        response = agent_query(query)
        print("\nAgent:", response, "\n")

if __name__ == "__main__":
    main()