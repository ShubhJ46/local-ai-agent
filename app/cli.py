from app.rag import agent_query

def main():
    print("🧠 Local AI Agent (type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        response = agent_query(query)
        print("\nAgent:", response, "\n")

if __name__ == "__main__":
    main()