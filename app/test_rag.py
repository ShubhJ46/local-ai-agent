from app.rag import answer_query
from app.vector_store import close_client
from app.rag import answer_query
from app.memory import init_memory_collection
from app.embed import get_embedding
from app.rag import agent_query


# # initialize memory
# init_memory_collection(len(get_embedding("test")))

# query = "Where is main function defined?"


# response = answer_query(query, 3)

# print("\nResponse:\n")
# print(response)

# print(answer_query("Can you explain it?"))


print(agent_query("Where is main function defined?"))
print(agent_query("Open main.cpp and explain it"))

close_client()