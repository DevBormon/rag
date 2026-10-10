from flow import vector_flow, query_flow
import os

def main():
    # user_input = input("Enter your question: ")
    # user_input = "How many vacation days do employees get?"
    # user_input = "How many vacation days?"
    # user_input = "How many days of paid annual leave do employees get?"
    # user_input = "A robot stopped in a corridor and is blinking orange. What do I do?"
    # user_input = "How much does the Carrier X2 cost and what does the price include?"
    # user_input = "What color is the Carrier X2?"
    user_input = "How many stufs in the Aurora Dynamics?"
    
    shared = {
        "texts": {},
        "chunks": [],
        "embeddings": [],
        "query": user_input,
        "query_embedding": None,
        "retrieved_document": []
    }
    
    if shared["query"] != "":
        # print(f"User question: {shared['query']}")
        
        query_flow.run(shared)
        return
    
    
    for filename in os.listdir("data"):
        if filename.endswith(".txt"):
            filepath = os.path.join("data", filename)
            with open(filepath, "r", encoding="utf-8") as f:
                if filename not in shared["texts"]:
                    shared["texts"][filename] = f.read()
    
    vector_flow.run(shared)
    
if __name__ == "__main__":
    main()