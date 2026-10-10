from pocketflow import Flow
from nodes import ChunkDocumentsNode, EmbedChunksNode, CreateVectorDBNode, EmbedQueryNode, SearchNode, GenerateAnswerNode

def create_vector_flow():
    chunk_docs_node = ChunkDocumentsNode()
    embed_chunks = EmbedChunksNode()
    vector_db = CreateVectorDBNode()
    
    chunk_docs_node >> embed_chunks >> vector_db
    
    return Flow(start=chunk_docs_node)

def create_query_flow():
    query_embed = EmbedQueryNode()
    search = SearchNode()
    answer = GenerateAnswerNode()
    query_embed >> search >> answer
    # Placeholder for query flow creation
    return Flow(start=query_embed)

query_flow = create_query_flow()

vector_flow = create_vector_flow()