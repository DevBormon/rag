from pocketflow import Flow, AsyncFlow
from nodes import (
    ChunkDocumentsNode, EmbedChunksNode, CreateVectorDBNode,
    EmbedQueryNode, SearchNode, GenerateAnswerNode, AsyncGenerateAnswerNode
)


def vector_flow():
    chunk_docs = ChunkDocumentsNode()
    embed = EmbedChunksNode()
    vdb = CreateVectorDBNode()
    chunk_docs >> embed >> vdb
    return Flow(start=chunk_docs)


def query_flow():
    embed = EmbedQueryNode()
    search = SearchNode()
    answer = GenerateAnswerNode()
    embed >> search >> answer
    return Flow(start=embed)


def async_query_flow():
    embed = EmbedQueryNode()
    search = SearchNode()
    answer = AsyncGenerateAnswerNode()
    embed >> search >> answer
    return AsyncFlow(start=embed)