from pocketflow import Node, BatchNode, AsyncNode
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

from utils import call_llm, call_llm_async, get_embedding

# ChromaDB client
chroma_client = chromadb.HttpClient(host="localhost", port=7000)
collection = chroma_client.get_or_create_collection("rag_docs")


class ChunkDocumentsNode(BatchNode):
    def prep(self, shared):
        return shared["texts"].items()

    def exec(self, item):
        file_name, text = item
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.create_documents([text], metadatas=[{"source": file_name}])
        return chunks

    def exec_fallback(self, prep_res, exc):
        print(f"Error Chunking Node : {exc}")
        # Provide a simple fallback instead of crashing
        raise exc

    def post(self, shared, prep_res, exec_res_list):
        for file_chunks in exec_res_list:
            for chunk in file_chunks:
                shared["chunks"].append(chunk)
        return "default"


class EmbedChunksNode(BatchNode):
    def prep(self, shared):
        return [(i, chunk) for i, chunk in enumerate(shared["chunks"])]

    def exec(self, item):
        i, chunk = item
        embedding = get_embedding(chunk.page_content)
        return {
            "id": f"{chunk.metadata['source']}_chunk_{i}",
            "chunk": chunk,
            "embedding": embedding,
        }

    def exec_fallback(self, prep_res, exc):
        print(f"Error Embding Node : {exc}")
        # Provide a simple fallback instead of crashing
        raise exc

    def post(self, shared, prep_res, exec_res_list):
        for res in exec_res_list:
            shared["embeddings"].append(res)
        return "default"


class CreateVectorDBNode(Node):
    def prep(self, shared):
        print(f"DEL : {shared["collection_del_status"]}")
        return shared["collection_del_status"], shared["embeddings"]

    def exec(self, prep_res):
        collection_del_status, embeddings = prep_res
        if collection_del_status:
            global collection
            collection = chroma_client.get_or_create_collection("rag_docs")

        print(f"Total documents in collection: {collection.count()}")
        for item in embeddings:
            collection.add(
                ids=[item["id"]],
                metadatas=[{"source": item["chunk"].metadata["source"]}],
                documents=[item["chunk"].page_content],
                embeddings=[item["embedding"]],
            )
        return collection

    def exec_fallback(self, prep_res, exc):
        print(f"Error Vector DB entry : {exc}")
        # Provide a simple fallback instead of crashing
        raise exc

    def post(self, shared, prep_res, exec_res):
        print(f"✅ Added {len(shared['embeddings'])} chunks to vector DB")
        return "default"


class EmbedQueryNode(Node):
    def prep(self, shared):
        return shared["query"]

    def exec(self, query):
        return get_embedding(query)

    def post(self, shared, prep_res, exec_res):
        shared["query_embedding"] = exec_res
        return "default"


class SearchNode(Node):
    def prep(self, shared):
        return shared["query"], shared["query_embedding"]

    def exec(self, inputs):
        _query, query_embed = inputs
        results = collection.query(query_embeddings=[query_embed], n_results=3)
        return results

    def post(self, shared, prep_res, exec_res):
        shared["retrieved_document"] = exec_res
        return "default"


class GenerateAnswerNode(Node):
    def prep(self, shared):
        return shared["query"], shared["retrieved_document"]

    def exec(self, inputs):
        query, retrieved = inputs
        docs = retrieved["documents"][0]
        context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(docs))

        prompt = f"""Answer briefly using only the context. Cite sources like [1], [2].
Context:
{context}
Question: {query}
Answer:"""

        return call_llm(prompt)

    def post(self, shared, prep_res, exec_res):
        shared["generated_answer"] = exec_res
        print(f"\n🤖 {exec_res}")
        return "default"


class AsyncGenerateAnswerNode(AsyncNode):
    async def prep_async(self, shared):
        metas = shared["retrieved_document"]["metadatas"][0]
        shared["sources"] = list(set(m.get("source", "unknown") for m in metas))
        return shared["query"], shared["retrieved_document"], shared.get("queue")

    async def exec_async(self, inputs):
        query, retrieved, q = inputs
        docs = retrieved["documents"][0]
        context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(docs))

        prompt = f"""Answer briefly using only the context. Cite sources like [1], [2].
Context:
{context}
Question: {query}
Answer:"""

        if q:

            async def on_token(tok):
                await q.put(tok)

            answer = await call_llm_async(prompt, on_token=on_token)
            await q.put(None)  # sentinel: done
        else:
            answer = await call_llm_async(prompt)

        return answer

    async def post_async(self, shared, prep_res, exec_res):
        shared["generated_answer"] = exec_res
        return "default"
