from pocketflow import Node, BatchNode
from utils import call_llm, get_embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

client = chromadb.HttpClient(
    host="localhost",
    port=7000
)

# client.delete_collection("aurora_docs")

collection = client.get_or_create_collection("aurora_docs")


class ChunkDocumentsNode(BatchNode):
    def prep(self, shared):
        return shared["texts"].items()
    
    def exec(self, item):
        """Chunk a single text into smaller pieces"""
        file_name, text = item
        # print(f"Chunking {file_name}...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        
        chunks = splitter.create_documents(
            [text],
            metadatas=[{"source": file_name}]
        )
                
        return chunks
    
    def post(self, shared, prep_res, exec_res_list):
        """Store chunked texts in the shared store"""
        # Flatten the list of lists into a single list of chunks
        for file_chunks in exec_res_list:
            for chunk in file_chunks:
                # print(f"Storing chunk from {chunk.metadata['source']}: {chunk.page_content[:50]}...")
                shared["chunks"].append(chunk)
                
        # print(f"Total chunks created: {len(shared['chunks'])}")
        
        return "default"
    
class EmbedChunksNode(BatchNode):
    def prep(self, shared):
        return [
            (i, chunk)
            for i, chunk in enumerate(shared["chunks"])
        ]
    
    def exec(self, item):
        i, chunk = item
        embedding = get_embedding(chunk.page_content)
        return {"id": f"{chunk.metadata['source']}_chunk_{i}", "chunk": chunk, "embedding": embedding}
    
    def post(self, shared, prep_res, exec_res_list):
        for res in exec_res_list:
            id = res["id"]
            chunk = res["chunk"]
            embedding = res["embedding"]
            shared["embeddings"].append({
                "id": id,
                "chunk": chunk,
                "embedding": embedding
            })
        
        # print(f"Storing embedding for\n id: {id}\n source: {chunk.metadata['source']}\n chunk: {chunk.page_content[:50]}\n embedding: {embedding[:4]}\n\n")
        
        return "default"
    
class CreateVectorDBNode(Node):
    def prep(self, shared):
        return shared["embeddings"]
    def exec(self, embeddings):
        for item in embeddings:
            collection.add(
                ids=[item["id"]],
                metadatas=[{"source": item["chunk"].metadata["source"]}],
                documents=[item["chunk"].page_content],
                embeddings=[item["embedding"]]
            )
        return collection
    def post(self, shared, prep_res, exec_res):
        data = exec_res.get()
        print(data)
        return "default"


class EmbedQueryNode(Node):
    def prep(self, shared):
        return shared["query"]
    
    def exec(self, query):
        # print(f"Generating embedding for query: {query}...")
        embedding = get_embedding(query)
        return embedding
    
    def post(self, shared, prep_res, exec_res):
        shared["query_embedding"] = exec_res
        return "default"

class SearchNode(Node):
    def prep(self, shared):
        return shared["query"], shared["query_embedding"]
    
    def exec(self, inputs):
        quety, query_embed = inputs
        # print(f"Searching for similar chunks to the query embedding...")
        results = collection.query(
            query_embeddings=[query_embed],
            # query_texts=[quety],
            n_results=3
        )
        return results
    
    def post(self, shared, prep_res, exec_res):
        
        # for i, (chunk, score) in enumerate(zip(exec_res['documents'][0], exec_res['distances'][0])):
        #     source = exec_res['metadatas'][0][i]['source']
        #     print(f"Result {i+1}:")
        #     print(f"Source: {source}")
        #     print(f"Chunk: {chunk[:80]}...")
        #     print(f"Score: {score}\n")
        
        shared["retrieved_document"] = exec_res
        return "default"
    
class GenerateAnswerNode(Node):
    def prep(self, shared):
        
        return shared["query"], shared["retrieved_document"]
    
    def exec(self, inputs):
        """Generate an answer using the LLM"""
        query, retrieved_doc = inputs
        
        context = "\n\n".join(retrieved_doc["documents"][0])
        
        # print(f"\n\nContext for the question:\n{context}\n\n")
        
        prompt = f"""
            Briefly answer the following question based on the context provided:             
            Question: {query}
            Context: {context}
            Answer:
            """
        
        answer = call_llm(prompt)

        return answer
    
    def post(self, shared, prep_res, exec_res):
        """Store generated answer in shared store"""
        # shared["generated_answer"] = exec_res
        print("\n🤖 Generated Answer:")
        print(exec_res)
        return "default"