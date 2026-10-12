import os
import shutil
import asyncio
import json
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from flow import vector_flow, async_query_flow
from nodes import chroma_client, collection
from utils import extract_text_from_pdf, get_embedding


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ingest existing files on startup if DB is empty"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    if collection.count() == 0:
        shared = {
            "texts": {},
            "chunks": [],
            "embeddings": [],
            "query": "",
            "query_embedding": None,
            "retrieved_document": [],
        }
        for filename in os.listdir("data"):
            filepath = os.path.join("data", filename)
            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    shared["texts"][filename] = f.read()
            elif filename.endswith(".pdf"):
                shared["texts"][filename] = extract_text_from_pdf(filepath)

        if shared["texts"]:
            vector_flow().run(shared)
            print(f"✅ Startup ingestion: {len(shared['chunks'])} chunks")
    yield


app = FastAPI(title="RAG Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

async_query_flow = async_query_flow()


class Query(BaseModel):
    question: str


@app.post("/chat")
async def chat(query: Query):
    """Non-streaming JSON endpoint"""
    from flow import query_flow

    shared = {
        "texts": {},
        "chunks": [],
        "embeddings": [],
        "query": query.question,
        "query_embedding": None,
        "retrieved_document": [],
    }
    query_flow().run(shared)
    return {
        "answer": shared.get("generated_answer"),
        "sources": shared.get("sources", []),
    }


@app.post("/chat/stream")
async def chat_stream(query: Query):
    """SSE streaming endpoint"""
    q = asyncio.Queue()

    shared = {
        "texts": {},
        "chunks": [],
        "embeddings": [],
        "query": query.question,
        "query_embedding": None,
        "retrieved_document": [],
        "queue": q,
    }

    asyncio.create_task(async_query_flow.run_async(shared))

    async def event_generator():
        while True:
            token = await q.get()
            if token is None:
                break
            yield f"data: {json.dumps({'token': token})}\n\n"

        yield f"data: {json.dumps({'sources': shared.get('sources', []), 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/db-clear")
async def clear_db():
    chroma_client.delete_collection("rag_docs")
    return {"message": "Database cleared successfully"}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Upload multiple PDF or TXT files and index them."""

    os.makedirs("data", exist_ok=True)

    results = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    for file in files:
        try:
            filename = os.path.basename(file.filename or "")

            if not filename.lower().endswith((".txt", ".pdf")):
                results.append(
                    {
                        "filename": filename,
                        "error": "Only .txt and .pdf files are supported",
                    }
                )
                continue

            path = os.path.join("data", filename)

            # Save file
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Extract text
            if filename.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = extract_text_from_pdf(path)

            if not text or not text.strip():
                results.append(
                    {
                        "filename": filename,
                        "error": "No text could be extracted from the file",
                    }
                )
                continue

            # Split text into chunks
            chunks = splitter.create_documents([text], metadatas=[{"source": filename}])

            # Embed and index chunks
            for i, chunk in enumerate(chunks):
                emb = get_embedding(chunk.page_content)

                collection.add(
                    ids=[f"{filename}_chunk_{i}"],
                    metadatas=[{"source": filename}],
                    documents=[chunk.page_content],
                    embeddings=[emb],
                )

            results.append({"filename": filename, "chunks": len(chunks)})

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

        finally:
            await file.close()

    return {"results": results}


# Serve the chat UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="127.0.1.2", port=8000)
