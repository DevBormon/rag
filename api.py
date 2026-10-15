import os
import shutil
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from flow import vector_flow, async_query_flow
from nodes import chroma_client, collection
from utils import extract_text_from_pdf

shared = {
    "texts": {},
    "chunks": [],
    "embeddings": [],
    "query": "",
    "query_embedding": None,
    "retrieved_document": [],
    "queue": "",
    "collection_del_status": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ingest existing files on startup if DB is empty"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    if os.listdir("data") and collection.count() == 0:

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


class Query(BaseModel):
    question: str


@app.post("/chat")
async def chat(query: Query):
    """Non-streaming JSON endpoint"""
    from flow import query_flow

    shared["query"] = query.question

    query_flow().run(shared)
    return {
        "answer": shared.get("generated_answer"),
        "sources": shared.get("sources", []),
    }


@app.post("/chat/stream")
async def chat_stream(query: Query):
    """SSE streaming endpoint"""
    q = asyncio.Queue()

    shared["query"] = query.question
    shared["queue"] = q

    asyncio.create_task(async_query_flow().run_async(shared))

    async def event_generator():
        while True:
            token = await q.get()
            if token is None:
                break
            yield f"data: {json.dumps({'token': token})}\n\n"

        yield f"data: {json.dumps({'sources': shared.get('sources', []), 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/db-clear")
def clear_db():
    results = []
    try:
        # 1. Delete the Chroma collection
        try:
            chroma_client.delete_collection("rag_docs")
            shared["collection_del_status"] = True
            results.append({"msg": "Database cleared successfully"})
        except Exception as e:
            # Optional: Ignore if the collection doesn't exist yet
            print(f"Error DB Del : {e}")
            results.append({"error": 'No data exiest in database'})
            pass

        # 2. Safely remove the local 'data' folder (or file if it's a file)
        if os.path.exists("data"):
            for filename in os.listdir("data"):
                file_path = os.path.join("data", filename)

                # Check if it's a file or a symbolic link, then remove
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    results.append({"msg": "Data cleared successfully"})

    except Exception as e:
        # Catch any unexpected errors and return a clean HTTP 500 response
        results.append({"error": str(e)})


    return {"results": results}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Upload multiple PDF or TXT files and index them."""

    os.makedirs("data", exist_ok=True)

    results = []

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

            shared["texts"][filename] = text

            if shared["texts"]:
                vector_flow().run(shared)
                print(f"✅ Upload ingestion: {len(shared['chunks'])} chunks")

            results.append({"filename": filename, "chunks": len(shared["chunks"])})

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

        finally:
            await file.close()

    return {"results": results}


# Serve the chat UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)