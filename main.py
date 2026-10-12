import os
from flow import vector_flow, query_flow
from utils import extract_text_from_pdf
from nodes import collection

shared = {
    "texts": {},
    "chunks": [],
    "embeddings": [],
    "query": "",
    "query_embedding": None,
    "retrieved_document": [],
}


def ensure_ingested():

    if collection.count() == 0:
        for filename in os.listdir("data"):
            filepath = os.path.join("data", filename)
            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    shared["texts"][filename] = f.read()
            elif filename.endswith(".pdf"):
                shared["texts"][filename] = extract_text_from_pdf(filepath)

        if shared["texts"]:
            vector_flow().run(shared)
            print(f"✅ Ingested {len(shared['chunks'])} chunks")


def ask_question(question):
    shared["query"] = question
    query_flow().run(shared)
    return shared.get("generated_answer")


if __name__ == "__main__":
    ensure_ingested()
    while True:
        q = input("\n🧑 Question: ")
        if q.lower() in ("exit", "quit"):
            break
        print(f"\n🤖 {ask_question(q)}")
