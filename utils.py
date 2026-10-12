import re
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from pypdf import PdfReader

load_dotenv()

# Sync client (CLI)
client = OpenAI(base_url="https://api.groq.com/openai/v1/", api_key=os.environ.get("GROQ_API_KEY"))
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1/", api_key=os.environ.get("OPENROUTER_API_KEY"))

# Async client (API streaming)
async_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1/", api_key=os.environ.get("GROQ_API_KEY"))


def call_llm(prompt):
    r = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    content = r.choices[0].message.content
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


async def call_llm_async(prompt, on_token=None):
    stream = on_token is not None
    r = await async_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
        stream=stream,
    )

    full = ""
    async for chunk in r:
        token = chunk.choices[0].delta.content or ""
        if token:
            token = re.sub(r"<think>.*?</think>", "", token, flags=re.DOTALL)
            full += token
            if on_token:
                await on_token(token)
    return full


def get_embedding(text):
    response = openrouter_client.embeddings.create(model="nvidia/nemotron-3-embed-1b:free", input=text)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
