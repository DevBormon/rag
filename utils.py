import numpy as np
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama"
)

def call_llm(prompt):    
    r = client.chat.completions.create(
        model="deepseek-r1:8b",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content

def get_embedding(text):
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
    
    response = client.embeddings.create(
        model="qwen3-embedding:8b",
        input=text
    )
    
    # Extract the embedding vector from the response
    embedding = response.data[0].embedding
    
    # Convert to numpy array for consistency with other embedding functions
    return np.array(embedding, dtype=np.float32)